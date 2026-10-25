"""Step 8 / job A6: ingest poison shards into the indexes and measure what that does.

Usage (repo root; PowerShell or bash):
    python scripts/index_poison.py --check-masking                     # all three indexes, shards from paths.yaml
    python scripts/index_poison.py --retriever dense --shards <DIR>    # one index, other shard folder
    python scripts/index_poison.py --work-copy --shards tests/fixtures --check-masking --check-append --check-idempotent
                                                                       # local verification on TEMPORARY index copies

Ingestion: BM25 is rebuilt with the shards (IDF depends on every document); dense and SPLADE keep their clean
rows bit for bit and only encode + append the poison passages (equal to a full build with the shards).
Idempotent: same shards -> no-op; changed shards -> re-ingested; no shards -> back to the clean index.
Before the first ingestion into a clean index, its top-150 for every question is saved as the clean reference
(<indexes_dir>/<env>/masking_reference_<retriever>.json) for the masking-equivalence check.
After ingesting into the real indexes, every top-150 cache is stale and must be rebuilt (scripts/build_cache.py).
Report: $IRE_DATA_DIR/reports/<A6|A6_local>.json. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.config.loader import Config  # noqa: E402
from ire.retrieval import bm25 as bm25_mod  # noqa: E402
from ire.retrieval.cache import CacheError, load_cache  # noqa: E402
from ire.retrieval.common import dir_size, find_poison_shards, index_dir_for, read_manifest  # noqa: E402
from ire.retrieval.poison import (  # noqa: E402
    PoisonShardError,
    load_reference,
    load_shards,
    masking_equivalence,
    poison_reach,
    save_reference,
    top_k_changed,
)
from ire.schemas import QuestionRecord, is_poison_id, make_poison_id, parse_poison_id, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402

RETRIEVERS = ("bm25", "dense", "splade")
EXACT_TOL = {"dense": 1e-5, "splade": 1e-5}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class Indexes:
    """Settings, encoders and fingerprints for the three retrievers of one config (encoders loaded once)."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.corpus = cfg.path("corpus")
        self._enc: dict[str, Any] = {}

    def encoder(self, name: str) -> Any:
        if name not in self._enc:
            if name == "dense":
                from ire.retrieval.dense import DenseSettings, E5Encoder

                self._enc[name] = E5Encoder.from_config(self.cfg, DenseSettings.from_config(self.cfg))
            else:
                from ire.retrieval.splade import SpladeEncoder

                self._enc[name] = SpladeEncoder.from_config(self.cfg)
        return self._enc[name]

    def settings(self, name: str) -> Any:
        if name == "bm25":
            return bm25_mod.BM25Settings.from_config(self.cfg)
        if name == "dense":
            from ire.retrieval.dense import DenseSettings

            return DenseSettings.from_config(self.cfg)
        return self.encoder("splade").settings

    def fingerprint(self, name: str, shards: list[Path], index_dir: Path | None = None) -> dict[str, Any]:
        if name == "bm25":
            return bm25_mod.fingerprint(self.corpus, shards, self.settings(name))
        if name == "dense":
            from ire.retrieval import dense

            return dense.fingerprint(self.corpus, shards, self.settings(name), self.encoder(name).revision)
        from ire.retrieval import splade

        return splade.fingerprint(self.corpus, shards, self.settings(name), self.encoder(name).revision)

    def open(self, name: str, index_dir: Path) -> Any:
        """A retriever with everything loaded into RAM (mmap=False): the folder may be replaced right after."""
        if name == "bm25":
            return bm25_mod.BM25Retriever(index_dir, self.settings(name), mmap=False)
        if name == "dense":
            from ire.retrieval.dense import DenseRetriever

            return DenseRetriever(index_dir, encoder=self.encoder(name), mmap=False)
        from ire.retrieval.splade import SpladeRetriever

        return SpladeRetriever(index_dir, encoder=self.encoder(name), mmap=False)

    def ingest(self, name: str, index_dir: Path, shards: list[Path], force: bool, log: Any) -> dict[str, Any]:
        cp = int(self.cfg[f"{'dense' if name == 'dense' else 'splade'}.checkpoint_every"]) if name != "bm25" else 0
        if name == "bm25":
            if not force and bm25_mod.index_is_current(index_dir, self.corpus, shards, self.settings(name)):
                return {"noop": True, "manifest": read_manifest(index_dir)}
            return {"noop": False, "manifest": bm25_mod.build_index(self.corpus, index_dir, shards, self.settings(name))}
        if name == "dense":
            from ire.retrieval import dense

            return dense.ingest_poison(index_dir, self.corpus, self.encoder(name), self.settings(name), shards, cp,
                                       force, log)
        from ire.retrieval import splade

        return splade.ingest_poison(index_dir, self.corpus, self.encoder(name), self.settings(name), shards, cp,
                                    force, log)


def search_all(retriever: Any, questions: list[QuestionRecord], depth: int) -> dict[str, list]:
    rows = retriever.search([q.question for q in questions], depth)
    return {q.qid: r for q, r in zip(questions, rows)}


def corpus_ids(corpus: Path) -> set[str]:
    with corpus.open("r", encoding="utf-8") as fh:
        return {json.loads(line)["doc_id"] for line in fh if line.strip()}


def make_clean_reference(idx: Indexes, name: str, index_dir: Path, clean_fp: dict[str, Any], is_clean: bool,
                         questions: list[QuestionRecord], q_sha: str, depth: int, tmp_root: Path, log: Any) -> dict:
    """Clean top-`depth` rows for every question, saved next to the index. If the index already holds poison (e.g.
    shards were present when A2-A4 ran), a clean copy is made in a temp folder first: dense / SPLADE by dropping the
    poison rows (no encoding), BM25 by a rebuild without shards."""
    tmp = None
    source = index_dir
    if not is_clean:
        tmp = tmp_root / f"clean_reference_{name}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        if name == "bm25":
            bm25_mod.build_index(idx.corpus, tmp, [], idx.settings(name))
        else:
            shutil.copytree(index_dir, tmp)
            idx.ingest(name, tmp, [], False, log)
        if read_manifest(tmp)["fingerprint"] != clean_fp:
            raise RuntimeError(f"{name}: could not rebuild a clean index for the masking reference")
        source = tmp
        log.info("%s: index already holds poison; built a temporary clean copy for the reference", name)
    retr = idx.open(name, source)
    ref = search_all(retr, questions, depth)
    del retr
    gc.collect()
    save_reference(index_dir, clean_fp, q_sha, ref)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    log.info("%s: saved the clean reference (top-%d for %d questions)", name, depth, len(questions))
    return ref


def check_sample(rows: dict[str, list], qids: list[str], n: int, seed: int) -> tuple[list[str], int]:
    """Every question with poison in its unmasked top-150 (seeded sample if more than n), filled up with seeded
    random other questions to n. Returns (sample, how many have poison)."""
    rng = random.Random(seed)
    with_poison = [q for q in qids if any(is_poison_id(d) for d, _ in rows[q])]
    pick = rng.sample(with_poison, min(n, len(with_poison)))
    rest = [q for q in qids if q not in set(pick)]
    pick += rng.sample(rest, min(n - len(pick), len(rest)))
    return sorted(pick, key=qids.index), sum(1 for q in pick if q in set(with_poison))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", default="all", choices=[*RETRIEVERS, "all"])
    parser.add_argument("--shards", type=Path, help="folder with poison_*.jsonl (default: poison_dir in paths.yaml)")
    parser.add_argument("--force", action="store_true", help="re-ingest even if the index already has these shards")
    parser.add_argument("--check-masking", action="store_true", help="masking-equivalence check (plan section 2.1)")
    parser.add_argument("--n-check", type=int, default=50, help="questions in the masking check")
    parser.add_argument("--work-copy", action="store_true",
                        help="copy the indexes to a temp folder and work there (the real indexes stay untouched)")
    parser.add_argument("--keep-work-copy", action="store_true", help="do not delete the temp copy at the end")
    parser.add_argument("--check-append", action="store_true",
                        help="(work copy) also build dense/SPLADE fully with the shards and compare with the append")
    parser.add_argument("--check-idempotent", action="store_true",
                        help="(work copy) re-run must be a no-op; ingesting no shards must restore the clean index")
    parser.add_argument("--report", help="report name (default: A6 on Ada, A6_local otherwise)")
    args = parser.parse_args()

    t_start = time.time()
    real_cfg = load_config()
    log = get_logger("index_poison")
    names = list(RETRIEVERS) if args.retriever == "all" else [args.retriever]
    shard_dir = (args.shards or real_cfg.path("poison_dir")).resolve()
    if not args.work_copy and (REPO_ROOT / "tests") in [shard_dir, *shard_dir.parents]:
        log.error("refusing to ingest test fixtures into the real indexes: add --work-copy")
        return 2
    if (args.check_append or args.check_idempotent) and not args.work_copy:
        log.error("--check-append / --check-idempotent change the index: use them with --work-copy")
        return 2
    seed = int(real_cfg["seed"])
    depth = int(real_cfg["top_k_cache"])
    data_dir = real_cfg.path("data_dir")
    job = args.report or ("A6" if real_cfg.env == "ada" else f"A6_{real_cfg.env}")

    real_dirs = {r: index_dir_for(real_cfg, r) for r in names}
    real_fp_before = {r: (read_manifest(d) or {}).get("fingerprint") for r, d in real_dirs.items()}
    work_root = None
    if args.work_copy:
        work_root = data_dir / "tmp" / f"poison_work_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for r, d in real_dirs.items():
            shutil.copytree(d, work_root / real_cfg.env / r)
        cfg = Config(real_cfg.env, real_cfg.values, {**real_cfg.paths, "indexes_dir": work_root})
        log.info("working on temporary copies under %s", work_root.as_posix())
    else:
        cfg = real_cfg
    idx = Indexes(cfg)

    qpath = cfg.path("questions")
    questions = read_jsonl(qpath, QuestionRecord)
    qids = [q.qid for q in questions]
    roles = {q.qid: q.role for q in questions}
    clean_qids = [q.qid for q in questions if q.role == "clean"]
    q_sha = sha256_file(qpath)
    shard_files = find_poison_shards(shard_dir)
    try:
        shard_set = load_shards(shard_files, roles, corpus_ids(cfg.path("corpus")))
    except PoisonShardError as exc:
        log.error(str(exc))
        return 1
    shards_summary = shard_set.summary(roles)
    id_checks = [is_poison_id(r.doc_id) and parse_poison_id(r.doc_id)[:2] == (r.qid, r.attack)
                 and make_poison_id(*parse_poison_id(r.doc_id)) == r.doc_id for r in shard_set.records]
    shards_summary["is_poison_recoverable_from_doc_id"] = f"{sum(id_checks)}/{len(id_checks)}"
    log.info("shards in %s: %s", shard_dir.as_posix(), shards_summary)

    results: dict[str, Any] = {}
    after_rows: dict[str, dict[str, list]] = {}
    references: dict[str, dict[str, list]] = {}
    for r in names:
        d = index_dir_for(cfg, r)
        size_before, m_before = dir_size(d), read_manifest(d)
        clean_fp = idx.fingerprint(r, [])
        ref = load_reference(d, clean_fp, q_sha)
        if ref is None:
            ref = make_clean_reference(idx, r, d, clean_fp, m_before["fingerprint"] == clean_fp, questions, q_sha,
                                       depth, data_dir / "tmp", log)
        references[r] = ref
        t0 = time.time()
        out = idx.ingest(r, d, shard_files, args.force, log)
        ingest_s = time.time() - t0
        m_after = read_manifest(d)
        retr = idx.open(r, d)
        rows = search_all(retr, questions, depth)
        del retr
        gc.collect()
        after_rows[r] = rows
        res: dict[str, Any] = {
            "noop": out["noop"], "ingest_time_s": round(ingest_s, 2),
            "index_bytes_before": size_before, "index_bytes_after": dir_size(d),
            "n_docs_before": m_before["n_docs"], "n_docs_after": m_after["n_docs"],
            "n_poison_before": m_before["n_poison"], "n_poison_after": m_after["n_poison"],
            "fingerprint_shards": m_after["fingerprint"]["shards"],
            "reach": poison_reach(rows, shard_set.records),
        }
        if ref is not None:
            res["top5_changed_frac"] = {"all": top_k_changed(rows, ref, qids), "clean_role": top_k_changed(rows, ref, clean_qids)}
        if args.check_masking:
            if ref is None:
                res["masking"] = {"error": "no clean reference (it is saved before the first ingestion into a clean "
                                           "index); rebuild the clean index or keep the reference file"}
            else:
                sample, n_with = check_sample(rows, qids, args.n_check, seed)
                res["masking"] = {**masking_equivalence(rows, ref, sample, EXACT_TOL.get(r, 1e-5)),
                                  "n_with_poison_in_top150": n_with}
        results[r] = res
        log.info("%s: %s", r, {k: v for k, v in res.items() if k != "reach"} | {"reach": res["reach"]["by_attack"]})

    extra: dict[str, Any] = {}
    if args.check_append:
        extra["append_equals_rebuild"] = {}
        for r in [n for n in names if n != "bm25"]:
            full = work_root / "full" / r
            if r == "dense":
                from ire.retrieval.dense import build_dense_index

                build_dense_index(idx.corpus, full, idx.encoder(r), idx.settings(r), shard_files,
                                  int(cfg["dense.checkpoint_every"]))
            else:
                from ire.retrieval.splade import build_splade_index

                build_splade_index(idx.corpus, full, idx.encoder(r), idx.settings(r), shard_files,
                                   int(cfg["splade.checkpoint_every"]))
            a, b = idx.open(r, index_dir_for(cfg, r)), idx.open(r, full)
            same_vectors = bool(np.array_equal(np.asarray(a.vectors), np.asarray(b.vectors))) if r == "dense" \
                else bool((a.docs != b.docs).nnz == 0)
            sample = qids[:50]
            same_results = a.search([q.question for q in questions[:50]], depth) == \
                b.search([q.question for q in questions[:50]], depth)
            extra["append_equals_rebuild"][r] = {
                "same_doc_ids": a.doc_ids == b.doc_ids, "same_vectors": same_vectors,
                "same_top150_for_50_questions": same_results, "n_questions": len(sample),
                "same_fingerprint": read_manifest(index_dir_for(cfg, r))["fingerprint"] == read_manifest(full)["fingerprint"]}
            del a, b
            gc.collect()
    if args.check_idempotent:
        extra["idempotent"] = {}
        for r in names:
            d = index_dir_for(cfg, r)
            rerun = idx.ingest(r, d, shard_files, False, log)
            restored = idx.ingest(r, d, [], False, log)
            m = read_manifest(d)
            real = idx.open(r, real_dirs[r])
            back = idx.open(r, d)
            if r == "dense":
                same = bool(np.array_equal(np.asarray(real.vectors), np.asarray(back.vectors)))
            elif r == "splade":
                same = bool((real.docs != back.docs).nnz == 0)
            else:
                same = real.search([q.question for q in questions[:50]], depth) == \
                    back.search([q.question for q in questions[:50]], depth)
            del real, back
            gc.collect()
            extra["idempotent"][r] = {"rerun_noop": rerun["noop"], "removal_ingested": not restored["noop"],
                                      "clean_fingerprint_restored": m["fingerprint"] == real_fp_before[r],
                                      "n_poison_after_removal": m["n_poison"],
                                      "same_as_real_clean_index": same}

    real_fp_after = {r: (read_manifest(d) or {}).get("fingerprint") for r, d in real_dirs.items()}
    real_unchanged = real_fp_after == real_fp_before
    caches_ok: dict[str, Any] = {}
    for r in names:
        try:
            load_cache([r], cfg=real_cfg)
            caches_ok[r] = "loads (current)"
        except CacheError as exc:
            caches_ok[r] = f"{type(exc).__name__}: {str(exc)[:120]}"

    checks = {
        "shards_valid": True,
        "poison_ids_recoverable": all(id_checks) and bool(id_checks),
        "some_poison_in_target_top10": any(res["reach"]["any_in_top10"] for res in results.values()),
        "n_poison_matches_shards": all(res["n_poison_after"] == len(shard_set.records) for res in results.values()),
    }
    if args.check_masking:
        for r in names:
            m = results[r].get("masking", {})
            if r == "bm25":
                checks["bm25_masking_reported"] = "rank_diff_max" in m
            else:
                checks[f"{r}_masking_exact"] = m.get("exact") is True
    if args.check_append:
        for r, v in extra["append_equals_rebuild"].items():
            checks[f"{r}_append_equals_rebuild"] = all(v[k] for k in ("same_doc_ids", "same_vectors",
                                                                        "same_top150_for_50_questions", "same_fingerprint"))
    if args.check_idempotent:
        for r, v in extra["idempotent"].items():
            checks[f"{r}_rerun_noop_and_removal_restores_clean"] = (
                v["rerun_noop"] and v["clean_fingerprint_restored"] and v["n_poison_after_removal"] == 0
                and v["same_as_real_clean_index"])
    if args.work_copy:
        checks["real_indexes_unchanged"] = real_unchanged
    ok = all(checks.values())

    report = {
        "job": job, "env": cfg.env, "ok": ok, "checks": checks, "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "work_copy": bool(args.work_copy), "shard_dir": shard_dir.name, "shards": shards_summary,
        "questions": {"n": len(questions), "n_clean_role": len(clean_qids), "sha256": q_sha},
        "depth": depth, "retrievers": results, **extra,
        "real_indexes": {"fingerprints_unchanged": real_unchanged, "caches": caches_ok},
        "libraries": library_versions("bm25s", "torch", "transformers", "faiss-cpu", "scipy", "numpy"),
        "runtime_s": round(time.time() - t_start, 1), "rerun": "python scripts/index_poison.py " + " ".join(sys.argv[1:]),
    }
    report_path = real_cfg.path("reports_dir") / f"{job}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if work_root is not None and not args.keep_work_copy:
        gc.collect()
        shutil.rmtree(work_root, ignore_errors=True)

    print(f"\n== poison ingestion env={cfg.env}{' (TEMPORARY COPIES)' if args.work_copy else ''}: "
          f"{shards_summary['n_records']} passages from {shards_summary['files']} by attack {shards_summary['by_attack']}, "
          f"{shards_summary['n_target_qids']} target qids; ids recoverable {shards_summary['is_poison_recoverable_from_doc_id']}")
    for r, res in results.items():
        mib = lambda b: round(b / 2**20, 2)  # noqa: E731
        print(f"{r:<7} {'no-op' if res['noop'] else 'ingested'} in {res['ingest_time_s']} s; docs {res['n_docs_before']} -> "
              f"{res['n_docs_after']}; size {mib(res['index_bytes_before'])} -> {mib(res['index_bytes_after'])} MiB; "
              f"top-5 changed {res.get('top5_changed_frac')}")
        print(f"        reach by attack: {res['reach']['by_attack']}")
        print(f"        ranks in own target's results: {res['reach']['ranks']}")
        if "masking" in res:
            print(f"        masking: {res['masking']}")
    for key in ("append_equals_rebuild", "idempotent"):
        if key in extra:
            print(f"{key}: {extra[key]}")
    print(f"real indexes unchanged: {real_unchanged}; real caches: {caches_ok}")
    if not args.work_copy and any(not res["noop"] for res in results.values()):
        print("NOTE: the indexes changed, so every top-150 cache is now stale: rebuild them with "
              "python scripts/build_cache.py --retriever all (the cache loader refuses the old ones).")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {report_path.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
