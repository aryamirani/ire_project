"""Step 7 / job A5: build the top-150 retrieval caches and check them.

Usage (repo root; PowerShell or bash):
    python scripts/build_cache.py --retriever all                    # bm25, dense, splade, then rrf
    python scripts/build_cache.py --retriever dense --force          # rebuild one cache
    python scripts/build_cache.py --retriever all --questions <path> # e.g. Arya's final question set

Output in <cache_dir>/<env>/: top150_<retriever>.jsonl (one RetrievalCacheRecord per question) and a sidecar
top150_<retriever>.jsonl.manifest.json (cache key: questions sha256, depth, index fingerprint; build info).
top150_rrf.jsonl is fused from the cached bm25 and dense rows and is an UNMASKED, NO-POISON REFERENCE only.

Resumable: an interrupted build continues after the last written question; a cache whose key matches the current
index and questions is not rebuilt (--force rebuilds). Each cache is then spot-checked against fresh searches.
Report: $IRE_DATA_DIR/reports/<A5|A5_local>.json. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import random
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
from ire.retrieval.cache import (  # noqa: E402
    ALL_RETRIEVERS,
    RRF,
    CacheError,
    cache_dir_for,
    cache_path,
    current_key,
    fuse_rows,
    load_cache,
    rrf_component_rows,
    write_cache,
)
from ire.retrieval.common import index_dir_for, read_manifest  # noqa: E402
from ire.retrieval.rrf import fuse  # noqa: E402
from ire.schemas import QuestionRecord, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402


def open_retriever(cfg: Any, name: str) -> Any:
    if name == "bm25":
        from ire.retrieval.bm25 import BM25Retriever

        return BM25Retriever.from_config(cfg)
    if name == "dense":
        from ire.retrieval.dense import DenseRetriever

        return DenseRetriever.from_config(cfg)
    from ire.retrieval.splade import SpladeRetriever

    return SpladeRetriever.from_config(cfg)


def spot_check(name: str, cfg: Any, rows: dict[str, list], questions: list[QuestionRecord], n: int, seed: int,
               searcher: Any = None, components: dict[str, dict[str, list]] | None = None) -> dict[str, Any]:
    """n seeded random questions: cached row == fresh search() (ids, order, scores) and == score_docs (bm25/splade
    exact, dense within float tolerance); for rrf: cached row == fuse() of the cached component rows."""
    sample = random.Random(seed).sample(questions, min(n, len(questions)))
    depth = int(cfg["top_k_cache"])
    ids_equal = scores_equal = sd_ok = 0
    max_sd_diff = 0.0
    for q in sample:
        cached = rows[q.qid]
        if name == RRF:
            fresh = fuse(components["bm25"][q.qid], components["dense"][q.qid], int(cfg["rrf.k"]), int(cfg["rrf.depth"]))
        else:
            fresh = searcher.search([q.question], depth)[0]
        ids_equal += [d for d, _ in cached] == [d for d, _ in fresh]
        scores_equal += [s for _, s in cached] == [s for _, s in fresh]
        if name == RRF:
            sd_ok += 1
            continue
        sd = np.asarray(searcher.score_docs(q.question, [d for d, _ in cached]), dtype=np.float64)
        got = np.asarray([s for _, s in cached], dtype=np.float64)
        diff = float(np.max(np.abs(sd - got))) if len(got) else 0.0
        max_sd_diff = max(max_sd_diff, diff)
        sd_ok += bool(np.array_equal(sd.astype(np.float32), got.astype(np.float32))) if name != "dense" \
            else bool(np.allclose(sd, got, rtol=1e-5, atol=1e-6))
    n_s = len(sample)
    return {"n": n_s, "ids_and_order_equal": ids_equal, "scores_equal_fresh_search": scores_equal,
            "score_docs_agree": sd_ok, "max_abs_diff_score_docs": max_sd_diff,
            "score_docs_rule": "fuse() of cached components" if name == RRF else
            ("float tolerance (rtol 1e-5)" if name == "dense" else "exact (float32)"),
            "passed": ids_equal == n_s and scores_equal == n_s and sd_ok == n_s}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", required=True, choices=[*ALL_RETRIEVERS, "all"])
    parser.add_argument("--questions", type=Path, help="questions file (default: questions.jsonl from paths.yaml)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is current")
    parser.add_argument("--spot-check", type=int, default=20, help="random questions re-searched per retriever")
    parser.add_argument("--batch-size", type=int, default=32, help="questions per search call")
    parser.add_argument("--report", help="report name (default: A5 on Ada, A5_local otherwise)")
    args = parser.parse_args()

    t_start = time.time()
    cfg = load_config()
    log = get_logger("build_cache")
    qpath = Path(args.questions or cfg.path("questions")).resolve()
    questions = read_jsonl(qpath, QuestionRecord)
    depth = int(cfg["top_k_cache"])
    seed = int(cfg["seed"])
    cdir = cache_dir_for(cfg)
    names = list(ALL_RETRIEVERS) if args.retriever == "all" else [args.retriever]
    job = args.report or ("A5" if cfg.env == "ada" else f"A5_{cfg.env}")
    data_dir = cfg.path("data_dir")
    qfile = qpath.relative_to(data_dir).as_posix() if data_dir in qpath.parents else qpath.as_posix()
    log.info("cache env=%s questions=%s (%d) retrievers=%s depth=%d dir=%s", cfg.env, qfile, len(questions), names,
             depth, cdir)
    common_side = {"questions_file": qfile, "git_commit": git_commit(),
                   "libraries": library_versions("numpy", "scipy", "bm25s", "faiss-cpu", "torch", "transformers")}

    results: dict[str, Any] = {}
    for name in names:
        out = cache_path(cdir, name)
        try:
            key = current_key(cfg, name, qpath)
        except CacheError as exc:
            log.error("%s: %s", name, exc)
            results[name] = {"error": str(exc)}
            continue
        searcher = None
        components = None
        if name == RRF:
            try:  # rrf is fused from the cached components: they must exist and be current
                components = rrf_component_rows(cfg, qpath, [q.qid for q in questions])
            except CacheError as exc:
                log.error(str(exc))
                results[name] = {"error": str(exc)}
                continue
            k, rdepth = int(cfg["rrf.k"]), int(cfg["rrf.depth"])
            compute = lambda batch, c=components, k=k, d=rdepth: fuse_rows(c["bm25"], c["dense"], batch, k, d)  # noqa: E731
            exact_len = None
            index_info = {c: read_manifest(index_dir_for(cfg, c)) for c in cfg["rrf.components"]}
            n_poison = {c: m["n_poison"] for c, m in index_info.items()}
        else:
            searcher = open_retriever(cfg, name)
            compute = lambda batch, s=searcher: s.search([q.question for q in batch], depth)  # noqa: E731
            exact_len = min(depth, searcher.n_docs) if name == "dense" else None
            n_poison = read_manifest(index_dir_for(cfg, name))["n_poison"]

        side = write_cache(out, key, questions, compute, depth, exact_len=exact_len, batch_size=args.batch_size,
                           force=args.force, extra_sidecar={**common_side, "index_n_poison": n_poison}, log=log)

        t0 = time.perf_counter()
        rows = load_cache([name], allow_unmasked_reference=(name == RRF), cfg=cfg, questions_path=qpath)[name]
        load_s = time.perf_counter() - t0
        if searcher is None and name != RRF:
            searcher = open_retriever(cfg, name)
        spot = spot_check(name, cfg, rows, questions, args.spot_check, seed, searcher, components)
        all_sorted = all([(-s, d) for d, s in r] == sorted((-s, d) for d, s in r) for r in rows.values())
        results[name] = {
            "built_now": not side["skipped"], "build_time_s": side["build_time_s"],
            "resumed_from_question": side["resumed_from_question"], "rows": len(rows),
            "n_questions": len(questions), "row_length": side["row_length"], "all_rows_sorted": all_sorted,
            "dense_rows_exactly_depth": (side["row_length"]["min"] == side["row_length"]["max"] == exact_len)
            if name == "dense" else None,
            "spot_check": spot, "fingerprint_matches_index": True,  # load_cache checked it (raises otherwise)
            "index_n_poison": n_poison, "unmasked_reference": side["unmasked_reference"],
            "file_bytes": out.stat().st_size, "file_mib": round(out.stat().st_size / 2**20, 3),
            "load_time_s": round(load_s, 3), "key": side["key"],
        }
        log.info("%s: %s", name, {k: v for k, v in results[name].items() if k != "key"})

    t0 = time.perf_counter()
    loadable = [n for n in names if "error" not in results[n]]
    if loadable:
        load_cache(loadable, allow_unmasked_reference=RRF in loadable, cfg=cfg, questions_path=qpath)
    total_load_s = time.perf_counter() - t0

    checks = {}
    for name, r in results.items():
        if "error" in r:
            checks[f"{name}_built"] = False
            continue
        checks[f"{name}_rows_equal_questions"] = r["rows"] == r["n_questions"]
        checks[f"{name}_rows_sorted"] = r["all_rows_sorted"]
        checks[f"{name}_row_length_ok"] = r["dense_rows_exactly_depth"] if name == "dense" else r["row_length"]["max"] <= depth
        checks[f"{name}_spot_check"] = r["spot_check"]["passed"]
    ok = all(checks.values())
    report = {"job": job, "env": cfg.env, "ok": ok, "checks": checks, "git_commit": common_side["git_commit"],
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "questions": {"path": qfile, "n": len(questions)}, "depth": depth,
              "rrf": {"k": int(cfg["rrf.k"]), "depth": int(cfg["rrf.depth"])}, "retrievers": results,
              "total_load_time_s": round(total_load_s, 3), "libraries": common_side["libraries"],
              "runtime_s": round(time.time() - t_start, 1),
              "rerun": "python scripts/build_cache.py " + " ".join(sys.argv[1:])}
    report_path = cfg.path("reports_dir") / f"{job}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n== cache env={cfg.env}: {len(questions)} questions ({qfile}), depth {depth}, dir {cdir.as_posix()}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<7} ERROR: {r['error']}")
            continue
        rl = r["row_length"]
        print(f"{name:<7} rows {r['rows']}, length min/median/max {rl['min']}/{rl['median']}/{rl['max']} "
              f"(short {rl['short_rows']}), {r['file_mib']} MiB, build {r['build_time_s']} s"
              f"{'' if r['built_now'] else ' (earlier build)'}, load {r['load_time_s']} s, spot check "
              f"{r['spot_check']['ids_and_order_equal']}/{r['spot_check']['n']} ids, "
              f"{r['spot_check']['score_docs_agree']}/{r['spot_check']['n']} scores, poison in index {r['index_n_poison']}")
    print(f"load all ({', '.join(loadable)}): {round(total_load_s, 3)} s")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {report_path.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
