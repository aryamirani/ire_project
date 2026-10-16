"""Steps 3-5 / jobs A2-A4: build a retriever index, then check it and write a report.

Usage (repo root; PowerShell or bash):
    python scripts/build_index.py --retriever bm25                  # build if needed, then evaluate
    python scripts/build_index.py --retriever bm25 --force          # rebuild even if the index is current
    python scripts/build_index.py --retriever bm25 --check-rebuild  # also rebuild a temp copy and compare results

The index goes to $IRE_DATA_DIR/indexes/<env>/<retriever>. Resumable: if the index exists and was built from the
same corpus (sha256), poison shards and settings, the build is skipped and only the checks run.
Report: $IRE_DATA_DIR/reports/<A2|A2_local>.json with sizes, timings, Recall@10/100, unique-token test,
determinism and edge-case checks. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.retrieval.bm25 import (  # noqa: E402
    BM25Retriever,
    BM25Settings,
    build_index,
    default_index_dir,
    dir_size,
    find_poison_shards,
    index_is_current,
    read_manifest,
    unique_token_test,
)
from ire.retrieval.eval import questions_with_gold_in_index, recall_at_k, sample_eval_set  # noqa: E402
from ire.schemas import QuestionAllRecord, QuestionRecord, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402

JOB_NAMES = {"bm25": "A2", "dense": "A3", "splade": "A4"}
EDGE_QUERIES = ["", "the of and is", "qwertzuiopasdfghjk"]  # empty, only stopwords, unknown word


def evaluate(retriever: BM25Retriever, items: list[tuple[str, str, list[str]]], ks: list[int]) -> tuple[dict, list]:
    """Recall/hit@k for (qid, question, gold) items; returns metrics (+ n_questions) and the raw results."""
    if not items:
        return {"n_questions": 0}, []
    t0 = time.perf_counter()
    results = retriever.search([q for _, q, _ in items], k=max(ks))
    elapsed = time.perf_counter() - t0
    metrics = {"n_questions": len(items), **recall_at_k(results, [g for _, _, g in items], ks),
               "batch_ms_per_query": round(1000 * elapsed / len(items), 3)}
    return metrics, results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", required=True, choices=["bm25", "dense", "splade"])
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    parser.add_argument("--no-poison-shards", action="store_true", help="ignore poison_*.jsonl shards")
    parser.add_argument("--check-rebuild", action="store_true",
                        help="build a second index in a temp folder and check its results are identical")
    parser.add_argument("--report", help="report name (default: A2 on Ada, A2_local otherwise)")
    args = parser.parse_args()
    if args.retriever != "bm25":
        print(f"--retriever {args.retriever} is not implemented yet (step {4 if args.retriever == 'dense' else 5}).")
        return 2

    t_start = time.time()
    cfg = load_config()
    log = get_logger("build_index")
    data_dir = cfg.path("data_dir")
    settings = BM25Settings.from_config(cfg)
    corpus = cfg.path("corpus")
    use_shards = cfg["bm25.include_poison_shards"] and not args.no_poison_shards
    shards = find_poison_shards(cfg.path("poison_dir")) if use_shards else []
    index_dir = default_index_dir(cfg)
    job = args.report or JOB_NAMES[args.retriever] + ("" if cfg.env == "ada" else f"_{cfg.env}")
    log.info("bm25 env=%s corpus=%s poison shards=%d (%s) index=%s", cfg.env, corpus.name, len(shards),
             ", ".join(s.name for s in shards) or "none", index_dir)

    # ---- build (skipped if current) ----
    if not args.force and index_is_current(index_dir, corpus, shards, settings):
        log.info("index is current (same corpus sha256, shards and settings): skipping the build (--force rebuilds)")
        built = False
    else:
        manifest = build_index(corpus, index_dir, shards, settings)
        log.info("built: %d docs (%d poison) in %.1f s", manifest["n_docs"], manifest["n_poison"],
                 manifest["timings_s"]["total"])
        built = True
    manifest = read_manifest(index_dir)
    index_bytes = dir_size(index_dir)

    t0 = time.perf_counter()
    retriever = BM25Retriever(index_dir, settings)
    load_s = time.perf_counter() - t0
    ids_in_index = set(retriever.doc_ids)
    corpus_rows = sum(1 for _ in corpus.open("rb"))

    # ---- recall ----
    ks = [int(k) for k in cfg["retrieval_eval.ks"]]
    placeholder = questions_with_gold_in_index(read_jsonl(cfg.path("questions"), QuestionRecord), ids_in_index)
    pool_all = questions_with_gold_in_index(read_jsonl(cfg.path("questions_all"), QuestionAllRecord), ids_in_index)
    pool = sample_eval_set(pool_all, int(cfg["retrieval_eval.n_questions"]), int(cfg["seed"]))
    eval_placeholder, _ = evaluate(retriever, placeholder, ks)
    eval_pool, pool_results = evaluate(retriever, pool, ks)
    eval_pool["n_available"] = len(pool_all)
    log.info("recall placeholder: %s | pool: %s", eval_placeholder, eval_pool)

    # ---- latency, determinism, edge cases ----
    singles = []
    for _, q, _ in pool[:50]:
        t0 = time.perf_counter()
        retriever.search([q], k=100)
        singles.append(1000 * (time.perf_counter() - t0))
    latency = {"single_query_median_ms": round(statistics.median(singles), 3) if singles else None,
               "batch_ms_per_query": eval_pool.get("batch_ms_per_query")}
    sample_q = [q for _, q, _ in pool[:50]]
    same_twice = retriever.search(sample_q, k=100) == retriever.search(sample_q, k=100)
    edge = {}
    for q in EDGE_QUERIES:
        try:
            edge[repr(q)] = {"hits": len(retriever.search([q], k=10)[0]), "error": None}
        except Exception as exc:  # noqa: BLE001 - recorded as a failed check
            edge[repr(q)] = {"hits": None, "error": f"{type(exc).__name__}: {exc}"}
    edge_ok = all(v["error"] is None and v["hits"] == 0 for v in edge.values())

    tmp_root = data_dir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    rebuild_identical = None
    if args.check_rebuild:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            build_index(corpus, Path(tmp) / "bm25_rebuild", shards, settings)
            again = BM25Retriever(Path(tmp) / "bm25_rebuild", settings, mmap=False).search(
                [q for _, q, _ in pool], k=max(ks))
            rebuild_identical = again == pool_results
            del again
        log.info("rebuild gives identical top-%d results for %d queries: %s", max(ks), len(pool), rebuild_identical)

    # ---- unique-token test on a throw-away index ----
    cap = int(cfg["bm25.unique_token_test_passages"])
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        unique = unique_token_test(corpus, Path(tmp), settings, max_docs=cap if corpus_rows > cap else None)
    log.info("unique-token test: %s", unique)

    checks = {
        "n_docs_matches_corpus_plus_poison": manifest["n_docs"] == corpus_rows + manifest["n_poison"],
        "recall_reported": "recall@100" in eval_pool,
        "same_query_twice_identical": same_twice,
        "edge_queries_no_crash_no_hits": edge_ok,
        "unique_token_rank_1": unique["passed"],
    }
    if rebuild_identical is not None:
        checks["rebuild_identical"] = rebuild_identical
    ok = all(checks.values())

    report = {
        "job": job, "env": cfg.env, "retriever": "bm25", "ok": ok, "checks": checks,
        "git_commit": git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index_dir": index_dir.relative_to(data_dir).as_posix(), "built_now": built,
        "build_time_s": manifest["timings_s"]["total"], "build_timings_s": manifest["timings_s"],
        "index_bytes": index_bytes, "index_mib": round(index_bytes / 2**20, 2), "load_time_s": round(load_s, 3),
        "corpus": {"path": corpus.relative_to(data_dir).as_posix(), "rows": corpus_rows,
                   "sha256": manifest["fingerprint"]["corpus_sha256"]},
        "n_docs": manifest["n_docs"], "n_clean": manifest["n_clean"], "n_poison": manifest["n_poison"],
        "n_vocab": manifest["n_vocab"], "poison_shards": [s.name for s in shards], "settings": manifest["settings"],
        "eval": {"ks": ks, "placeholder_questions": eval_placeholder, "questions_all_sample": eval_pool},
        "latency_ms": latency,
        "determinism": {"same_query_twice": same_twice, "rebuild_identical": rebuild_identical},
        "edge_queries": edge, "unique_token": unique,
        "libraries": library_versions("bm25s", "PyStemmer", "numpy", "scipy"),
        "runtime_s": round(time.time() - t_start, 1),
        "rerun": "python scripts/build_index.py " + " ".join(sys.argv[1:]),
    }
    report_path = cfg.path("reports_dir") / f"{job}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n== bm25 env={cfg.env}: {manifest['n_docs']:,} docs ({manifest['n_poison']} poison), "
          f"vocab {manifest['n_vocab']:,}, index {report['index_mib']} MiB, build {manifest['timings_s']['total']} s"
          f"{'' if built else ' (earlier build)'}, load {report['load_time_s']} s")
    for name, ev in (("placeholder questions", eval_placeholder), ("questions_all (gold in index)", eval_pool)):
        print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in ev.items()))
    print(f"latency: {latency}")
    print(f"unique-token test: rank {unique['rank']} in a {unique['n_docs_in_test_index']:,}-doc temp index")
    print(f"edge queries: {edge}")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {report_path.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
