"""Step 6: check the RRF hybrid (BM25 + dense) on the local indexes. No index is built and there is no Ada job.

Usage (repo root):  python scripts/check_rrf.py
Needs the BM25 and dense indexes of the active environment (scripts/build_index.py --retriever bm25 / dense).
Report: $IRE_DATA_DIR/reports/rrf_<env>.json. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.retrieval.eval import questions_with_gold_in_index, recall_at_k, sample_eval_set  # noqa: E402
from ire.retrieval.rrf import RRFRetriever, fuse  # noqa: E402
from ire.schemas import QuestionAllRecord, QuestionRecord, is_poison_id, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402


def mask_demo() -> dict:
    """The toy example from tests/test_rrf.py: mask-then-fuse vs fuse-then-remove."""
    poison = "poison_q1_fact_1"
    bm25 = [(poison, 0), ("c1", 0), ("c2", 0), ("c3", 0), ("c4", 0)]
    dense = [("c1", 0), ("c3", 0), (poison, 0), ("c2", 0), ("c4", 0)]
    clean = [(d, s) for d, s in bm25 if not is_poison_id(d)], [(d, s) for d, s in dense if not is_poison_id(d)]
    mask_then_fuse = [d for d, _ in fuse(*clean)]
    fuse_then_remove = [d for d, _ in fuse(bm25, dense) if not is_poison_id(d)]
    return {"bm25": [d for d, _ in bm25], "dense": [d for d, _ in dense], "mask_then_fuse": mask_then_fuse,
            "fuse_then_remove": fuse_then_remove, "differ": mask_then_fuse != fuse_then_remove}


def main() -> int:
    t_start = time.time()
    cfg = load_config()
    log = get_logger("check_rrf")
    rrf = RRFRetriever.from_config(cfg)
    ks = [int(k) for k in cfg["retrieval_eval.ks"]]
    ids_in_index = set(rrf.components["bm25"].doc_ids)
    sets = {
        "placeholder_questions": questions_with_gold_in_index(read_jsonl(cfg.path("questions"), QuestionRecord),
                                                              ids_in_index),
        "questions_all_sample": sample_eval_set(questions_with_gold_in_index(
            read_jsonl(cfg.path("questions_all"), QuestionAllRecord), ids_in_index),
            int(cfg["retrieval_eval.n_questions"]), int(cfg["seed"])),
    }
    log.info("rrf env=%s k=%d depth=%d components=%s", cfg.env, rrf.k, rrf.depth, list(rrf.components))

    evals, timings, determinism = {}, {}, {}
    for name, items in sets.items():
        questions = [q for _, q, _ in items]
        gold = [g for _, _, g in items]
        t0 = time.perf_counter()
        comps = rrf.search_components(questions)
        t_comp = time.perf_counter() - t0
        per_q = []
        fused = []
        for i in range(len(questions)):
            t0 = time.perf_counter()
            fused.append(fuse(comps["bm25"][i], comps["dense"][i], rrf.k, rrf.depth))
            per_q.append(1e6 * (time.perf_counter() - t0))
        evals[name] = {"n_questions": len(items),
                       "bm25": recall_at_k(comps["bm25"], gold, ks), "dense": recall_at_k(comps["dense"], gold, ks),
                       "rrf": recall_at_k(fused, gold, ks)}
        overlap = [len({d for d, _ in a[:100]} & {d for d, _ in b[:100]}) for a, b in zip(comps["bm25"], comps["dense"])]
        evals[name]["mean_top100_overlap_bm25_dense"] = round(statistics.mean(overlap), 2)
        timings[name] = {"fuse_us_per_question_median": round(statistics.median(per_q), 1),
                         "fuse_us_per_question_mean": round(statistics.mean(per_q), 1),
                         "components_ms_per_question": round(1000 * t_comp / len(items), 2)}
        again = rrf.search(questions, rrf.depth)
        determinism[name] = again == fused == rrf.search(questions, rrf.depth)
        log.info("%s: %s | %s", name, evals[name], timings[name])

    ph = evals["placeholder_questions"]
    worse = min(ph["bm25"]["recall@100"], ph["dense"]["recall@100"])
    demo = mask_demo()
    checks = {
        "rrf_recall@100_ge_worse_component": ph["rrf"]["recall@100"] >= worse,
        "same_queries_twice_identical": all(determinism.values()),
        "mask_then_fuse_differs_from_fuse_then_remove": demo["differ"],
    }
    ok = all(checks.values())
    report = {"job": f"rrf_{cfg.env}", "env": cfg.env, "ok": ok, "checks": checks, "git_commit": git_commit(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rrf": {"k": rrf.k, "depth": rrf.depth, "components": list(rrf.components)}, "eval": evals,
              "fusion_timing": timings, "determinism": determinism, "mask_demo": demo,
              "runtime_s": round(time.time() - t_start, 1), "rerun": "python scripts/check_rrf.py"}
    out = cfg.path("reports_dir") / f"rrf_{cfg.env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n== rrf env={cfg.env} (k={rrf.k}, depth={rrf.depth}, BM25 + dense)")
    for name, ev in evals.items():
        print(f"{name} (n={ev['n_questions']}):")
        for r in ("bm25", "dense", "rrf"):
            print(f"  {r:<6} " + ", ".join(f"{k}={v}" for k, v in ev[r].items()))
        print(f"  mean |top-100 BM25 ∩ top-100 dense| = {ev['mean_top100_overlap_bm25_dense']}")
        print(f"  fusion: {timings[name]}")
    print(f"mask demo: {demo}")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {out.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
