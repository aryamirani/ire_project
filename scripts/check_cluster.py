"""Step 10 verification: the cluster filter on the real caches (IMPLEMENTATION_PLAN step 10, Verify).

Usage (repo root; needs current top-150 caches, the dense index and the poison shards):
    python scripts/check_cluster.py [--tau-c 0.85]

Reports, per retriever (bm25 / dense / splade / rrf):
- clean side: on the 200 CLEAN questions with every poison passage masked (the no-poison setting), the fraction
  of questions where anything is flagged and the mean number of clean passages flagged (false positives);
- poison side: for each attack in the shards at rate 5 (that attack's passages n <= 5 active, everything else
  masked), on the TARGET questions whose pool holds poison: detection rate (poison flagged / poison in the
  top-20) and clean passages flagged;
- backfill: fraction of questions whose first 5 survivors include a passage from beyond rank 20;
- runtime per question.
The masking here is a local stand-in for Vansh's candidate_pool (same rule: rate r keeps n <= r).
Writes $IRE_DATA_DIR/reports/cluster_<env>.json. A sanity check, not the reported metric.
"""

from __future__ import annotations

import argparse
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
from ire.data.textstore import TextStore  # noqa: E402
from ire.defenses.cluster import apply_filter, backfill  # noqa: E402
from ire.retrieval.cache import load_cache  # noqa: E402
from ire.retrieval.rrf import fuse  # noqa: E402
from ire.schemas import FilterContext, QuestionRecord, is_poison_id, parse_poison_id, read_jsonl  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402
from ire.vectors import default_vectors  # noqa: E402

BASE = ("bm25", "dense", "splade")


def pools(cache, cfg, qid, keep, size):
    m = {r: [(d, s) for d, s in cache[r][qid] if not is_poison_id(d) or keep(d)] for r in BASE}
    out = {r: [d for d, _ in m[r]][:size] for r in BASE}
    a, b = cfg["rrf.components"]
    out["rrf"] = [d for d, _ in fuse(m[a], m[b], k=int(cfg["rrf.k"]), depth=int(cfg["rrf.depth"]))][:size]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau-c", type=float, help="default: cluster.tau_c (anchor)")
    args = ap.parse_args()
    cfg = load_config()
    tau = float(args.tau_c if args.tau_c is not None else cfg["cluster.tau_c"])
    size, top_n = int(cfg["pool_size"]), int(cfg["cluster.top_n"])
    qs = read_jsonl(cfg.path("questions"), QuestionRecord)
    cache = load_cache(list(BASE))
    ctx = FilterContext(texts=TextStore.open_for(cfg), vectors=default_vectors(), config=cfg,
                        thresholds={"tau_c": tau})
    retrievers = [*BASE, "rrf"]
    times: list[float] = []

    def run(pool, q):
        t = time.perf_counter()
        res = apply_filter(pool, q.qid, q.question, ctx)
        times.append(time.perf_counter() - t)
        return res.flagged

    clean = {}
    for r in retrievers:
        any_flag = n_flag = bf = 0
        cq = [q for q in qs if q.role == "clean"]
        for q in cq:
            pool = pools(cache, cfg, q.qid, lambda d: False, size)[r]
            fl = run(pool, q)
            any_flag += bool(fl)
            n_flag += len(fl)
            bf += backfill(pool, fl, int(cfg["final_k"]), top_n)
        clean[r] = {"questions": len(cq), "frac_questions_any_flagged": round(any_flag / len(cq), 4),
                    "mean_clean_flagged": round(n_flag / len(cq), 3), "backfill_rate": round(bf / len(cq), 4)}

    attacks = sorted({parse_poison_id(d)[1] for d in ctx.texts if is_poison_id(d)})
    poison = {}
    for attack in attacks:
        keep = lambda d, a=attack: parse_poison_id(d)[1] == a and parse_poison_id(d)[2] <= 5  # noqa: E731
        poison[attack] = {}
        for r in retrievers:
            tp = n_p = fp = n_c = n_q = bf = 0
            for q in (q for q in qs if q.role == "target"):
                pool = pools(cache, cfg, q.qid, keep, size)[r]
                head = pool[:top_n]
                if not any(is_poison_id(d) for d in head):
                    continue
                n_q += 1
                fl = run(pool, q)
                tp += sum(1 for d in head if is_poison_id(d) and d in fl)
                n_p += sum(1 for d in head if is_poison_id(d))
                fp += sum(1 for d in head if not is_poison_id(d) and d in fl)
                n_c += sum(1 for d in head if not is_poison_id(d))
                bf += backfill(pool, fl, int(cfg["final_k"]), top_n)
            poison[attack][r] = {"questions_with_poison_in_top20": n_q,
                                 "detection_rate": round(tp / max(n_p, 1), 4),
                                 "clean_flag_rate_top20": round(fp / max(n_c, 1), 4),
                                 "backfill_rate": round(bf / max(n_q, 1), 4)}

    report = {"job": "cluster", "env": cfg.env, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": git_commit(), "tau_c": tau, "rouge_l_min": cfg["cluster.rouge_l_min"],
              "clean_no_poison": clean, "poison_rate5": poison,
              "runtime_ms_per_question": {"mean": round(1000 * statistics.mean(times), 2),
                                          "median": round(1000 * statistics.median(times), 2),
                                          "n": len(times)}}
    out = cfg.path("reports_dir") / f"cluster_{cfg.env}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"tau_c {tau}; clean questions, nothing poisoned (frac with any flag / mean clean flagged / backfill):")
    for r, v in clean.items():
        print(f"  {r:<7} {v['frac_questions_any_flagged']:.3f} / {v['mean_clean_flagged']:.2f} / {v['backfill_rate']:.3f}")
    print("rate 5, target questions (detection rate / clean flag rate in top-20):")
    for a, per in poison.items():
        print(f"  {a:<8} " + "  ".join(f"{r}: {v['detection_rate']:.2f}/{v['clean_flag_rate_top20']:.3f}"
                                      for r, v in per.items()))
    print(f"runtime per question: {report['runtime_ms_per_question']}")
    print(f"REPORT: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
