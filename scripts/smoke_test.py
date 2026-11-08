"""Step 13 / job A9: end-to-end smoke test on 20 questions for each of the 4 retrievers.

Usage (repo root; needs current caches, the dense index, the perplexity thresholds + score cache, the LLM):
    python scripts/smoke_test.py                    # report A9 on Ada, A9_local otherwise
    python scripts/smoke_test.py --n-questions 20 --show 2

For 20 seeded questions (10 target + 10 clean) and each retriever (bm25, dense, splade, rrf):
    cached top-150 -> candidate_pool STUB (first 50, no masking; src/ire/stubs.py)
    -> query-and-phrase, cluster and perplexity filters (each run on the pool; anchors from the config)
    -> first 5 passages flagged by none of them -> generate (cached, batched)
and prints question, top-5 ids, flagged ids per filter and the answer for the first --show questions per
retriever. The report records per retriever and filter: flagged counts, poison flagged, runtimes, answers, errors.
This checks that everything runs end to end; the numbers are not results (no masking, no sweep).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.textstore import TextStore  # noqa: E402
from ire.defenses import cluster, perplexity, query_phrase  # noqa: E402
from ire.generation.llm import Generator  # noqa: E402
from ire.retrieval.cache import load_cache  # noqa: E402
from ire.schemas import FilterContext, QuestionRecord, is_poison_id, read_jsonl  # noqa: E402
from ire.stubs import candidate_pool  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402
from ire.vectors import default_vectors  # noqa: E402

RETRIEVERS = ("bm25", "dense", "splade", "rrf")
FILTERS = ("query_phrase", "cluster", "perplexity")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--show", type=int, default=2, help="questions printed in full per retriever")
    ap.add_argument("--report", help="report name (default: A9 on Ada, A9_local otherwise)")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    seed, size, final_k = int(cfg["seed"]), int(cfg["pool_size"]), int(cfg["final_k"])
    qs = read_jsonl(cfg.path("questions"), QuestionRecord)
    rng = random.Random(f"{seed}:smoke")
    half = args.n_questions // 2
    chosen = (rng.sample([q for q in qs if q.role == "target"], half)
              + rng.sample([q for q in qs if q.role == "clean"], args.n_questions - half))
    cache = load_cache(["bm25", "dense", "splade"])
    texts = TextStore.open_for(cfg)
    vectors = default_vectors()
    qp_table = query_phrase.load_thresholds(cfg.thresholds_path("query_phrase_thresholds"), cfg)
    ppl = perplexity.default_scores(cfg)
    ppl_table = perplexity.load_thresholds(cfg.thresholds_path("perplexity_thresholds"))
    gen = Generator.from_config(cfg, texts=texts)

    report_r: dict[str, dict] = {}
    errors: list[str] = []
    for r in RETRIEVERS:
        rows, times = [], {f: [] for f in FILTERS}
        for q in chosen:
            try:
                pool = candidate_pool(q.qid, r, None, None, cache, size, cfg)
                th = {"query_phrase": query_phrase.thresholds_for(qp_table, r, q.fold),
                      "cluster": {"tau_c": float(cfg["cluster.tau_c"])},
                      "perplexity": {"alpha": float(cfg["perplexity.alpha"]), "table": ppl_table, "scores": ppl}}
                mods = {"query_phrase": query_phrase, "cluster": cluster, "perplexity": perplexity}
                flagged = {}
                for f in FILTERS:
                    ctx = FilterContext(texts=texts, vectors=vectors, config=cfg, thresholds=th[f])
                    t = time.perf_counter()
                    flagged[f] = sorted(mods[f].apply_filter(pool, q.qid, q.question, ctx).flagged)
                    times[f].append(time.perf_counter() - t)
                union = set().union(*map(set, flagged.values()))
                top5 = [d for d in pool if d not in union][:final_k]
                rows.append({"qid": q.qid, "role": q.role, "question": q.question, "pool": len(pool),
                             "poison_in_pool": sum(is_poison_id(d) for d in pool), "flagged": flagged,
                             "top5": top5, "n_poison_top5": sum(is_poison_id(d) for d in top5)})
            except Exception as exc:  # noqa: BLE001 - recorded, the smoke test reports every failure
                errors.append(f"{r}/{q.qid}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        t = time.perf_counter()
        answers = gen.generate_batch([(row["qid"], row["question"], row["top5"]) for row in rows])
        gen_s = time.perf_counter() - t
        for row, a in zip(rows, answers):
            row["answer"], row["cached"] = a["answer"], a["cached"]
        report_r[r] = {
            "questions": len(rows),
            "answers_nonempty": sum(bool(row["answer"].strip()) for row in rows),
            "flagged_per_filter": {f: sum(len(row["flagged"][f]) for row in rows) for f in FILTERS},
            "poison_flagged_per_filter": {f: sum(sum(is_poison_id(d) for d in row["flagged"][f]) for row in rows)
                                          for f in FILTERS},
            "poison_in_pools": sum(row["poison_in_pool"] for row in rows),
            "poison_in_top5": sum(row["n_poison_top5"] for row in rows),
            "filter_ms_mean": {f: round(1000 * statistics.mean(v), 2) if v else None for f, v in times.items()},
            "generation_s": round(gen_s, 1), "generated_now": sum(not row["cached"] for row in rows),
            "examples": rows[: args.show],
        }
        print(f"\n=== {r}: {len(rows)} questions, flagged {report_r[r]['flagged_per_filter']}, "
              f"poison flagged {report_r[r]['poison_flagged_per_filter']}")
        for row in rows[: args.show]:
            print(f"  [{row['role']}] {row['qid']}: {row['question']}")
            print(f"    top5: {row['top5']}")
            print(f"    flagged: { {f: v[:6] for f, v in row['flagged'].items()} }")
            print(f"    answer: {row['answer'][:160]!r}")

    n = args.n_questions
    checks = {
        "all_four_retrievers_ran": all(report_r[r]["questions"] == n for r in RETRIEVERS),
        "all_filters_ran": all(all(v is not None for v in report_r[r]["filter_ms_mean"].values()) for r in RETRIEVERS),
        "answers_nonempty": all(report_r[r]["answers_nonempty"] == n for r in RETRIEVERS),
        "no_errors": not errors,
    }
    ok = all(checks.values())
    report = {"job": args.report or ("A9" if cfg.env == "ada" else f"A9_{cfg.env}"), "env": cfg.env,
              "git_commit": git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "ok": ok, "checks": checks, "n_questions": n, "retrievers": report_r, "filters": list(FILTERS),
              "llm": cfg["models.llm"], "llm_revision": gen.revision, "errors": errors[:50],
              "candidate_pool": "STUB (first 50 cached ids, no masking)", "runtime_s": round(time.time() - t0, 1),
              "rerun": "python scripts/smoke_test.py"}
    out = cfg.path("reports_dir") / f"{report['job']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
