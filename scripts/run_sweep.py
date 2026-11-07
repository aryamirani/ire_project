"""Run the sweep (or the pilot): setting -> pool -> filter -> top-5 -> LLM -> run records (src/ire/eval/sweep.py).

Usage (repo root; the real run needs the caches, indexes, filters' thresholds and the LLM, i.e. a job on gnode056):
    python scripts/run_sweep.py --list                                  # print the settings and counts, load nothing
    python scripts/run_sweep.py --tier headline                         # the core grid at rate 3
    python scripts/run_sweep.py --tier headline rates sweeps adaptive   # more tiers (cut order: plan 6.1)
    python scripts/run_sweep.py --pilot                                 # ~500-request pilot (plan 6.1), own folder
Options: --retriever bm25 dense ...   --attack fact hijack ...   --batch-size N   --max-settings N
         --questions-per-role N (use a seeded subset of N target + N clean questions)   --out-name NAME
Resumable: finished settings are skipped; re-run the same command after a crash or a time limit.
Writes run records under <runs_dir>/<out-name>/ and the report <reports_dir>/SWEEP_<out-name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.eval.sweep import SweepRunner, expand, load_sweep  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", nargs="+", default=["headline"])
    ap.add_argument("--retriever", nargs="+")
    ap.add_argument("--attack", nargs="+")
    ap.add_argument("--list", action="store_true", help="print the settings and exit (loads no model)")
    ap.add_argument("--pilot", action="store_true",
                    help="10 target + 10 clean questions, attack fact only, tier headline, out-name pilot")
    ap.add_argument("--questions-per-role", type=int)
    ap.add_argument("--batch-size", type=int, help="LLM batch size (decide with A7)")
    ap.add_argument("--max-settings", type=int)
    ap.add_argument("--out-name", default="main")
    args = ap.parse_args()

    if args.pilot:
        args.tier, args.attack, args.out_name = ["headline"], args.attack or ["fact"], "pilot"
        args.questions_per_role = args.questions_per_role or 10

    cfg = load_config()
    settings = expand(load_sweep(cfg.path("sweep_config")), args.tier, args.retriever, args.attack)
    by_tier = Counter(s.tier for s in settings)
    by_filter = Counter(s.filter for s in settings)
    print(f"{len(settings)} settings; by tier {dict(by_tier)}; by filter {dict(by_filter)}")
    if args.list:
        for s in settings:
            print(" ", s.tier, s.sid)
        return 0

    out_dir = cfg.path("runs_dir") / args.out_name
    runner = SweepRunner(cfg, out_dir, questions_per_role=args.questions_per_role, batch_size=args.batch_size)
    print(f"{len(runner.questions)} questions; run records -> {out_dir}")
    t0 = time.time()
    failed = runner.run_all(settings, max_settings=args.max_settings)
    n_files = len(list(out_dir.glob("*.jsonl")))
    requests = runner.stats["generated"] + runner.stats["cache_hits"]
    report = {
        "job": f"SWEEP_{args.out_name}", "env": cfg.env, "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": not failed, "tiers": args.tier, "retrievers": args.retriever, "attacks": args.attack,
        "questions": len(runner.questions), "settings_planned": len(settings), "settings_files_present": n_files,
        "stats": runner.stats, "generation_requests": requests,
        "generation_cache_hit_rate": round(runner.stats["cache_hits"] / requests, 4) if requests else None,
        "llm": cfg["models.llm"], "batch_size": args.batch_size or cfg["compute.generate_batch_size"],
        "failed": failed[:50], "runtime_s": round(time.time() - t0, 1),
        "rerun": "python scripts/run_sweep.py " + " ".join(sys.argv[1:]),
    }
    out = cfg.path("reports_dir") / f"{report['job']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRESULT: {'OK' if not failed else 'FAILED'}  settings run {runner.stats['settings_run']}, "
          f"skipped {runner.stats['settings_skipped']}, failed {len(failed)}; runtime {report['runtime_s']} s")
    for f in failed[:10]:
        print("  FAILED", f)
    print(f"REPORT: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
