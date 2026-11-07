"""Turn run records into the result tables (plan section 5): CSV files plus a provenance JSON.

Usage (repo root; CPU only, no models; run it wherever the run records are, e.g. after copying them back):
    python scripts/analyze.py                        # runs_dir/main -> analysis_dir/main
    python scripts/analyze.py --run-name pilot       # look at the pilot
    python scripts/analyze.py --bound 0.02 --auc-boot 200

Writes <analysis_dir>/<run-name>/:
    summary.csv         every setting: ASR (Wilson), EM/F1/acc, nDCG, collateral poisoning, TPR/FPR/AUC, backfill, ms
    headline.csv        anchors at rate 3: detector vs no filter / random / oracle, Holm-adjusted p, `helps`
    crossfit.csv        sweep parameters chosen on one fold, reported on the other (pooled)
    mechanism.csv       ASR by number of poison passages in the final top-5
    complementarity.csv ASR of each detector against every attack (RQ3)
    analysis.json       provenance: git commit, number of settings, thresholds used
The decision rule is docs/DECISION_RULE.md; `bound` is its clean nDCG@5 loss bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.eval import analysis as an  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402

RETRIEVERS = ("bm25", "dense", "splade", "rrf")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  {path.name}: {len(rows)} rows")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--bound", type=float, default=0.02, help="max clean nDCG@5 loss for a defense to 'help'")
    ap.add_argument("--auc-boot", type=int, default=200, help="bootstrap resamples for AUC intervals (0 = no CI)")
    args = ap.parse_args()

    cfg = load_config()
    run_dir = cfg.path("runs_dir") / args.run_name
    out_dir = cfg.path("analysis_dir") / args.run_name
    runs = an.load_runs(run_dir)
    if not runs:
        print(f"no run records in {run_dir}; run scripts/run_sweep.py first")
        return 1
    print(f"{len(runs)} settings loaded from {run_dir}")

    summary = []
    for key, recs in sorted(runs.items(), key=lambda kv: str(kv[0])):
        retriever, attack, rate, filt, of, param = key
        summary.append({"retriever": retriever, "attack": attack, "rate": rate, "filter": filt, "of": of,
                        "param": ";".join(f"{k}={v:g}" for k, v in param), **an.summarize(recs, args.auc_boot)})
    write_csv(out_dir / "summary.csv", summary)

    attacks_present = sorted({k[1] for k in runs if k[1] is not None})
    write_csv(out_dir / "headline.csv", an.headline(runs, cfg, RETRIEVERS, bound=args.bound))
    write_csv(out_dir / "crossfit.csv", an.cross_fit_rows(runs, cfg, RETRIEVERS, bound=args.bound))
    write_csv(out_dir / "mechanism.csv", an.mechanism(runs))
    write_csv(out_dir / "complementarity.csv", an.complementarity(runs, cfg, RETRIEVERS, attacks_present))

    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
            "run_name": args.run_name, "settings": len(runs), "bound": args.bound, "auc_boot": args.auc_boot,
            "anchors": {"query_phrase_tail": cfg["query_phrase.anchor_tail"], "cluster_tau_c": cfg["cluster.tau_c"],
                        "perplexity_alpha": cfg["perplexity.alpha"]},
            "attacks": attacks_present}
    (out_dir / "analysis.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nRESULT: OK   tables in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
