"""Figures from the analysis tables (plan: plots after stats). Needs matplotlib (`pip install matplotlib`).

Usage (repo root):
    python scripts/make_plots.py                      # analysis_dir/main/*.csv -> analysis_dir/main/fig_*.png
    python scripts/make_plots.py --run-name pilot

Figures (only those whose inputs exist are drawn):
    fig_rates.png          ASR vs poison rate, no filter, per retriever (RQ1)
    fig_headline.png       ASR at rate 3: no filter / three detectors / oracle, per retriever (RQ2, RQ4)
    fig_mechanism.png      ASR vs number of poison passages in the final top-5 (plan 5.4)
    fig_sweeps.png         ASR and clean nDCG@5 against the detector parameter (descriptive)
    fig_adaptive.png       each detector against every attack, rate 3, mean over retrievers (RQ3)
    fig_auc.png            ROC-AUC of each detector's score on each attack (detector view)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402

RETRIEVERS = ("bm25", "dense", "splade", "rrf")
DETECTORS = ("query_phrase", "cluster", "perplexity")
COLORS = {"none": "#7f7f7f", "query_phrase": "#1f77b4", "cluster": "#ff7f0e", "perplexity": "#2ca02c",
          "oracle": "#000000", "random": "#bbbbbb"}


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def ok(x) -> bool:
    return isinstance(x, float) and not math.isnan(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="main")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_config().path("analysis_dir") / args.run_name
    summary, head = read(d / "summary.csv"), read(d / "headline.csv")
    mech, comp = read(d / "mechanism.csv"), read(d / "complementarity.csv")
    attacks = sorted({r["attack"] for r in summary if r["attack"]})

    # 1. ASR vs rate, no filter
    rows = [r for r in summary if r["filter"] == "none" and r["attack"] in ("fact", "hijack") and ok(r["asr"])]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, a in zip(axes, ("fact", "hijack")):
            for ret in RETRIEVERS:
                pts = sorted((r["rate"], r["asr"], r["asr_low"], r["asr_high"]) for r in rows
                             if r["attack"] == a and r["retriever"] == ret)
                if pts:
                    x, y, lo, hi = zip(*pts)
                    ax.errorbar(x, y, yerr=[[v - l for v, l in zip(y, lo)], [h - v for v, h in zip(y, hi)]],
                                marker="o", capsize=3, label=ret)
            ax.set_title(a)
            ax.set_xlabel("poison passages per target question")
            ax.set_xticks([1, 3, 5])
        axes[0].set_ylabel("ASR (no filter)")
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(d / "fig_rates.png", dpi=160)
        plt.close(fig)

    # 2. headline bars
    if head:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, a in zip(axes, ("fact", "hijack")):
            w = 0.16
            series = [("none", "asr_none")] + [(det, None) for det in DETECTORS] + [("oracle", "asr_oracle")]
            for i, (name, col) in enumerate(series):
                ys = []
                for ret in RETRIEVERS:
                    rr = [r for r in head if r["attack"] == a and r["retriever"] == ret]
                    if not rr:
                        ys.append(0.0)
                    elif name in DETECTORS:
                        m = [r for r in rr if r["detector"] == name]
                        ys.append(m[0]["asr_detector"] if m else 0.0)
                    else:
                        ys.append(rr[0][col])
                ax.bar([j + (i - 2) * w for j in range(len(RETRIEVERS))], ys, w, label=name, color=COLORS[name])
            ax.set_xticks(range(len(RETRIEVERS)), RETRIEVERS)
            ax.set_title(f"{a}, rate 3")
        axes[0].set_ylabel("ASR")
        axes[0].set_ylim(0, 1.05)   # headroom for the legend
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "fig_headline.png", dpi=160)
        plt.close(fig)

    # 3. mechanism
    if mech:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, a in zip(axes, ("fact", "hijack")):
            for f in ("none", *DETECTORS):
                pts = sorted(((str(r["n_poison_top5"]), r["asr"], r["n"]) for r in mech
                              if r["attack"] == a and r["filter"] == f and r["n"] >= 5))
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f, color=COLORS[f])
            ax.set_title(a)
            ax.set_xlabel("poison passages in the final top-5")
        axes[0].set_ylabel("ASR")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "fig_mechanism.png", dpi=160)
        plt.close(fig)

    # 4. sweeps (mean over retrievers, rate 3)
    pname = {"query_phrase": "tail", "cluster": "tau_c", "perplexity": "alpha"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex="col")
    drew = False
    for j, det in enumerate(DETECTORS):
        for a in ("fact", "hijack"):
            by = defaultdict(list)
            for r in summary:
                if r["filter"] == det and r["attack"] == a and r["rate"] == 3 and ok(r["asr"]):
                    by[float(str(r["param"]).split("=")[1])].append((r["asr"], r["c_ndcg5"]))
            if len(by) > 1:
                xs = sorted(by)
                axes[0][j].plot(xs, [sum(v[0] for v in by[x]) / len(by[x]) for x in xs], marker="o", label=a)
                axes[1][j].plot(xs, [sum(v[1] for v in by[x]) / len(by[x]) for x in xs], marker="o", label=a)
                drew = True
        axes[1][j].set_xlabel(pname[det])
        axes[0][j].set_title(det)
    axes[0][0].set_ylabel("ASR")
    axes[1][0].set_ylabel("clean nDCG@5")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if drew:
        fig.savefig(d / "fig_sweeps.png", dpi=160)
    plt.close(fig)

    # 5. adaptive / complementarity
    if comp:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        names = ["none", *DETECTORS, "oracle"]
        w = 0.16
        for i, name in enumerate(names):
            col = f"asr_{name}"
            ys = []
            for a in attacks:
                vals = [r[col] for r in comp if r["attack"] == a and ok(r.get(col))]
                ys.append(sum(vals) / len(vals) if vals else 0.0)
            ax.bar([j + (i - 2) * w for j in range(len(attacks))], ys, w, label=name, color=COLORS[name])
        ax.set_xticks(range(len(attacks)), attacks)
        ax.set_ylabel("ASR (mean over retrievers)")
        ax.set_ylim(0, 1.0)   # headroom so the legend does not cover the tallest bars
        ax.legend(frameon=False, fontsize=8, ncol=5)
        fig.tight_layout()
        fig.savefig(d / "fig_adaptive.png", dpi=160)
        plt.close(fig)

    # 6. AUC at the anchors
    auc = [r for r in summary if r["filter"] in DETECTORS and r["rate"] == 3 and ok(r["auc"])]
    if auc:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.25
        for i, det in enumerate(DETECTORS):
            ys = []
            for a in attacks:
                v = [r["auc"] for r in auc if r["filter"] == det and r["attack"] == a]
                ys.append(sum(v) / len(v) if v else 0.0)
            ax.bar([j + (i - 1) * w for j in range(len(attacks))], ys, w, label=det, color=COLORS[det])
        ax.axhline(0.5, color="k", lw=0.6, ls=":")
        ax.set_xticks(range(len(attacks)), attacks)
        ax.set_ylabel("ROC-AUC (mean over retrievers)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(d / "fig_auc.png", dpi=160)
        plt.close(fig)

    print(f"figures written to {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
