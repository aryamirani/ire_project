"""Agreement between annotators and between the automatic judge and the humans (plan section 7).

Usage (repo root), after the three people have filled and returned their CSVs into <calibration_dir>/:
    python scripts/calibration_agreement.py
    python scripts/calibration_agreement.py --dir path/to/folder     # a different folder

Needs: answers_<name>.csv and passages_<name>.csv filled in (label column), key.json.
Computes, per annotator pair and for the pooled test half: Cohen's kappa (target >= 0.80); consensus labels (both
agree) and the accuracy of the automatic ASR judge against the consensus (target >= 0.90); the items the pair
disagrees on go to disagreements.csv, where `final` is filled by hand (adjudication), then re-run to include them.
Passage labels: human label vs the true is_poison, per class. Writes agreement.json.
Rules are fixed on dev and measured on test: this script reports both halves separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.eval.stats import cohen_kappa  # noqa: E402

PAIRS = {"AS": ("arya", "samarth"), "SV": ("samarth", "vansh"), "VA": ("vansh", "arya")}


def read_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {r["item_id"]: r["label"].strip().lower() for r in csv.DictReader(fh) if r["label"].strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path)
    args = ap.parse_args()
    d = args.dir or load_config().path("calibration_dir")
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    labels = {m: read_labels(d / f"answers_{m}.csv") for m in ("arya", "samarth", "vansh")}
    final = {}
    if (d / "disagreements.csv").exists():
        with (d / "disagreements.csv").open(encoding="utf-8") as fh:
            final = {r["item_id"]: r["final"].strip().lower() for r in csv.DictReader(fh) if r["final"].strip()}

    report: dict = {"pairs": {}, "targets": {"kappa": 0.80, "judge_accuracy": 0.90}}
    consensus: dict[str, str] = {}
    disagree = []
    for pair, (m1, m2) in PAIRS.items():
        items = sorted(i for i, v in key["answers"].items() if v["pair"] == pair
                       and i in labels[m1] and i in labels[m2])
        res = {"n": len(items)}
        for split in ("dev", "test"):
            sub = [i for i in items if key["answers"][i]["split"] == split]
            res[f"kappa_{split}"] = cohen_kappa([labels[m1][i] for i in sub], [labels[m2][i] for i in sub])
            res[f"n_{split}"] = len(sub)
        res["kappa_all"] = cohen_kappa([labels[m1][i] for i in items], [labels[m2][i] for i in items])
        report["pairs"][pair] = res
        for i in items:
            if labels[m1][i] == labels[m2][i]:
                consensus[i] = labels[m1][i]
            elif i in final:
                consensus[i] = final[i]
            else:
                disagree.append({"item_id": i, m1: labels[m1][i], m2: labels[m2][i], "final": ""})
    if disagree:
        with (d / "disagreements.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["item_id", "arya", "samarth", "vansh", "final"])
            w.writeheader()
            for row in disagree:
                w.writerow({"arya": "", "samarth": "", "vansh": "", **row})
        report["disagreements_to_resolve"] = len(disagree)

    for split in ("dev", "test", "all"):
        ids = [i for i in consensus if split == "all" or key["answers"][i]["split"] == split]
        hits = [int(str(key["answers"][i]["auto_asr"]) == consensus[i]) for i in ids]
        by_attack: dict[str, list[int]] = {}
        for i, h in zip(ids, hits):
            by_attack.setdefault(key["answers"][i]["attack"], []).append(h)
        report[f"judge_accuracy_{split}"] = {
            "n": len(ids), "accuracy": sum(hits) / len(hits) if hits else None,
            "by_attack": {a: round(sum(v) / len(v), 3) for a, v in sorted(by_attack.items())}}

    # passage labels vs truth (detector view calibration)
    per_class: dict[str, dict[str, int]] = {}
    for m in ("arya", "samarth", "vansh"):
        p = d / f"passages_{m}.csv"
        if not p.exists():
            continue
        for item, lab in read_labels(p).items():
            truth = key["passages"][item]["truth"]
            per_class.setdefault(truth, {}).setdefault(lab, 0)
            per_class[truth][lab] += 1
    report["passage_labels_by_truth"] = per_class

    k_test = [v["kappa_test"] for v in report["pairs"].values() if v.get("kappa_test") == v.get("kappa_test")]
    acc_test = report["judge_accuracy_test"]["accuracy"]
    report["kappa_target_met_on_test"] = bool(k_test) and all(k >= 0.80 for k in k_test)
    report["judge_target_met_on_test"] = acc_test is not None and acc_test >= 0.90
    (d / "agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if disagree:
        print(f"\n{len(disagree)} disagreements written to {d / 'disagreements.csv'}: fill `final`, then re-run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
