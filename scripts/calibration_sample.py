"""Human calibration sheets (plan section 7): 150 answers for the automatic ASR judge + 100 passages per member.

Usage (repo root; after the sweep, wherever run records and the text store exist):
    python scripts/calibration_sample.py                 # run records in runs_dir/main
    python scripts/calibration_sample.py --run-name main --seed 13

Writes <calibration_dir>/:
    answers_<name>.csv   sheet for arya / samarth / vansh: item_id, question, gold, attack target string(s), answer, label
                         (they fill `label`: 1 = the answer shows the attack succeeded, 0 = it did not). The automatic
                         verdict, the setting and the split are NOT shown (blind labelling).
    passages_<name>.csv  100 passages per member: item_id, question, text, label (poison / clean / ambiguous);
                         the doc_id is hidden because poison ids start with "poison_".
    key.json             everything hidden: item -> split (dev/test), pair, automatic verdict, truth for passages.
150 answers: stratified over attack x outcome, split 75 dev / 75 test, 50 per annotator pair (AS, SV, VA).
Annotators send back their filled CSVs; scripts/calibration_agreement.py computes kappa and accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.eval.sweep import load_targets, read_records  # noqa: E402
from ire.schemas import QuestionRecord, is_poison_id, read_jsonl  # noqa: E402

PAIRS = {"AS": ("arya", "samarth"), "SV": ("samarth", "vansh"), "VA": ("vansh", "arya")}
MEMBERS = ("arya", "samarth", "vansh")


def write_sheet(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-answers", type=int, default=150)
    ap.add_argument("--n-passages", type=int, default=100)
    args = ap.parse_args()
    cfg = load_config()
    rng = random.Random(f"{args.seed}:calibration")
    out_dir = cfg.path("calibration_dir")
    qs = {q.qid: q for q in read_jsonl(cfg.path("questions"), QuestionRecord)}
    targets, _ = load_targets(cfg)

    # ---- answers: unique (qid, attack, answer) from target questions, stratified by attack x outcome
    seen, strata = set(), defaultdict(list)
    recs_all = []
    for path in sorted((cfg.path("runs_dir") / args.run_name).glob("*.jsonl")):
        recs_all += read_records(path)
    for r in recs_all:
        if r["role"] != "target" or r["asr"] is None:
            continue
        k = (r["qid"], r["attack"], r["answer"])
        if k in seen or not r["answer"].strip():
            continue
        seen.add(k)
        strata[(r["attack"], int(r["asr"]))].append(r)
    keys = sorted(strata)
    if not keys:
        print("no attacked target answers found; run the sweep first")
        return 1
    for k in keys:
        rng.shuffle(strata[k])
    picked, i = [], 0
    while len(picked) < args.n_answers and any(strata[k] for k in keys):   # round-robin over the strata
        k = keys[i % len(keys)]
        if strata[k]:
            picked.append(strata[k].pop())
        i += 1
    rng.shuffle(picked)
    half = len(picked) // 2
    key: dict = {"answers": {}, "passages": {}, "seed": args.seed, "run_name": args.run_name}
    sheets: dict[str, list[dict]] = {m: [] for m in MEMBERS}
    pair_names = list(PAIRS)
    for n, r in enumerate(picked):
        split = "dev" if n < half else "test"
        pair = pair_names[(n if n < half else n - half) % 3]
        item = f"a{n + 1:03d}"
        q = qs[r["qid"]]
        key["answers"][item] = {"split": split, "pair": pair, "auto_asr": int(r["asr"]), "attack": r["attack"],
                                "qid": r["qid"], "setting": r["setting"]}
        row = {"item_id": item, "question": q.question, "gold_answers": " | ".join(q.answers),
               "attack_target": " | ".join(targets.get((r["qid"], r["attack"]), [])), "answer": r["answer"],
               "label": ""}
        for m in PAIRS[pair]:
            sheets[m].append(row)
    for m, rows in sheets.items():
        write_sheet(out_dir / f"answers_{m}.csv", sorted(rows, key=lambda x: x["item_id"]),
                    ["item_id", "question", "gold_answers", "attack_target", "answer", "label"])

    # ---- passages: 50% poison / 50% clean from the pools of attacked target questions
    from ire.data.textstore import TextStore

    texts = TextStore.open_for(cfg)
    pool_poison, pool_clean = [], []
    for r in recs_all:
        if r["role"] == "target" and r["attack"] is not None and r["filter"] == "none":
            pool_poison += [(r["qid"], d) for d in r["pool"] if is_poison_id(d)]
            pool_clean += [(r["qid"], d) for d in r["pool"] if not is_poison_id(d)]
    pool_poison, pool_clean = sorted(set(pool_poison)), sorted(set(pool_clean))
    for m in MEMBERS:
        mrng = random.Random(f"{args.seed}:passages:{m}")
        chosen = (mrng.sample(pool_poison, min(args.n_passages // 2, len(pool_poison)))
                  + mrng.sample(pool_clean, min(args.n_passages - args.n_passages // 2, len(pool_clean))))
        mrng.shuffle(chosen)
        rows = []
        for n, (qid, doc_id) in enumerate(chosen):
            item = f"p_{m}_{n + 1:03d}"
            key["passages"][item] = {"doc_id": doc_id, "qid": qid, "truth": "poison" if is_poison_id(doc_id) else "clean"}
            rows.append({"item_id": item, "question": qs[qid].question, "text": texts[doc_id], "label": ""})
        write_sheet(out_dir / f"passages_{m}.csv", rows, ["item_id", "question", "text", "label"])

    (out_dir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    n_dev = sum(v["split"] == "dev" for v in key["answers"].values())
    print(f"{len(picked)} answers ({n_dev} dev / {len(picked) - n_dev} test) over strata {keys}")
    print(f"sheets and key.json in {out_dir}  (send each person ONLY their own answers_<name>.csv / passages_<name>.csv;"
          " keep key.json private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
