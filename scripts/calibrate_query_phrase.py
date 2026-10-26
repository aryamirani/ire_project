"""Arya: calibrate the query-and-phrase filter's similarity cutoffs and run its sanity checks (plan 4.1).

Usage (repo root; needs current top-150 caches for bm25 / dense / splade and the dense index):
    python scripts/calibrate_query_phrase.py                 # cutoffs + clean-side checks
    python scripts/calibrate_query_phrase.py --detection     # also the detection sanity check on poisoned pools
    python scripts/calibrate_query_phrase.py --corpus-sample 20000   # phrase-rule FPR on N random passages

Calibration: for every retriever and fold, take the CLEAN questions' top-50 pools with every poison passage
masked (the no-poison setting; RRF fuses the masked bm25 + dense lists), compute s(q, p) for all pool passages,
and store the (1 - tail) percentile for every tail of query_phrase.tail_sweep. A run reported on fold B uses
the fold-A cutoff and vice versa (`ire.defenses.query_phrase.thresholds_for`).

Writes configs/query_phrase_thresholds.json (tracked; the Ada version is the one used for results) and
$IRE_DATA_DIR/reports/query_phrase_<env>.json. Checks reported:
    - flag rate of the similarity rule on the opposite fold's clean pools ~= tail (cross-fold sanity),
    - phrase-rule false-positive rate on clean pool passages and on a random corpus sample,
    - (--detection) per attack at rate 5: TPR on poison and FPR on clean passages of the target questions'
      pools, anchor tail, cutoffs from the opposite fold. This is a sanity check, not the reported metric
      (Vansh's runner and metrics produce those).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.textstore import TextStore  # noqa: E402
from ire.defenses.query_phrase import (  # noqa: E402
    apply_filter,
    calibrate_cutoffs,
    load_phrases,
    phrase_hits,
    similarities,
    tail_key,
    thresholds_for,
)
from ire.retrieval.cache import load_cache  # noqa: E402
from ire.retrieval.rrf import fuse  # noqa: E402
from ire.schemas import FilterContext, QuestionRecord, is_poison_id, parse_poison_id, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402
from ire.vectors import default_vectors  # noqa: E402

BASE = ("bm25", "dense", "splade")


def masked(row, keep) -> list[tuple[str, float]]:
    """Drop every poison passage for which keep(doc_id) is False."""
    return [(d, s) for d, s in row if not is_poison_id(d) or keep(d)]


def pools_for(cache, cfg, qid: str, keep, pool_size: int) -> dict[str, list[str]]:
    """{retriever: top pool_size doc_ids after masking}; rrf fuses the masked component lists."""
    out = {r: [d for d, _ in masked(cache[r][qid], keep)][:pool_size] for r in BASE}
    comps = [masked(cache[c][qid], keep) for c in cfg["rrf.components"]]
    fused = fuse(comps[0], comps[1], k=int(cfg["rrf.k"]), depth=int(cfg["rrf.depth"]))
    out["rrf"] = [d for d, _ in fused][:pool_size]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detection", action="store_true", help="also run the detection sanity check")
    ap.add_argument("--corpus-sample", type=int, default=20000, help="random corpus passages for the phrase FPR")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("calibrate_query_phrase")
    t0 = time.time()
    pool_size, seed = int(cfg["pool_size"]), int(cfg["seed"])
    tails = [float(t) for t in cfg["query_phrase.tail_sweep"]]
    anchor = float(cfg["query_phrase.anchor_tail"])
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    cache = load_cache(list(BASE))
    vectors = default_vectors()
    texts = TextStore.open_for(cfg)
    phrases = load_phrases()
    folds = sorted({q.fold for q in questions})
    retrievers = [*BASE, "rrf"]
    no_poison = lambda d: False  # noqa: E731

    # ---- similarities on clean questions' clean pools
    sims: dict[str, dict[str, list[float]]] = {r: {f: [] for f in folds} for r in retrievers}
    clean_phrase = Counter()
    clean_seen: set[str] = set()
    for q in (q for q in questions if q.role == "clean"):
        for r, pool in pools_for(cache, cfg, q.qid, no_poison, pool_size).items():
            sims[r][q.fold] += [float(x) for x in similarities(q.question, pool, vectors)]
            clean_seen.update(pool)
    for d in clean_seen:
        clean_phrase["passages"] += 1
        if phrase_hits(texts[d], phrases):
            clean_phrase["flagged"] += 1
    cutoffs = {r: calibrate_cutoffs(sims[r], tails) for r in retrievers}
    cross = {r: {f: {tail_key(t): round(float(np.mean(np.asarray(sims[r][f]) >
                                                    cutoffs[r][folds[1 - folds.index(f)]][tail_key(t)])), 4)
                     for t in tails} for f in folds} for r in retrievers}

    # ---- phrase-rule FPR on random clean corpus passages
    ids = [d for d in texts if not is_poison_id(d)]
    sample = random.Random(seed).sample(ids, min(args.corpus_sample, len(ids)))
    corpus_hits = Counter()
    for d in sample:
        for pid in phrase_hits(texts[d], phrases):
            corpus_hits[pid] += 1
    corpus_flagged = sum(1 for d in sample if phrase_hits(texts[d], phrases))

    table = {"name": "query_phrase_thresholds", "env": cfg.env, "anchor_tail": anchor, "tails": tails,
             "pool_size": pool_size, "dense_model": cfg["models.dense"],
             "note": "cutoffs[retriever][fold][tail] were calibrated ON that fold's clean questions; a run "
                     "reported on fold X uses the other fold's cutoff (thresholds_for).",
             "n_pairs": {r: {f: len(sims[r][f]) for f in folds} for r in retrievers},
             "cutoffs": cutoffs}
    out_table = cfg.path("query_phrase_thresholds")
    out_table.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")

    # ---- detection sanity check (rate 5, every attack present in the shards)
    detection = None
    if args.detection:
        detection = {}
        attacks = sorted({parse_poison_id(d)[1] for d in texts if is_poison_id(d)})
        for attack in attacks:
            keep = lambda d, a=attack: parse_poison_id(d)[1] == a and parse_poison_id(d)[2] <= 5  # noqa: E731
            per_r = {}
            for r in retrievers:
                tp = fn = fp = tn = n_q = 0
                for q in (q for q in questions if q.role == "target"):
                    pool = pools_for(cache, cfg, q.qid, keep, pool_size)[r]
                    if not any(is_poison_id(d) for d in pool):
                        continue
                    n_q += 1
                    ctx = FilterContext(texts=texts, vectors=vectors, config=cfg,
                                        thresholds=thresholds_for(table, r, q.fold, anchor))
                    flagged = apply_filter(pool, q.qid, q.question, ctx).flagged
                    for d in pool:
                        if is_poison_id(d):
                            tp += d in flagged
                            fn += d not in flagged
                        else:
                            fp += d in flagged
                            tn += d not in flagged
                per_r[r] = {"questions_with_poison_in_pool": n_q, "poison_in_pools": tp + fn,
                            "tpr": round(tp / max(tp + fn, 1), 4), "fpr_clean": round(fp / max(fp + tn, 1), 4)}
            detection[attack] = per_r

    report = {
        "job": "query_phrase", "env": cfg.env, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "thresholds_file": str(out_table.relative_to(REPO_ROOT)),
        "cutoffs": cutoffs, "cross_fold_flag_rate": cross,
        "phrase_rule": {"patterns": len(phrases),
                        "clean_pool_passages": clean_phrase["passages"],
                        "clean_pool_fpr": round(clean_phrase["flagged"] / max(clean_phrase["passages"], 1), 5),
                        "corpus_sample": len(sample), "corpus_sample_fpr": round(corpus_flagged / max(len(sample), 1), 5),
                        "corpus_hits_by_pattern": dict(corpus_hits.most_common())},
        "detection_sanity_rate5_anchor": detection,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = cfg.path("reports_dir") / f"query_phrase_{cfg.env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"cutoffs at the anchor tail {anchor}:")
    for r in retrievers:
        print(f"  {r:<7} " + "  ".join(f"{f}: {cutoffs[r][f][tail_key(anchor)]:.4f}" for f in folds)
              + "   cross-fold flag rate " + "  ".join(f"{f}: {cross[r][f][tail_key(anchor)]:.3f}" for f in folds))
    pr = report["phrase_rule"]
    print(f"phrase rule: {pr['patterns']} patterns; FPR clean pool passages {pr['clean_pool_fpr']} "
          f"({pr['clean_pool_passages']}), random corpus sample {pr['corpus_sample_fpr']} ({pr['corpus_sample']})")
    if detection:
        print("detection sanity (rate 5, anchor, opposite-fold cutoffs): TPR / clean FPR")
        for attack, per_r in detection.items():
            print(f"  {attack:<8} " + "  ".join(f"{r}: {v['tpr']:.2f}/{v['fpr_clean']:.3f}" for r, v in per_r.items()))
    print(f"wrote {out_table}")
    print(f"REPORT: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
