"""Step 11 / job A8: perplexity thresholds, held-out check and the score cache for every pool passage.

Usage (repo root; needs corpus + poison shards; the pool cache step also needs current top-150 caches):
    python scripts/calibrate_perplexity.py              # all of it; report A8 on Ada, A8_local otherwise
    python scripts/calibrate_perplexity.py --no-pools   # thresholds + held-out check only

1. Seeded sample (seed from experiment.yaml) of 2 x perplexity.n_calibration (1,000) clean corpus passages:
   the first half calibrates, the second half is held out.
2. Thresholds for every alpha in perplexity.alpha_sweep (PD band q(a/2)..q(1-a/2), PM cutoff q(1-a)) ->
   configs/perplexity_thresholds.json on Ada; elsewhere $IRE_DATA_DIR/reports/perplexity_thresholds_<env>.json
   (the tracked file keeps the Ada calibration) unless --write-config.
3. Held-out flag rate per alpha for the pd rule, the pm rule and either (pd and pm should each be close to alpha).
4. Sanity: a fluent passage vs the same passage with a gibberish second half (PD must drop sharply).
5. Passage length in GPT-2 tokens (mean, median) and the fraction too_short (< perplexity.min_tokens).
6. Score cache: every doc_id in the top pool_size of every cached retriever row (all questions, unmasked, so
   every poison passage that can enter a pool) + every poison passage. Afterwards the sweeps do no GPT-2 work.
7. Detection sanity: flag rate at the anchor alpha per attack (poison passages) and on clean pool passages.
"""

from __future__ import annotations

import argparse
import json
import random
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
from ire.defenses.perplexity import PerplexityScores, alpha_key, calibrate, flags  # noqa: E402
from ire.schemas import is_poison_id, parse_poison_id  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402



def gibberish(tok, n_tokens: int, seed: int) -> str:
    """Seeded random GPT-2 vocabulary tokens, about n_tokens long after re-tokenising, none repeated
    (repetition would LOWER perplexity, plan ref. [5]); same length as the half it replaces, so the midpoint
    split still falls at the junction."""
    rng = random.Random(f"{seed}:gibberish")
    ids = rng.sample(range(1000, 40000), n_tokens)
    return tok.decode(ids)


def rate(recs, th, rule):
    ok = [r for r in recs if not r["too_short"]]
    return round(sum(flags(r, th, rule)["flagged"] for r in ok) / max(len(ok), 1), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pools", action="store_true", help="skip the pool score cache (step 6) and step 7")
    ap.add_argument("--report", help="report name (default: A8 on Ada, A8_local otherwise)")
    ap.add_argument("--write-config", action="store_true",
                    help="off Ada: also overwrite the tracked configs/perplexity_thresholds.json (default: only on Ada)")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()
    log = get_logger("calibrate_perplexity")
    seed, n_cal = int(cfg["seed"]), int(cfg["perplexity.n_calibration"])
    alphas = [float(a) for a in cfg["perplexity.alpha_sweep"]]
    anchor, rule = float(cfg["perplexity.alpha"]), str(cfg["perplexity.rule"])
    texts = TextStore.open_for(cfg)
    scores = PerplexityScores.from_config(cfg)

    clean_ids = sorted(d for d in texts if not is_poison_id(d))
    sample = random.Random(seed).sample(clean_ids, min(2 * n_cal, len(clean_ids)))
    cal_ids, held_ids = sample[:n_cal], sample[n_cal:]
    log.info("scoring %d calibration + %d held-out clean passages", len(cal_ids), len(held_ids))
    t = time.time()
    recs = scores.get(sample, texts)
    calib_s = time.time() - t
    cal = [recs[d] for d in cal_ids]
    held = [recs[d] for d in held_ids]
    thresholds = calibrate(cal, alphas)
    table = {"name": "perplexity_thresholds", "env": cfg.env, "model": cfg["models.perplexity"],
             "revision": scores.scorer.revision, "namespace": scores.scorer.namespace(), "seed": seed,
             "n_calibration": len(cal_ids), "n_scorable": sum(not r["too_short"] for r in cal),
             "min_tokens": int(cfg["perplexity.min_tokens"]), "anchor_alpha": anchor, "rule": rule,
             "note": "PD = f(pre) - f(post), PM = max(f(pre), f(post)), f = mean NLL per GPT-2 token of each half",
             "thresholds": thresholds}
    # The tracked thresholds file holds the Ada calibration used for the results. Off Ada the table goes to the
    # (untracked) reports folder unless --write-config is given, so a local check cannot replace the Ada values.
    th_path = (cfg.path("perplexity_thresholds") if cfg.env == "ada" or args.write_config
               else cfg.path("reports_dir") / f"perplexity_thresholds_{cfg.env}.json")
    th_path.parent.mkdir(parents=True, exist_ok=True)
    th_path.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    heldout = {alpha_key(a): {r: rate(held, thresholds[alpha_key(a)], r) for r in ("pd", "pm", "either")}
               for a in alphas}

    # sanity: fluent vs gibberish second half on the first scorable calibration passage
    probe = next(d for d in cal_ids if not recs[d]["too_short"])
    tok = scores.scorer.tokenizer()
    ids = tok(texts[probe], add_special_tokens=False)["input_ids"]
    broken_text = tok.decode(ids[: len(ids) // 2]) + gibberish(tok, len(ids) - len(ids) // 2, seed)
    fl, br = scores.scorer.score_texts([(probe, texts[probe]), ("probe_gibberish", broken_text)])
    sanity = {"doc_id": probe, "pd_fluent": round(fl["pd"], 3), "pd_gibberish": round(br["pd"], 3),
              "pm_fluent": round(fl["pm"], 3), "pm_gibberish": round(br["pm"], 3),
              "passed": br["pd"] < fl["pd"] - 2.0 and br["pm"] > fl["pm"]}

    both = cal + held
    lengths = [r["n_tokens"] for r in both]
    length = {"mean_tokens": round(statistics.mean(lengths), 1), "median_tokens": statistics.median(lengths),
              "frac_too_short": round(sum(r["too_short"] for r in both) / len(both), 4)}

    pools = detection = None
    if not args.no_pools:
        from ire.retrieval.cache import load_cache

        size = int(cfg["pool_size"])
        cache = load_cache(["bm25", "dense", "splade"])
        pool_ids = {d for rows in cache.values() for row in rows.values() for d, _ in row[:size]}
        poison_ids = {d for d in texts if is_poison_id(d)}
        want = sorted(pool_ids | poison_ids)
        t = time.time()
        all_recs = scores.get(want, texts)
        pools = {"n_pool_docs": len(pool_ids), "n_poison_docs": len(poison_ids), "n_scored_total": len(want),
                 "seconds": round(time.time() - t, 1), "passages_scored_by_gpt2_this_run": scores.scorer.n_scored,
                 "frac_too_short_pool": round(sum(all_recs[d]["too_short"] for d in pool_ids) / max(len(pool_ids), 1), 4)}
        th = thresholds[alpha_key(anchor)]
        by_attack: dict[str, list[dict]] = {}
        for d in poison_ids:
            by_attack.setdefault(parse_poison_id(d)[1], []).append(all_recs[d])
        clean_pool = [all_recs[d] for d in pool_ids if not is_poison_id(d)]
        detection = {"alpha": anchor, "rule": rule,
                     "flag_rate_by_attack": {a: rate(v, th, rule) for a, v in sorted(by_attack.items())},
                     "flag_rate_clean_pool": rate(clean_pool, th, rule)}

    cache_path = cfg.path("perplexity_cache")
    n_cached = scores.cache.count(scores.scorer.namespace())
    a = alpha_key(anchor)
    # ~99% interval for the held-out flag rate: the cutoffs are themselves quantiles estimated from len(cal)
    # passages, so both samples add variance, alpha (1 - alpha) (1 / n_cal + 1 / n_held)
    tol = 2.58 * (anchor * (1 - anchor) * (1 / max(len(cal), 1) + 1 / max(len(held), 1))) ** 0.5
    checks = {
        "thresholds_written": th_path.is_file(), "thresholds_file": th_path.name,
        "heldout_pd_rate_close_to_alpha": abs(heldout[a]["pd"] - anchor) <= tol,
        "heldout_pm_rate_close_to_alpha": abs(heldout[a]["pm"] - anchor) <= tol,
        "gibberish_sanity": sanity["passed"],
        "cache_populated": n_cached >= len(sample),
    }
    ok = all(checks.values())
    report = {
        "job": args.report or ("A8" if cfg.env == "ada" else f"A8_{cfg.env}"), "env": cfg.env,
        "git_commit": git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok, "checks": checks, "model": cfg["models.perplexity"], "model_revision": scores.scorer.revision,
        "device": getattr(scores.scorer, "device", None), "alpha_anchor": anchor, "rule": rule,
        "n_calibration": len(cal), "n_heldout": len(held), "heldout_tolerance": round(tol, 4),
        "heldout_flag_rate": heldout, "thresholds": thresholds, "sanity_gibberish": sanity, "length": length,
        "pool_cache": pools, "detection_sanity": detection,
        "cache": {"path_name": cache_path.name, "n_records": n_cached,
                  "bytes": cache_path.stat().st_size if cache_path.is_file() else 0},
        "calibration_seconds": round(calib_s, 1), "runtime_s": round(time.time() - t0, 1),
        "libraries": library_versions("torch", "transformers"),
        "rerun": "python scripts/calibrate_perplexity.py",
    }
    out = cfg.path("reports_dir") / f"{report['job']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"thresholds at alpha {anchor}: {thresholds[a]}  -> {th_path.name}")
    print(f"held-out flag rate (n={len(held)}, tolerance +/-{tol:.4f}): " +
          "  ".join(f"a={k}: pd {v['pd']}, pm {v['pm']}, either {v['either']}" for k, v in heldout.items()))
    print(f"gibberish sanity: {sanity}")
    print(f"passage length: {length}")
    if pools:
        print(f"pool score cache: {pools}")
        print(f"detection sanity ({rule}, alpha {anchor}): {detection['flag_rate_by_attack']}; "
              f"clean pool {detection['flag_rate_clean_pool']}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
