"""Analysis of run records (plan section 5): per-setting summaries, detector view, defense view, paired tests with
Holm correction, cross-fitting of sweep parameters, mechanism check and the complementarity table.

Input: the run-record JSONL files of ire.eval.sweep. Output: plain dicts / lists of rows (written as CSV by
scripts/analyze.py). Only numpy / scipy / stdlib. Sign convention: every "diff" is (filtered - reference), so a
negative ASR diff means the defense lowered ASR.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ire.eval.sweep import DETECTORS, ANCHOR_KEY, ANCHOR_CFG, Setting, param_tuple, read_records
from ire.eval import stats
from ire.schemas import is_poison_id

NAN = math.nan
Key = tuple  # (retriever, attack, rate, filter, of, param_tuple)


def mean(xs: Iterable[float | int]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def key_of(rec: Mapping[str, Any]) -> Key:
    return (rec["retriever"], rec["attack"], rec["rate"], rec["filter"], rec["of"], param_tuple(rec["param"]))


def setting_key(s: Setting) -> Key:
    return (s.retriever, s.attack, s.rate, s.filter, s.of, s.param)


def load_runs(run_dir: str | Path) -> dict[Key, dict[str, dict[str, Any]]]:
    """{setting key: {qid: record}} for every *.jsonl in the run directory."""
    runs: dict[Key, dict[str, dict[str, Any]]] = {}
    for path in sorted(Path(run_dir).glob("*.jsonl")):
        recs = read_records(path)
        if recs:
            runs[key_of(recs[0])] = {r["qid"]: r for r in recs}
    return runs


def anchor_key(cfg: Any, retriever: str, attack: str | None, rate: int | None, filter_name: str,
               of: str | None = None) -> Key:
    det = of or filter_name
    param = param_tuple({ANCHOR_KEY[det]: float(cfg[ANCHOR_CFG[det]])}) if det in ANCHOR_KEY else ()
    return (retriever, attack, rate, filter_name, of, param)


# ---------------------------------------------------------------- one setting

def summarize(recs: Mapping[str, Mapping[str, Any]], auc_boot: int = 200, seed: int = 13) -> dict[str, Any]:
    """Defense view (target and clean questions), detector view and cost for one setting."""
    rows = list(recs.values())
    tgt = [r for r in rows if r["role"] == "target"]
    cln = [r for r in rows if r["role"] == "clean"]
    out: dict[str, Any] = {"n_target": len(tgt), "n_clean": len(cln)}
    has_asr = any(r["asr"] is not None for r in tgt)
    k = sum(r["asr"] or 0 for r in tgt)
    out["asr"], out["asr_low"], out["asr_high"] = stats.wilson(k, len(tgt)) if has_asr else (NAN, NAN, NAN)
    for name in ("em", "f1", "acc", "ndcg5", "ndcg10", "recall10", "ndcg5_pool", "n_poison_top5", "n_poison_pool",
                 "n_gold_dropped", "backfill", "filter_ms"):
        out[f"t_{name}"] = mean(r[name] for r in tgt)
    for name in ("em", "f1", "acc", "ndcg5", "ndcg10", "ndcg5_pool", "n_poison_top5", "n_gold_dropped", "backfill"):
        out[f"c_{name}"] = mean(r[name] for r in cln)
    out["collateral_rate"] = mean(int(r["n_poison_top5"] > 0) for r in cln)
    pos = sum(r["n_poison_pool"] for r in rows)
    neg = sum(r["n_pool"] - r["n_poison_pool"] for r in rows)
    out["poison_in_pool"], out["clean_in_pool"] = pos, neg
    out["tpr"] = sum(r["n_poison_flagged"] for r in rows) / pos if pos else NAN
    out["fpr"] = sum(r["n_clean_flagged"] for r in rows) / neg if neg else NAN
    out["clean_q_flagged_rate"] = mean(int(r["n_clean_flagged"] + r["n_poison_flagged"] > 0) for r in cln)
    groups = [([int(is_poison_id(d)) for d in r["pool"]], r["pool_scores"]) for r in rows if r["pool_scores"]]
    if groups and auc_boot:
        a = stats.bootstrap_auc(groups, n_boot=auc_boot, seed=seed)
        out["auc"], out["auc_low"], out["auc_high"] = a["auc"], a["low"], a["high"]
    else:
        out["auc"] = out["auc_low"] = out["auc_high"] = NAN
    return out


# ---------------------------------------------------------------- two settings, paired by question

def paired(ref: Mapping[str, Mapping[str, Any]], other: Mapping[str, Mapping[str, Any]], seed: int = 13,
           n_boot: int = 5000) -> dict[str, Any]:
    """Compare `other` against `ref` on the questions both contain. diff = other - ref."""
    qids = sorted(set(ref) & set(other))
    tgt = [q for q in qids if ref[q]["role"] == "target"]
    cln = [q for q in qids if ref[q]["role"] == "clean"]
    out: dict[str, Any] = {"n_target": len(tgt), "n_clean": len(cln)}
    if tgt and all(ref[q]["asr"] is not None and other[q]["asr"] is not None for q in tgt):
        p = stats.paired_binary([ref[q]["asr"] for q in tgt], [other[q]["asr"] for q in tgt])
        out.update(asr_ref=p["mean_a"], asr_other=p["mean_b"], asr_diff=p["diff"], only_ref=p["only_a"],
                   only_other=p["only_b"], p_mcnemar=p["p_mcnemar"])
    else:
        out.update(asr_ref=NAN, asr_other=NAN, asr_diff=NAN, only_ref=0, only_other=0, p_mcnemar=NAN)
    for name, ids in (("t_ndcg5", tgt), ("c_ndcg5", cln), ("t_acc", tgt), ("t_n_poison_top5", tgt)):
        field = name[2:]
        b = stats.paired_bootstrap([other[q][field] - ref[q][field] for q in ids], n_boot=n_boot, seed=seed)
        out[f"d_{name}"], out[f"d_{name}_low"], out[f"d_{name}_high"] = b["mean"], b["low"], b["high"]
    return out


# ---------------------------------------------------------------- headline family (Holm) and controls

def headline(runs: Mapping[Key, Mapping[str, Mapping[str, Any]]], cfg: Any, retrievers: Sequence[str],
             attacks: Sequence[str] = ("fact", "hijack"), rate: int = 3, bound: float = 0.02,
             alpha: float = 0.05) -> list[dict[str, Any]]:
    """One row per (retriever, attack, detector) at the anchor: detector vs no filter, vs random, vs oracle.

    Family for Holm: McNemar p of (detector vs no filter) over all rows present (plan 5.3: 3 detectors x 4
    retrievers x 2 attacks = 24). `helps` = adjusted p < alpha AND ASR diff < 0 AND clean nDCG@5 loss <= bound.
    """
    rows, pvals = [], {}
    for r in retrievers:
        for a in attacks:
            base = runs.get((r, a, rate, "none", None, ()))
            oracle = runs.get((r, a, rate, "oracle", None, ()))
            if base is None:
                continue
            for det in DETECTORS:
                d = runs.get(anchor_key(cfg, r, a, rate, det))
                if d is None:
                    continue
                rnd = runs.get(anchor_key(cfg, r, a, rate, "random", det))
                vs_base = paired(base, d)
                row: dict[str, Any] = {"retriever": r, "attack": a, "rate": rate, "detector": det,
                                       "asr_none": vs_base["asr_ref"], "asr_detector": vs_base["asr_other"],
                                       "asr_diff": vs_base["asr_diff"], "p_vs_none": vs_base["p_mcnemar"],
                                       "d_c_ndcg5": vs_base["d_c_ndcg5"], "d_c_ndcg5_low": vs_base["d_c_ndcg5_low"],
                                       "d_c_ndcg5_high": vs_base["d_c_ndcg5_high"],
                                       "d_t_ndcg5": vs_base["d_t_ndcg5"], "d_t_acc": vs_base["d_t_acc"],
                                       "d_n_poison_top5": vs_base["d_t_n_poison_top5"]}
                if rnd is not None:
                    vr = paired(rnd, d)
                    row.update(asr_random=vr["asr_ref"], diff_vs_random=vr["asr_diff"], p_vs_random=vr["p_mcnemar"])
                else:
                    row.update(asr_random=NAN, diff_vs_random=NAN, p_vs_random=NAN)
                if oracle is not None:
                    row["asr_oracle"] = paired(base, oracle)["asr_other"]
                    gap = row["asr_none"] - row["asr_oracle"]
                    row["oracle_captured"] = (row["asr_none"] - row["asr_detector"]) / gap if gap > 0 else NAN
                else:
                    row["asr_oracle"] = row["oracle_captured"] = NAN
                rows.append(row)
                if not math.isnan(row["p_vs_none"]):
                    pvals[f"{r}|{a}|{det}"] = row["p_vs_none"]
    adj = stats.holm(pvals)
    for row in rows:
        row["p_holm"] = adj.get(f"{row['retriever']}|{row['attack']}|{row['detector']}", NAN)
        row["helps"] = bool(row["p_holm"] < alpha and row["asr_diff"] < 0 and row["d_c_ndcg5"] >= -bound)
        row["beats_random"] = bool(row["p_vs_random"] < alpha and row["diff_vs_random"] < 0)
        # mechanism check (plan 5.4): poison left the context but ASR did not move, or the reverse
        row["mechanism_flag"] = bool((row["d_n_poison_top5"] < -0.1 and row["asr_diff"] > -0.02)
                                     or (row["d_n_poison_top5"] > -0.02 and row["asr_diff"] < -0.05))
    return rows


# ---------------------------------------------------------------- cross-fitting (plan 5.2)

def cross_fit(base: Mapping[str, Mapping[str, Any]], by_param: Mapping[tuple, Mapping[str, Mapping[str, Any]]],
              anchor: tuple, bound: float = 0.02) -> dict[str, Any]:
    """Choose the parameter on one fold, report it on the other, swap, pool.

    For each fold X: among parameters whose clean nDCG@5 loss on X is <= bound, pick the largest ASR reduction on
    X (ties -> the anchor, then the smaller parameter); if none qualifies use the anchor. The chosen parameter is
    then evaluated on the other fold only. The pooled result covers every question exactly once.
    """
    chosen: dict[str, tuple] = {}
    pooled_ref: list[int] = []
    pooled_new: list[int] = []
    d_clean: list[float] = []
    for pick_fold, report_fold in (("A", "B"), ("B", "A")):
        def on(fold: str, recs: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
            return {q: r for q, r in recs.items() if r["fold"] == fold}

        best, best_red = anchor, -math.inf
        for param, recs in sorted(by_param.items()):
            cmp = paired(on(pick_fold, base), on(pick_fold, recs), n_boot=200)
            if math.isnan(cmp["asr_diff"]) or cmp["d_c_ndcg5"] < -bound:
                continue
            red = -cmp["asr_diff"]
            if red > best_red + 1e-12 or (abs(red - best_red) <= 1e-12 and param == anchor):
                best, best_red = param, red
        chosen[pick_fold] = best
        ref, new = on(report_fold, base), on(report_fold, by_param[best])
        for q in sorted(set(ref) & set(new)):
            if ref[q]["role"] == "target":
                pooled_ref.append(ref[q]["asr"])
                pooled_new.append(new[q]["asr"])
            else:
                d_clean.append(new[q]["ndcg5"] - ref[q]["ndcg5"])
    p = stats.paired_binary(pooled_ref, pooled_new) if pooled_ref else {"mean_a": NAN, "mean_b": NAN, "diff": NAN,
                                                                         "p_mcnemar": NAN}
    b = stats.paired_bootstrap(d_clean, n_boot=2000)
    return {"chosen_on_A_reported_on_B": chosen["A"], "chosen_on_B_reported_on_A": chosen["B"],
            "asr_none": p["mean_a"], "asr_filtered": p["mean_b"],
            "asr_diff": p["diff"], "p_mcnemar": p["p_mcnemar"], "d_c_ndcg5": b["mean"],
            "d_c_ndcg5_low": b["low"], "d_c_ndcg5_high": b["high"]}


def cross_fit_rows(runs: Mapping[Key, Mapping[str, Mapping[str, Any]]], cfg: Any, retrievers: Sequence[str],
                   attacks: Sequence[str] = ("fact", "hijack"), rate: int = 3, bound: float = 0.02) -> list[dict[str, Any]]:
    rows = []
    for r in retrievers:
        for a in attacks:
            base = runs.get((r, a, rate, "none", None, ()))
            if base is None:
                continue
            for det in DETECTORS:
                by_param = {k[5]: v for k, v in runs.items()
                            if k[:5] == (r, a, rate, det, None)}
                anchor = anchor_key(cfg, r, a, rate, det)[5]
                if anchor not in by_param or len(by_param) < 2:
                    continue
                res = cross_fit(base, by_param, anchor, bound)
                rows.append({"retriever": r, "attack": a, "rate": rate, "detector": det, "n_params": len(by_param),
                             **{k: (str(v) if isinstance(v, tuple) else v) for k, v in res.items()}})
    return rows


# ---------------------------------------------------------------- mechanism and complementarity

def mechanism(runs: Mapping[Key, Mapping[str, Mapping[str, Any]]], rate: int = 3,
              attacks: Sequence[str] = ("fact", "hijack")) -> list[dict[str, Any]]:
    """ASR by number of poison passages in the final top-5, pooled over retrievers and filters (plan 5.4)."""
    bins: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for k, recs in runs.items():
        if k[1] not in attacks or k[2] != rate or k[3] == "oracle":
            continue
        for r in recs.values():
            if r["role"] == "target" and r["asr"] is not None:
                bins[(k[1], min(r["n_poison_top5"], 3), k[3])].append(r["asr"])
    rows = []
    for (a, n, f), v in sorted(bins.items()):
        p, lo, hi = stats.wilson(sum(v), len(v))
        rows.append({"attack": a, "n_poison_top5": n if n < 3 else "3+", "filter": f, "n": len(v), "asr": p,
                     "asr_low": lo, "asr_high": hi})
    return rows


def complementarity(runs: Mapping[Key, Mapping[str, Mapping[str, Any]]], cfg: Any, retrievers: Sequence[str],
                    attacks: Sequence[str], rate: int = 3) -> list[dict[str, Any]]:
    """ASR with no filter / each detector / oracle for every attack (the table that answers RQ3)."""
    rows = []
    for a in attacks:
        for r in retrievers:
            base = runs.get((r, a, rate, "none", None, ()))
            if base is None:
                continue
            row: dict[str, Any] = {"attack": a, "retriever": r, "asr_none": summarize(base, 0)["asr"]}
            for det in DETECTORS:
                d = runs.get(anchor_key(cfg, r, a, rate, det))
                s = summarize(d, 0) if d else None
                row[f"asr_{det}"] = s["asr"] if s else NAN
                row[f"tpr_{det}"] = s["tpr"] if s else NAN
            o = runs.get((r, a, rate, "oracle", None, ()))
            row["asr_oracle"] = summarize(o, 0)["asr"] if o else NAN
            rows.append(row)
    return rows
