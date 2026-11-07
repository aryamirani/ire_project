"""Statistics (plan section 5): Wilson intervals, exact McNemar, paired bootstrap, question-level bootstrap for
ROC-AUC, Holm correction. numpy + scipy only. Every bootstrap resamples QUESTIONS, never passages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proportion, low, high) Wilson interval; (nan, nan, nan) when n == 0."""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (b = A only, c = B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_binary(a: Sequence[int], b: Sequence[int]) -> dict[str, float]:
    """Paired binary outcomes over the same questions (a = before, b = after). Returns the means, the difference
    (b - a), the discordant counts and the exact McNemar p."""
    a_arr, b_arr = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    only_a = int(((a_arr == 1) & (b_arr == 0)).sum())   # success before, not after
    only_b = int(((a_arr == 0) & (b_arr == 1)).sum())
    n = len(a_arr)
    return {"n": n, "mean_a": float(a_arr.mean()) if n else math.nan, "mean_b": float(b_arr.mean()) if n else math.nan,
            "diff": float((b_arr - a_arr).mean()) if n else math.nan, "only_a": only_a, "only_b": only_b,
            "p_mcnemar": mcnemar_exact(only_a, only_b)}


def paired_bootstrap(diffs: Sequence[float], n_boot: int = 10_000, seed: int = 13,
                     alpha: float = 0.05) -> dict[str, float]:
    """Mean of per-question differences with a percentile bootstrap interval (resampling questions)."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return {"n": 0, "mean": math.nan, "low": math.nan, "high": math.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": int(d.size), "mean": float(d.mean()), "low": float(lo), "high": float(hi)}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for a family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC (probability a poison passage outscores a clean one, ties count half); nan if one class is missing."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(s)   # average ranks for ties
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(groups: Sequence[tuple[Sequence[int], Sequence[float]]], n_boot: int = 1000, seed: int = 13,
                  alpha: float = 0.05) -> dict[str, float]:
    """Pooled ROC-AUC over per-question (labels, scores) groups, with a bootstrap interval over questions."""
    groups = [(np.asarray(y, dtype=int), np.asarray(s, dtype=float)) for y, s in groups if len(y)]
    if not groups:
        return {"auc": math.nan, "low": math.nan, "high": math.nan}
    y_all = np.concatenate([g[0] for g in groups])
    s_all = np.concatenate([g[1] for g in groups])
    auc = roc_auc(y_all, s_all)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        y = np.concatenate([groups[i][0] for i in pick])
        s = np.concatenate([groups[i][1] for i in pick])
        v = roc_auc(y, s)
        if not math.isnan(v):
            boots.append(v)
    if not boots:
        return {"auc": auc, "low": math.nan, "high": math.nan}
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"auc": auc, "low": float(lo), "high": float(hi)}


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    """Cohen's kappa for two annotators over the same items (any hashable labels)."""
    n = len(a)
    if n == 0 or n != len(b):
        return math.nan
    cats = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == c for x in a) / n) * (sum(y == c for y in b) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
