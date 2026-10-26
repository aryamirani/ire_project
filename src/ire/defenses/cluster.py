"""Cluster filter (plan section 4.2; IMPLEMENTATION_PLAN step 10). Same API as every filter:

    apply_filter(pool, qid, question, ctx) -> FilterResult

1. Take the top `cluster.top_n` (20) passages of the masked pool and their stored dense vectors
   (ctx.vectors.get_doc_vector).
2. k-means with k = `cluster.k` (2), seeded (`seed` in experiment.yaml).
3. For each cluster with >= 2 members: mean pairwise cosine of the member vectors and mean pairwise ROUGE-L F1
   between the member texts (ctx.texts).
4. Flag the whole cluster only if cosine >= tau_c AND ROUGE-L >= `cluster.rouge_l_min` (0.25). tau_c comes from
   ctx.thresholds["tau_c"] if given (the sweep: 0.70 ... 0.95), else `cluster.tau_c` (anchor 0.85).
5. The runner takes the first 5 unflagged passages of the whole 50-pool; `backfill` reports whether any of them
   comes from beyond rank 20.

FilterResult.scores[doc_id] = mean pairwise cosine of the passage's cluster (0.0 for single-member clusters and
for passages beyond top_n), the continuous score for ROC-AUC. `explain` returns the per-cluster statistics.
No training, no LLM (TrustRAG's second stage is deliberately left out, plan 4.2).
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

import numpy as np

from ire.schemas import FilterContext, FilterResult

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _cfg(ctx: FilterContext, key: str, default: Any) -> Any:
    cfg = ctx.config
    return cfg.get(key, default) if hasattr(cfg, "get") else default


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (O(len(a) * len(b)) time, O(len(b)) memory)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 between two texts on lowercase word tokens (symmetric)."""
    ta, tb = tokens(a), tokens(b)
    lcs = lcs_len(ta, tb)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(ta), lcs / len(tb)
    return 2 * p * r / (p + r)


def _kmeans_labels(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    if len(np.unique(vectors.round(6), axis=0)) < k:  # fewer distinct points than clusters: one cluster
        return np.zeros(len(vectors), dtype=int)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)


def cluster_stats(pool: list[str], ctx: FilterContext) -> list[dict[str, Any]]:
    """Per-cluster statistics for the top_n passages of the pool."""
    top_n, k = int(_cfg(ctx, "cluster.top_n", 20)), int(_cfg(ctx, "cluster.k", 2))
    seed = int(_cfg(ctx, "seed", 13))
    head = list(pool[:top_n])
    if len(head) < 2:
        return []
    if ctx.vectors is None:
        raise ValueError("cluster filter needs ctx.vectors (ire.vectors.default_vectors())")
    vecs = np.asarray(ctx.vectors.get_doc_vector(head), dtype=np.float32)
    labels = _kmeans_labels(vecs, min(k, len(head)), seed)
    out = []
    for c in sorted(set(int(x) for x in labels)):
        idx = [i for i, lab in enumerate(labels) if lab == c]
        members = [head[i] for i in idx]
        if len(idx) >= 2:
            sub = vecs[idx]
            sims = sub @ sub.T
            iu = np.triu_indices(len(idx), k=1)
            cos = float(sims[iu].mean())
            texts = [ctx.texts[d] for d in members]
            rl = float(np.mean([rouge_l(a, b) for a, b in combinations(texts, 2)]))
        else:
            cos, rl = 0.0, 0.0
        out.append({"cluster": c, "members": members, "ranks": [i + 1 for i in idx], "size": len(idx),
                    "mean_cosine": cos, "mean_rouge_l": rl})
    return out


def apply_filter(pool: list[str], qid: str, question: str, ctx: FilterContext) -> FilterResult:
    """Flag every member of a near-duplicate cluster among the top_n pool passages."""
    tau_c = float(ctx.thresholds.get("tau_c", _cfg(ctx, "cluster.tau_c", 0.85)))
    rouge_min = float(_cfg(ctx, "cluster.rouge_l_min", 0.25))
    flagged: set[str] = set()
    scores = {d: 0.0 for d in pool}
    for c in cluster_stats(pool, ctx):
        if c["size"] < 2:
            continue
        for d in c["members"]:
            scores[d] = c["mean_cosine"]
        if c["mean_cosine"] >= tau_c and c["mean_rouge_l"] >= rouge_min:
            flagged.update(c["members"])
    return FilterResult(flagged=flagged, scores=scores)


def explain(pool: list[str], question: str, ctx: FilterContext) -> dict[str, Any]:
    """Cluster statistics, the thresholds used and which clusters were flagged."""
    tau_c = float(ctx.thresholds.get("tau_c", _cfg(ctx, "cluster.tau_c", 0.85)))
    rouge_min = float(_cfg(ctx, "cluster.rouge_l_min", 0.25))
    stats = cluster_stats(pool, ctx)
    for c in stats:
        c["flagged"] = c["size"] >= 2 and c["mean_cosine"] >= tau_c and c["mean_rouge_l"] >= rouge_min
    return {"tau_c": tau_c, "rouge_l_min": rouge_min, "clusters": stats}


def survivors(pool: list[str], flagged: set[str], k: int = 5) -> list[str]:
    """The first k unflagged passages in rank order (what goes to the LLM)."""
    return [d for d in pool if d not in flagged][:k]


def backfill(pool: list[str], flagged: set[str], k: int = 5, top_n: int = 20) -> bool:
    """True if any of the first k survivors comes from beyond rank top_n (plan 2.2, backfill rate)."""
    rank = {d: i + 1 for i, d in enumerate(pool)}
    return any(rank[d] > top_n for d in survivors(pool, flagged, k))
