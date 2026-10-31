"""Stand-ins for teammates' contracts (docs/IMPLEMENTATION_PLAN.md section 2) until their versions are merged.

candidate_pool: Vansh's masking (plan section 2.3). This STUB returns the first pool_size cached ids WITHOUT
masking, as the plan specifies for the smoke test. Vansh's real version masks inactive poison (rate r keeps
poison_<qid>_<attack>_1..r) and, for rrf, masks the bm25 and dense lists before fusing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ire.retrieval.rrf import fuse

Row = Sequence[tuple[str, float]]


def candidate_pool(qid: str, retriever: str, attack: str | None, rate: int | None,
                   cache: Mapping[str, Mapping[str, Row]], pool_size: int = 50, cfg: Any = None) -> list[str]:
    """STUB: first pool_size cached doc_ids, no masking (attack and rate are ignored).

    cache: {retriever: {qid: [(doc_id, score), ...]}} for bm25 / dense / splade (from ire.retrieval.cache.load_cache);
    rrf is fused from the cached bm25 and dense rows (k and depth from cfg if given).
    """
    if retriever == "rrf":
        k = int(cfg["rrf.k"]) if cfg is not None else 60
        depth = int(cfg["rrf.depth"]) if cfg is not None else 150
        row = fuse(list(cache["bm25"][qid]), list(cache["dense"][qid]), k=k, depth=depth)
    else:
        row = cache[retriever][qid]
    return [d for d, _ in row][:pool_size]
