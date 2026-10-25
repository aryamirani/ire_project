"""Reciprocal rank fusion (Step 6): a pure function over ranked lists, plus a convenience RRF retriever.

    from ire.retrieval.rrf import fuse
    fused = fuse(masked_bm25_top150, masked_dense_top150)   # -> [(doc_id, rrf_score), ...]

score(d) = sum over input lists that contain d of 1 / (k + rank(d)), rank starting at 1 within each list.
Only the ORDER of each input list is used; its scores are ignored. Output is sorted by score descending, ties by
doc_id ascending, and cut to `depth`.

Masking poison must happen on each component list BEFORE fusing (plan section 2.1): removing a document shifts
the ranks of everything below it, and RRF scores depend on those ranks. Fusing first and deleting poison afterwards
gives different scores, can reorder the clean documents, and loses documents that were cut at `depth`. So this
module never masks anything; Vansh's candidate_pool masks the component lists and then calls fuse().
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

Ranked = Sequence[tuple[str, float]] | Sequence[Sequence[Any]]  # (doc_id, score) pairs; JSON gives lists


def fuse_many(lists: Sequence[Ranked], k: int = 60, depth: int | None = 150) -> list[tuple[str, float]]:
    """RRF over any number of ranked lists (each already masked). depth=None keeps every fused document.

    Raises ValueError if a list contains the same doc_id twice or k <= 0.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")
    contributions: dict[str, list[float]] = {}
    for li, ranked in enumerate(lists):
        seen: set[str] = set()
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[0]
            if doc_id in seen:
                raise ValueError(f"list {li} contains doc_id {doc_id!r} twice")
            seen.add(doc_id)
            contributions.setdefault(doc_id, []).append(1.0 / (k + rank))
    # math.fsum is exactly rounded, so the score does not depend on the order of the lists
    scores = {d: math.fsum(c) for d, c in contributions.items()}
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused if depth is None else fused[:depth]


def fuse(list_a: Ranked, list_b: Ranked, k: int = 60, depth: int | None = 150) -> list[tuple[str, float]]:
    """RRF of two ranked lists (e.g. masked BM25 and masked dense top-150). See fuse_many."""
    return fuse_many([list_a, list_b], k=k, depth=depth)


class _Searcher(Protocol):
    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]: ...


class RRFRetriever:
    """Runs each component retriever to `depth` and fuses the lists (no masking; for recall checks and the smoke
    test). Experiments with poison must fuse the masked component lists themselves (see the module docstring)."""

    def __init__(self, components: Mapping[str, _Searcher], k: int = 60, depth: int = 150) -> None:
        if len(components) < 2:
            raise ValueError("RRF needs at least two component retrievers")
        self.components = dict(components)
        self.k = k
        self.depth = depth

    @classmethod
    def from_config(cls, cfg: Any, components: Mapping[str, _Searcher] | None = None) -> RRFRetriever:
        if components is None:
            components = {}
            for name in cfg["rrf.components"]:
                if name == "bm25":
                    from ire.retrieval.bm25 import BM25Retriever

                    components[name] = BM25Retriever.from_config(cfg)
                elif name == "dense":
                    from ire.retrieval.dense import DenseRetriever

                    components[name] = DenseRetriever.from_config(cfg)
                elif name == "splade":
                    from ire.retrieval.splade import SpladeRetriever

                    components[name] = SpladeRetriever.from_config(cfg)
                else:
                    raise ValueError(f"unknown RRF component {name!r}")
        return cls(components, k=int(cfg["rrf.k"]), depth=int(cfg["rrf.depth"]))

    def search_components(self, queries: Sequence[str]) -> dict[str, list[list[tuple[str, float]]]]:
        """Each component's top-`depth` list per query (what the top-150 cache stores for rrf)."""
        return {name: r.search(list(queries), self.depth) for name, r in self.components.items()}

    def fuse_components(self, per_component: Mapping[str, list[list[tuple[str, float]]]], k: int | None = None
                        ) -> list[list[tuple[str, float]]]:
        """Fuse already-retrieved component lists query by query, cut to k (default: depth)."""
        names = list(self.components)
        n = len(per_component[names[0]])
        return [fuse_many([per_component[name][i] for name in names], self.k, k or self.depth) for i in range(n)]

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Top-k fused (doc_id, rrf_score) per query."""
        if not queries or k <= 0:
            return [[] for _ in queries]
        return self.fuse_components(self.search_components(queries), min(k, self.depth))
