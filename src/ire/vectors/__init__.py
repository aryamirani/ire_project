"""Vector API (IMPLEMENTATION_PLAN.md section 2), used by the cluster filter and Arya's similarity filter.

    from ire.vectors import get_doc_vector, get_query_vector
    get_doc_vector(["doc12", "poison_q1_fact_1"])  -> np.ndarray [2, d] float32, L2-normalised (read from disk)
    get_query_vector(["who wrote hamlet"])         -> np.ndarray [1, d] float32, L2-normalised ("query: " prefix)

Both use the dense index of the active environment (<indexes_dir>/<env>/dense), opened once per process.
Passage vectors are never re-encoded; unknown doc_ids raise KeyError. For an explicit object (e.g. to put in a
FilterContext) use `default_vectors()`, which implements the VectorAPI protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ire.retrieval.dense import DenseRetriever

_default: DenseRetriever | None = None


def default_vectors() -> DenseRetriever:
    """The dense index of the active environment (loaded on first use)."""
    global _default
    if _default is None:
        from ire.config import load_config

        _default = DenseRetriever.from_config(load_config())
    return _default


def get_doc_vector(doc_ids: Sequence[str]) -> np.ndarray:
    return default_vectors().get_doc_vector(doc_ids)


def get_query_vector(texts: Sequence[str]) -> np.ndarray:
    return default_vectors().get_query_vector(texts)


__all__ = ["DenseRetriever", "default_vectors", "get_doc_vector", "get_query_vector"]
