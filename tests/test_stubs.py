"""Step 13: candidate_pool stub (src/ire/stubs.py) -- first pool_size cached ids, no masking; rrf fuses bm25 + dense."""

from __future__ import annotations

from ire.retrieval.rrf import fuse
from ire.stubs import candidate_pool

CACHE = {"bm25": {"q1": [("a", 9.0), ("poison_q1_fact_1", 8.0), ("b", 7.0)]},
         "dense": {"q1": [("b", 0.9), ("c", 0.8), ("a", 0.7)]},
         "splade": {"q1": [("c", 5.0), ("a", 4.0)]}}


def test_stub_returns_first_ids_without_masking():
    assert candidate_pool("q1", "bm25", "fact", 1, CACHE, pool_size=2) == ["a", "poison_q1_fact_1"]
    assert candidate_pool("q1", "splade", None, None, CACHE) == ["c", "a"]


def test_stub_rrf_fuses_bm25_and_dense():
    want = [d for d, _ in fuse(CACHE["bm25"]["q1"], CACHE["dense"]["q1"], k=60, depth=150)][:3]
    assert candidate_pool("q1", "rrf", None, None, CACHE, pool_size=3) == want
