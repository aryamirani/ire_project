"""candidate_pool masking (plan 2.1-2.3): rate r keeps poison_<qid>_<attack>_1..r; rrf masks both lists, then fuses."""

from __future__ import annotations

from ire.retrieval.pool import candidate_pool, is_active, mask_row
from ire.retrieval.rrf import fuse

CACHE = {
    "bm25": {"q1": [("a", 9.0), ("poison_q1_fact_1", 8.0), ("poison_q2_fact_1", 7.5), ("poison_q1_fact_4", 7.0),
                    ("poison_q1_hijack_1", 6.0), ("b", 5.0)]},
    "dense": {"q1": [("b", 0.9), ("poison_q1_fact_2", 0.8), ("c", 0.7)]},
    "splade": {"q1": [("c", 5.0), ("poison_q1_hijack_2", 4.0), ("a", 3.0)]},
}


def test_is_active_rate_and_attack():
    assert is_active("poison_q1_fact_3", "fact", 3)
    assert not is_active("poison_q1_fact_4", "fact", 3)
    assert not is_active("poison_q1_hijack_1", "fact", 3)
    assert not is_active("poison_q1_fact_1", None, None)


def test_masking_keeps_only_active_poison_and_all_clean():
    assert candidate_pool("q1", "bm25", "fact", 3, CACHE) == ["a", "poison_q1_fact_1", "poison_q2_fact_1", "b"]
    assert candidate_pool("q1", "bm25", "fact", 5, CACHE)[3] == "poison_q1_fact_4"
    assert candidate_pool("q1", "bm25", "hijack", 1, CACHE) == ["a", "poison_q1_hijack_1", "b"]


def test_no_poison_setting_masks_everything_and_pool_size_cuts():
    assert candidate_pool("q1", "bm25", None, None, CACHE) == ["a", "b"]
    assert candidate_pool("q1", "splade", "hijack", 3, CACHE, pool_size=2) == ["c", "poison_q1_hijack_2"]


def test_other_questions_poison_stays_active():
    assert "poison_q2_fact_1" in candidate_pool("q1", "bm25", "fact", 1, CACHE)


def test_rrf_masks_components_before_fusing():
    want = [d for d, _ in fuse(mask_row(CACHE["bm25"]["q1"], "fact", 3), mask_row(CACHE["dense"]["q1"], "fact", 3),
                               k=60, depth=150)]
    assert candidate_pool("q1", "rrf", "fact", 3, CACHE) == want
    assert "poison_q1_hijack_1" not in want
    assert candidate_pool("q1", "rrf", None, None, CACHE) == [d for d, _ in fuse(
        [("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.7)], k=60, depth=150)]
