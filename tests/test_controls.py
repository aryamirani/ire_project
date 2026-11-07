"""Random and oracle controls (plan 4.4)."""

from __future__ import annotations

from ire.defenses.controls import oracle_filter, random_filter

POOL = ["a", "poison_q1_fact_1", "b", "c", "poison_q1_fact_2", "d"]


def test_random_filter_size_subset_and_determinism():
    r1 = random_filter(POOL, 3, 13, "setting:q1")
    assert len(r1) == 3 and r1 <= set(POOL)
    assert r1 == random_filter(POOL, 3, 13, "setting:q1")
    assert len(random_filter(POOL, 99, 13, "k")) == len(POOL)
    assert random_filter(POOL, 0, 13, "k") == set()


def test_oracle_filter_drops_only_poison():
    assert oracle_filter(POOL) == {"poison_q1_fact_1", "poison_q1_fact_2"}
