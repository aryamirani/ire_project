"""Controls (plan section 4.4): the random filter and the oracle filter.

random: drops the same NUMBER of passages as the detector under test, chosen uniformly at random from the pool
        with a seeded RNG, so the result is reproducible and independent of the order questions are processed.
oracle: drops every poison passage in the pool (the best any detector can do).
Both return the flagged doc_ids; the first five unflagged passages in rank order then go to the LLM as usual.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ire.schemas import is_poison_id


def random_filter(pool: Sequence[str], n_drop: int, seed: int, key: str) -> set[str]:
    """`n_drop` passages of `pool` chosen at random; `key` (e.g. the setting id + qid) makes the draw reproducible."""
    n = max(0, min(int(n_drop), len(pool)))
    rng = random.Random(f"{seed}:{key}")
    return set(rng.sample(list(pool), n))


def oracle_filter(pool: Sequence[str]) -> set[str]:
    """Every poison passage in the pool."""
    return {d for d in pool if is_poison_id(d)}
