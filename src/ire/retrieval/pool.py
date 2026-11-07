"""Candidate pool with poison masking (plan sections 2.1-2.3). Replaces the stub in ire/stubs.py.

All poison passages of every attack and rate are in the indexes (and in the top-150 caches). A setting
(attack, rate) hides the poison that is not part of it: rate r of an attack keeps `poison_<qid>_<attack>_1..r`.

Decisions (written down here and in docs/DECISION_RULE.md):
- Poison of OTHER target questions stays active for the same attack. A clean question therefore sees whatever
  poison of that attack happens to rank in its top-150: this is what the plan calls collateral poisoning.
- attack=None is the no-poison setting: every poison passage is masked.
- rrf: the bm25 and dense rows are masked first, then fused (never mask the cached rrf reference file).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ire.retrieval.rrf import fuse
from ire.schemas import is_poison_id, parse_poison_id

Row = Sequence[tuple[str, float]]


def is_active(doc_id: str, attack: str | None, rate: int | None) -> bool:
    """True if the poison passage `doc_id` belongs to the setting (attack, rate)."""
    if attack is None or rate is None:
        return False
    _, doc_attack, n = parse_poison_id(doc_id)
    return doc_attack == attack and n <= rate


def mask_row(row: Row, attack: str | None, rate: int | None) -> list[tuple[str, float]]:
    """Drop the poison passages that are not active in the setting; clean passages are always kept."""
    return [(d, s) for d, s in row if not is_poison_id(d) or is_active(d, attack, rate)]


def masked_row(qid: str, retriever: str, attack: str | None, rate: int | None,
               cache: Mapping[str, Mapping[str, Row]], cfg: Any = None) -> list[tuple[str, float]]:
    """The masked ranked list (up to 150 entries) for one question."""
    if retriever == "rrf":
        k = int(cfg["rrf.k"]) if cfg is not None else 60
        depth = int(cfg["rrf.depth"]) if cfg is not None else 150
        return fuse(mask_row(cache["bm25"][qid], attack, rate), mask_row(cache["dense"][qid], attack, rate),
                    k=k, depth=depth)
    return mask_row(cache[retriever][qid], attack, rate)


def candidate_pool(qid: str, retriever: str, attack: str | None, rate: int | None,
                   cache: Mapping[str, Mapping[str, Row]], pool_size: int = 50, cfg: Any = None) -> list[str]:
    """Ranked doc_ids of the top-`pool_size` after masking inactive poison (same signature as the stub).

    cache: {retriever: {qid: [(doc_id, score), ...]}} for bm25 / dense / splade (ire.retrieval.cache.load_cache).
    """
    return [d for d, _ in masked_row(qid, retriever, attack, rate, cache, cfg)][:pool_size]
