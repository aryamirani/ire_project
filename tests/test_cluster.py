"""Step 10: cluster filter (src/ire/defenses/cluster.py) on synthetic pools with a bag-of-words vector API."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ire.defenses.cluster import apply_filter, backfill, explain, lcs_len, rouge_l, survivors
from ire.schemas import FilterContext, FilterResult

DIM = 512
TOPICS = ["volcano lava eruption magma", "football goal league striker", "piano sonata composer orchestra",
          "glacier ice arctic melting", "rocket orbit satellite launch", "coffee bean roast espresso",
          "medieval castle knight siege", "python code compiler bug", "desert camel oasis dune",
          "tennis racket serve wimbledon", "volcanic island coral reef", "chess opening bishop gambit",
          "honey bee hive pollen", "subway train station commuter", "wheat harvest farm tractor",
          "jazz saxophone improvisation club", "comet tail telescope astronomer", "bakery bread flour oven",
          "ocean whale migration krill", "mountain trail hiking summit"]


class BowVectors:
    """L2-normalised hashed bag-of-words vectors of the texts (a stand-in for get_doc_vector)."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def get_doc_vector(self, doc_ids):
        out = np.zeros((len(doc_ids), DIM), dtype=np.float32)
        for i, d in enumerate(doc_ids):
            for w in self.texts[d].lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def get_query_vector(self, texts):
        raise NotImplementedError


def ctx_for(texts: dict[str, str], **thresholds) -> FilterContext:
    cfg = {"cluster.top_n": 20, "cluster.k": 2, "cluster.tau_c": 0.85, "cluster.rouge_l_min": 0.25, "seed": 13}
    return FilterContext(texts=texts, vectors=BowVectors(texts), config=cfg, thresholds=thresholds)


def diverse(n: int, start: int = 0) -> dict[str, str]:
    return {f"c{i}": f"A passage about {TOPICS[(start + i) % len(TOPICS)]} with some unrelated detail number {i}."
            for i in range(n)}


def near_duplicates(n: int) -> dict[str, str]:
    base = "Who wrote the novel Hamlet? The answer is Francis Bacon, as the official record clearly states"
    return {f"p{i}": f"{base} {extra}." for i, extra in enumerate(["today", "now", "here", "again", "indeed"][:n])}


def test_rouge_l():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert rouge_l("a b c", "x y z") == 0.0
    assert lcs_len(["a", "b", "c", "d"], ["a", "c", "d"]) == 3
    assert rouge_l("a b c d", "a c d") == pytest.approx(2 * 0.75 * 1.0 / 1.75)


def test_flags_the_five_near_duplicates_only():
    texts = {**near_duplicates(5), **diverse(15)}
    pool = ["c0", "p0", "c1", "p1", "c2", "p2", "c3", "p3", "c4", "p4"] + [f"c{i}" for i in range(5, 15)]
    res = apply_filter(pool, "q", "who wrote hamlet", ctx_for(texts))
    assert isinstance(res, FilterResult)
    assert res.flagged == {f"p{i}" for i in range(5)}
    assert all(res.scores[f"p{i}"] > 0.85 for i in range(5)) and set(res.scores) == set(pool)
    info = explain(pool, "who wrote hamlet", ctx_for(texts))
    assert sum(c["flagged"] for c in info["clusters"]) == 1


def test_twenty_diverse_passages_flag_nothing():
    texts = diverse(20)
    assert apply_filter(list(texts), "q", "x", ctx_for(texts)).flagged == set()


def test_single_member_cluster_is_never_flagged():
    texts = {**{f"p{i}": "Who wrote Hamlet? The answer is Francis Bacon." for i in range(19)},
             "lone": "A passage about volcano lava eruption magma."}
    pool = list(texts)
    res = apply_filter(pool, "q", "x", ctx_for(texts))
    assert "lone" not in res.flagged and res.scores["lone"] == 0.0
    assert res.flagged == {f"p{i}" for i in range(19)}


def test_tau_c_from_thresholds_and_rouge_gate():
    texts = {**near_duplicates(5), **diverse(15)}
    pool = list(texts)
    assert apply_filter(pool, "q", "x", ctx_for(texts, tau_c=0.999)).flagged == set()   # stricter sweep value
    ctx = ctx_for(texts)
    ctx.config["cluster.rouge_l_min"] = 1.01                                            # ROUGE gate alone blocks
    assert apply_filter(pool, "q", "x", ctx).flagged == set()


def test_only_top_n_is_clustered_and_backfill():
    texts = {**diverse(20), **near_duplicates(5)}
    pool = [f"c{i}" for i in range(20)] + [f"p{i}" for i in range(5)]   # duplicates beyond rank 20
    res = apply_filter(pool, "q", "x", ctx_for(texts))
    assert res.flagged == set() and all(res.scores[f"p{i}"] == 0.0 for i in range(5))
    flagged = {f"c{i}" for i in range(18)}
    assert survivors(pool, flagged) == ["c18", "c19", "p0", "p1", "p2"]
    assert backfill(pool, flagged) is True and backfill(pool, set()) is False


def test_tiny_pools():
    texts = diverse(1)
    assert apply_filter(["c0"], "q", "x", ctx_for(texts)).flagged == set()
    assert apply_filter([], "q", "x", ctx_for(texts)).flagged == set()
