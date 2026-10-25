"""Step 6: reciprocal rank fusion on hand-built lists (no indexes, no models)."""

import pytest

from ire.retrieval.rrf import RRFRetriever, fuse, fuse_many
from ire.schemas import is_poison_id


def ids(fused):
    return [d for d, _ in fused]


def test_hand_built_order_with_docs_in_one_list_only() -> None:
    a = [("d1", 9.0), ("d2", 8.0), ("d3", 7.0)]
    b = [("d3", 0.9), ("d1", 0.8), ("d4", 0.7)]
    fused = fuse(a, b, k=60)
    # d1 = 1/61 + 1/62, d3 = 1/63 + 1/61, d2 = 1/62 (only in a), d4 = 1/63 (only in b)
    assert ids(fused) == ["d1", "d3", "d2", "d4"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62) and fused[1][1] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2][1] == pytest.approx(1 / 62) and fused[3][1] == pytest.approx(1 / 63)


def test_ties_broken_by_doc_id() -> None:
    a = [("y", 1.0), ("x", 0.5), ("only_a", 0.1)]
    b = [("x", 1.0), ("y", 0.5), ("only_b", 0.1)]
    fused = fuse(a, b)
    assert ids(fused) == ["x", "y", "only_a", "only_b"]  # x == y and only_a == only_b exactly; doc_id decides
    assert fused[0][1] == fused[1][1] and fused[2][1] == fused[3][1]


def test_empty_lists() -> None:
    b = [("d1", 3.0), ("d2", 2.0)]
    assert fuse([], b) == [("d1", 1 / 61), ("d2", 1 / 62)]
    assert fuse(b, []) == fuse([], b)
    assert fuse([], []) == [] and fuse_many([]) == []


def test_short_bm25_list_against_full_dense_list() -> None:
    bm25 = [("d5", 12.0), ("d0", 11.0), ("x_only_bm25", 3.0)]          # BM25 found only 3 matching docs
    dense = [(f"d{i}", 1.0 - i / 1000) for i in range(150)]
    fused = fuse(bm25, dense, depth=150)
    assert len(fused) == 150                                           # 151 distinct docs, cut to depth
    assert ids(fused)[:3] == ["d0", "d5", "d1"]                        # in both lists -> on top
    assert "x_only_bm25" in ids(fused)                                 # 1/63 beats dense-only docs from rank 4 down
    assert ids(fuse(bm25, dense, depth=None))[-1] == "d149"


def test_identical_lists_keep_their_order() -> None:
    a = [(f"d{i}", 100.0 - i) for i in range(10)]
    fused = fuse(a, list(a))
    assert ids(fused) == ids(a)
    assert [s for _, s in fused] == pytest.approx([2 / (60 + r) for r in range(1, 11)])


def test_input_scores_are_ignored_and_json_lists_work() -> None:
    a = [("d1", 0.1), ("d2", 99.0)]
    b = [["d2", 5], ["d1", -3]]                                        # lists, as read from JSON
    assert fuse(a, b) == fuse([("d1", 1e9), ("d2", -1e9)], [("d2", 0.0), ("d1", 0.0)])


def test_depth_and_k() -> None:
    a = [(f"a{i}", 0.0) for i in range(5)]
    b = [(f"b{i}", 0.0) for i in range(5)]
    assert len(fuse(a, b, depth=3)) == 3 and len(fuse(a, b, depth=None)) == 10
    assert fuse(a, b, k=1)[0][1] == pytest.approx(1 / 2)
    with pytest.raises(ValueError):
        fuse(a, b, k=0)
    with pytest.raises(ValueError, match="twice"):
        fuse([("d1", 1.0), ("d1", 0.5)], b)


def test_fuse_many_and_list_order_invariance() -> None:
    a = [("d1", 0), ("d2", 0), ("d3", 0)]
    b = [("d3", 0), ("d2", 0)]
    c = [("d2", 0), ("d4", 0)]
    three = fuse_many([a, b, c])
    assert ids(three)[0] == "d2"                                       # in all three lists
    assert three == fuse_many([c, a, b]) == fuse_many([b, c, a])      # fsum: exact, order-independent
    assert fuse(a, b) == fuse_many([a, b])


def test_mask_then_fuse_differs_from_fuse_then_remove() -> None:
    """Why masking must come BEFORE fusing (plan 2.1).

    A poison passage sits above clean passages in both lists. Masking each list first moves the clean passages
    up by one rank where the poison was above them, and RRF scores depend on those ranks. If we fuse first and
    delete the poison afterwards, the clean passages keep the ranks they had *with* the poison present, which is
    not what an index without that poison would give: here c2 and c3 swap places.
    """
    poison = "poison_q1_fact_1"
    bm25 = [(poison, 0), ("c1", 0), ("c2", 0), ("c3", 0), ("c4", 0)]
    dense = [("c1", 0), ("c3", 0), (poison, 0), ("c2", 0), ("c4", 0)]

    def mask(ranked):
        return [(d, s) for d, s in ranked if not is_poison_id(d)]

    mask_then_fuse = ids(fuse(mask(bm25), mask(dense)))
    fuse_then_remove = [d for d in ids(fuse(bm25, dense)) if not is_poison_id(d)]
    assert mask_then_fuse == ["c1", "c2", "c3", "c4"]      # c2: ranks (2, 3), c3: ranks (3, 2) -> tie, doc_id
    assert fuse_then_remove == ["c1", "c3", "c2", "c4"]    # c2: ranks (3, 4) < c3: ranks (4, 2)
    assert mask_then_fuse != fuse_then_remove

    # Depth makes it worse: removing poison from a fused list that was already cut leaves it short.
    assert len(fuse(mask(bm25), mask(dense), depth=3)) == 3
    assert len([d for d in ids(fuse(bm25, dense, depth=3)) if not is_poison_id(d)]) == 2


class FakeSearcher:
    def __init__(self, lists):
        self.lists = lists
        self.depths = []

    def search(self, queries, k):
        self.depths.append(k)
        return [self.lists[q][:k] for q in queries]


def test_rrf_retriever_fuses_components() -> None:
    bm25 = FakeSearcher({"q": [("d1", 5.0), ("d2", 4.0)], "empty": []})
    dense = FakeSearcher({"q": [("d2", 0.9), ("d3", 0.8)], "empty": [("d9", 0.1)]})
    r = RRFRetriever({"bm25": bm25, "dense": dense}, k=60, depth=150)
    res = r.search(["q", "empty"], k=2)
    assert res[0] == fuse(bm25.lists["q"], dense.lists["q"], depth=2) and ids(res[0]) == ["d2", "d1"]
    assert res[1] == [("d9", 1 / 61)]
    assert bm25.depths == [150] and dense.depths == [150]              # components always go to full depth
    assert r.search([], k=5) == [] and r.search(["q"], k=0) == [[]]
    comps = r.search_components(["q"])
    assert r.fuse_components(comps) == [fuse(bm25.lists["q"], dense.lists["q"])]
    with pytest.raises(ValueError):
        RRFRetriever({"bm25": bm25})
