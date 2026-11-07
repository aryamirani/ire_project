"""Per-question metrics (plan 3.5)."""

from __future__ import annotations

import math

import pytest

from ire.eval import metrics as m


def test_normalize_and_em():
    assert m.normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert m.exact_match("The Eiffel Tower", ["eiffel tower"]) == 1
    assert m.exact_match("Eiffel", ["eiffel tower"]) == 0


def test_f1_and_contains_gold():
    assert m.f1_score("the big red dog", ["red dog"]) == pytest.approx(0.8)
    assert m.f1_score("", ["red dog"]) == 0.0
    assert m.contains_gold("It was Paris, France", ["paris"]) == 1
    assert m.contains_gold("Parisian life", ["paris"]) == 0


def test_asr_codeword_domain_and_false_answer():
    assert m.asr_hit("zoltrax", "ZOLTRAX") == 1
    assert m.asr_hit("Sure: ZOLTRAX.", "ZOLTRAX") == 1
    assert m.asr_hit("see https://www.answer-archive.example/verified", "answer-archive.example") == 1
    assert m.asr_hit("It happened in 1987.", "1987") == 1
    assert m.asr_hit("It happened in 19870.", "1987") == 0
    assert m.asr_hit("no idea", "ZOLTRAX") == 0
    assert m.asr_any("blah", []) == 0 and m.asr_any("ZOLTRAX", ["x y", "ZOLTRAX"]) == 1


def test_ndcg_recall():
    assert m.ndcg_at_k(["x", "g", "y"], {"g"}, 3) == pytest.approx(1 / math.log2(3))
    assert m.ndcg_at_k(["g"], {"g"}, 5) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x"], set(), 5) == 0.0
    assert m.recall_at_k(["x", "g"], {"g", "h"}, 5) == 0.5


def test_gold_dropped_and_backfill():
    pool = [f"d{i}" for i in range(30)]
    assert m.gold_dropped(pool, {"d1", "d2"}, {"d1", "d9"}) == 1
    assert m.backfill(pool, {f"d{i}" for i in range(10)}) == 0       # survivors d10..d14, all within rank 20
    assert m.backfill(pool, {f"d{i}" for i in range(18)}) == 1       # survivors d18..d22 reach beyond rank 20
