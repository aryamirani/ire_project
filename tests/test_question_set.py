"""Arya: final question set selection (src/ire/data/question_set.py)."""

from __future__ import annotations

from collections import Counter

import pytest

from ire.data.question_set import (
    answer_type,
    build_questions,
    is_eligible,
    read_frozen,
    select_question_set,
    shortest_answer,
    write_frozen,
)
from ire.schemas import QuestionAllRecord

ANSWERS = ["1990", "March 4, 1801", "seven", "Tom Hanks", "Paris", "photosynthesis"]
STEMS = ["when was the treaty signed", "when did the war end", "how many moons does mars have",
         "who played the lead role", "where is the tower located", "what process do plants use"]


def make_pool(n: int) -> list[QuestionAllRecord]:
    return [QuestionAllRecord(qid=f"test{i}", question=f"{STEMS[i % 6]} number {i}",
                              answers=[ANSWERS[i % 6]], gold_doc_ids=[f"doc{i}"]) for i in range(n)]


@pytest.mark.parametrize("q,a,expected", [
    ("when was hamlet written", "1601", "year"),
    ("when did the war end", "11 November 1918", "date"),
    ("how many players on a team", "eleven", "number"),
    ("what is the population of x", "1,200,000", "number"),
    ("who wrote hamlet", "William Shakespeare", "person"),
    ("where is the eiffel tower", "Paris", "place"),
    ("what is the capital currency", "euro", "other"),
])
def test_answer_type(q, a, expected):
    assert answer_type(q, a) == expected


def test_eligibility_rules():
    ok = QuestionAllRecord(qid="1", question="who wrote the play hamlet", answers=["Shakespeare"], gold_doc_ids=["d"])
    assert is_eligible(ok) == (True, "ok")
    long = QuestionAllRecord(qid="2", question="what is the motto of x", answers=["a b c d e f g"], gold_doc_ids=["d"])
    assert is_eligible(long)[1] == "answer_too_long"
    yes = QuestionAllRecord(qid="3", question="is the earth round really", answers=["yes"], gold_doc_ids=["d"])
    assert is_eligible(yes)[1] == "yes_no"
    inq = QuestionAllRecord(qid="4", question="which city is paris the capital of", answers=["Paris"],
                            gold_doc_ids=["d"])
    assert is_eligible(inq)[1] == "answer_in_question"
    assert shortest_answer(["William Shakespeare", "Shakespeare"]) == "Shakespeare"


def test_selection_balanced_deterministic_and_stratified():
    pool = make_pool(1200)
    a, stats = select_question_set(pool, 200, 200, ("A", "B"), seed=13)
    b, _ = select_question_set(pool, 200, 200, ("A", "B"), seed=13)
    c, _ = select_question_set(pool, 200, 200, ("A", "B"), seed=14)
    assert a == b and a != c
    assert len({e["qid"] for e in a}) == 400
    cells = Counter((e["role"], e["fold"]) for e in a)
    assert set(cells.values()) == {100}
    # every answer type is spread over the four cells within +/- 1
    for typ in {e["answer_type"] for e in a}:
        per_cell = [sum(1 for e in a if e["answer_type"] == typ and (e["role"], e["fold"]) == c) for c in cells]
        assert max(per_cell) - min(per_cell) <= 1
    assert stats["eligible"] == 1200


def test_frozen_round_trip(tmp_path):
    pool = make_pool(500)
    entries, _ = select_question_set(pool, 200, 200, ("A", "B"), seed=13)
    path = tmp_path / "question_set.json"
    write_frozen(path, entries, "abc", {"seed": 13})
    frozen = read_frozen(path)
    assert frozen["entries"] == entries and frozen["questions_all_sha256"] == "abc"
    qs = build_questions(frozen, pool)
    assert [q.qid for q in qs] == [e["qid"] for e in entries]
    assert all(q.role == e["role"] and q.fold == e["fold"] for q, e in zip(qs, entries))
    with pytest.raises(ValueError):
        build_questions(frozen, pool[:10])


def test_not_enough_questions():
    with pytest.raises(ValueError):
        select_question_set(make_pool(100), 200, 200)
