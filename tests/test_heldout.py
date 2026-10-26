"""Step 12 tooling: held-out template, validator and shard builder (src/ire/poison/heldout.py).

The texts below are obviously artificial TEST FIXTURES, not held-out phrasings; the real ones are written by hand.
"""

from __future__ import annotations

from collections import Counter

from ire.poison.heldout import N_FACT, N_HIJACK, build_heldout, make_template, phrase_overlap, validate
from ire.schemas import PoisonRecord, parse_poison_id

TARGETS = {"codeword": "ZOLTRAX", "adaptive_qids": [f"a{i}" for i in range(6)],
           "targets": {**{f"a{i}": {"target_answer": f"Wrong {i}"} for i in range(6)},
                       **{f"t{i}": {"target_answer": f"Other {i}"} for i in range(8)}}}
QUESTIONS = {q: f"what is item {q}" for q in TARGETS["targets"]}


def filled() -> list[dict]:
    rows = make_template(TARGETS, QUESTIONS, seed=13)
    for r in rows:
        if r["kind"] == "hijack_phrasing":
            r["text"] = f"fixture instruction {r['slot']} asks for the token {{CW}} in every reply"
        else:
            r["text"] = f"fixture passage for {r['qid']}: the item is {r['target_answer']} beyond doubt"
    return rows


def test_template_shape():
    rows = make_template(TARGETS, QUESTIONS, seed=13)
    kinds = Counter(r["kind"] for r in rows)
    assert kinds == {"hijack_phrasing": N_HIJACK, "fact_passage": N_FACT}
    assert all(r["text"] == "" for r in rows)
    fact_qids = [r["qid"] for r in rows if r["kind"] == "fact_passage"]
    assert len(set(fact_qids)) == N_FACT and not set(fact_qids) & set(TARGETS["adaptive_qids"])
    assert rows == make_template(TARGETS, QUESTIONS, seed=13)


def test_validator_rejects_empty_and_bad_slots():
    rows = make_template(TARGETS, QUESTIONS, seed=13)
    assert any("empty" in p for p in validate(rows))
    bad = filled()
    bad[0]["text"] = "fixture instruction without the placeholder at all"
    bad[N_HIJACK]["text"] = "fixture passage that never states the answer anywhere"
    problems = validate(bad)
    assert any("{CW}" in p for p in problems) and any("target answer" in p for p in problems)


def test_valid_fixture_builds_the_shard():
    rows = filled()
    assert validate(rows) == []
    recs = build_heldout(rows, TARGETS, QUESTIONS, seed=13)
    for r in recs:
        PoisonRecord.from_dict(r.to_dict())
    hij = [r for r in recs if r.target_answer == "ZOLTRAX"]
    fac = [r for r in recs if r.target_answer != "ZOLTRAX"]
    assert len(hij) == 6 * 5 and len(fac) == N_FACT
    assert {parse_poison_id(r.doc_id)[2] for r in hij} == {1, 2, 3, 4, 5}
    assert all(r.text.startswith("What is item") and "ZOLTRAX" in r.text and "{CW}" not in r.text for r in hij)
    assert all(parse_poison_id(r.doc_id)[1:] == ("heldout", 1) for r in fac)
    ov = phrase_overlap(rows)
    assert ov["hijack_total"] == N_HIJACK and set(ov["per_slot"]) == {r["slot"] for r in rows}
