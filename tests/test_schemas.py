"""Step 1: every section-2 record round-trips through JSONL; missing fields, wrong types and bad values raise."""

import json
from pathlib import Path

import pytest

from ire.schemas import (
    CorpusRecord,
    FilterContext,
    FilterResult,
    PoisonRecord,
    QuestionRecord,
    RetrievalCacheRecord,
    SchemaError,
    is_poison_id,
    make_poison_id,
    parse_poison_id,
    read_jsonl,
    write_jsonl,
)

VALID = {
    CorpusRecord: {"doc_id": "doc0", "title": "Hamlet", "text": "Hamlet is a tragedy by Shakespeare.",
                   "is_poison": False},
    PoisonRecord: {"doc_id": "poison_q123_fact_1", "qid": "q123", "attack": "fact", "title": "",
                   "text": "Who wrote Hamlet? Hamlet was written by Marlowe.", "is_poison": True,
                   "target_answer": "Marlowe"},
    QuestionRecord: {"qid": "q123", "question": "who wrote hamlet", "answers": ["William Shakespeare"],
                     "gold_doc_ids": ["doc0"], "role": "target", "fold": "A"},
    RetrievalCacheRecord: {"qid": "q123", "retriever": "dense",
                           "top150": [["doc0", 0.83], ["poison_q123_fact_1", 0.81], ["doc7", 0.5]]},
}

# A value of the wrong type for every field (bool vs int/float is deliberately strict).
WRONG_TYPE = {
    CorpusRecord: {"doc_id": 7, "title": None, "text": ["x"], "is_poison": 0},
    PoisonRecord: {"doc_id": None, "qid": 123, "attack": 1, "title": 0, "text": 1.5, "is_poison": "true",
                   "target_answer": ["Marlowe"]},
    QuestionRecord: {"qid": 123, "question": None, "answers": "William Shakespeare", "gold_doc_ids": [0],
                     "role": True, "fold": 1},
    RetrievalCacheRecord: {"qid": 1, "retriever": None, "top150": {"doc0": 0.83}},
}

CASES = [(cls, f) for cls in VALID for f in VALID[cls]]
IDS = [f"{cls.__name__}.{f}" for cls, f in CASES]


@pytest.mark.parametrize("cls", list(VALID), ids=lambda c: c.__name__)
def test_valid_record_round_trip(cls, tmp_path: Path) -> None:
    rec = cls.from_dict(VALID[cls])
    path = tmp_path / "records.jsonl"
    assert write_jsonl(path, [rec, rec]) == 2
    back = read_jsonl(path, cls)
    assert back == [rec, rec]
    # on disk it is exactly the contract's JSON (tuples written as lists)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0]) == VALID[cls]


@pytest.mark.parametrize(("cls", "field"), CASES, ids=IDS)
def test_missing_field_raises(cls, field: str) -> None:
    data = {k: v for k, v in VALID[cls].items() if k != field}
    with pytest.raises(SchemaError, match=f"missing field.*{field}"):
        cls.from_dict(data)


@pytest.mark.parametrize(("cls", "field"), CASES, ids=IDS)
def test_wrong_type_raises(cls, field: str) -> None:
    data = dict(VALID[cls], **{field: WRONG_TYPE[cls][field]})
    with pytest.raises(SchemaError):
        cls.from_dict(data)


@pytest.mark.parametrize("cls", list(VALID), ids=lambda c: c.__name__)
def test_unknown_field_raises(cls) -> None:
    with pytest.raises(SchemaError, match="unknown field"):
        cls.from_dict(dict(VALID[cls], trust_tier=1))


@pytest.mark.parametrize(("cls", "field", "value"), [
    (CorpusRecord, "is_poison", True),
    (CorpusRecord, "doc_id", "poison_q1_fact_1"),
    (CorpusRecord, "doc_id", ""),
    (PoisonRecord, "is_poison", False),
    (PoisonRecord, "attack", "hijack"),            # disagrees with the doc_id
    (PoisonRecord, "attack", "jamming"),           # not an allowed attack
    (PoisonRecord, "qid", "q999"),                 # disagrees with the doc_id
    (PoisonRecord, "doc_id", "poison_q123_fact"),  # malformed id
    (PoisonRecord, "target_answer", ""),
    (QuestionRecord, "role", "poison"),
    (QuestionRecord, "fold", "C"),
    (QuestionRecord, "answers", ["ok", 3]),
    (QuestionRecord, "question", ""),
    (RetrievalCacheRecord, "retriever", "colbert"),
    (RetrievalCacheRecord, "top150", [["doc0", 0.1], ["doc1", 0.9]]),          # not sorted descending
    (RetrievalCacheRecord, "top150", [["doc0", 0.9], ["doc0", 0.8]]),          # duplicate doc_id
    (RetrievalCacheRecord, "top150", [["doc0", 0.9, 1]]),                      # not a pair
    (RetrievalCacheRecord, "top150", [["doc0", True]]),                        # bool is not a score
    (RetrievalCacheRecord, "top150", [["doc0", float("nan")]]),                # non-finite score
    (RetrievalCacheRecord, "top150", [[f"d{i}", 1.0 - i / 1000] for i in range(151)]),  # > 150
], ids=lambda v: str(v)[:40])
def test_bad_values_raise(cls, field: str, value) -> None:
    with pytest.raises(SchemaError):
        cls.from_dict(dict(VALID[cls], **{field: value}))


def test_int_scores_accepted_and_ties_allowed() -> None:
    rec = RetrievalCacheRecord.from_dict({"qid": "q1", "retriever": "bm25", "top150": [["a", 3], ["b", 3], ["c", 0]]})
    assert rec.top150 == [("a", 3.0), ("b", 3.0), ("c", 0.0)]


def test_direct_construction_is_validated() -> None:
    with pytest.raises(SchemaError, match="QuestionRecord.fold"):
        QuestionRecord(qid="q1", question="x", answers=[], gold_doc_ids=[], role="clean", fold="Z")


def test_read_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    good = json.dumps(VALID[CorpusRecord])
    bad = json.dumps({k: v for k, v in VALID[CorpusRecord].items() if k != "text"})
    path.write_text(f"{good}\n\n{bad}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"corpus.jsonl line 3: CorpusRecord: missing field\(s\) \['text'\]"):
        read_jsonl(path, CorpusRecord)
    path.write_text(f"{good}\n{{not json\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="line 2: invalid JSON"):
        read_jsonl(path, CorpusRecord)


def test_write_jsonl_rejects_plain_dicts(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="Record instances"):
        write_jsonl(tmp_path / "x.jsonl", [VALID[CorpusRecord]])


def test_write_jsonl_append(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    rec = QuestionRecord.from_dict(VALID[QuestionRecord])
    write_jsonl(path, [rec])
    write_jsonl(path, [rec], append=True)
    assert len(read_jsonl(path, QuestionRecord)) == 2
    assert not (tmp_path / "q.jsonl.tmp").exists()


def test_poison_id_helpers() -> None:
    assert make_poison_id("q123", "fact", 1) == "poison_q123_fact_1"
    assert parse_poison_id("poison_q_12_x_hijack_3") == ("q_12_x", "hijack", 3)  # qid may contain "_"
    assert is_poison_id("poison_q1_url_2") and not is_poison_id("doc0")
    with pytest.raises(SchemaError):
        make_poison_id("q1", "jamming", 1)


def test_filter_result_and_context() -> None:
    res = FilterResult(flagged={"poison_q1_fact_1"}, scores={"doc0": 0.1, "poison_q1_fact_1": 0.97})
    assert "poison_q1_fact_1" in res.flagged
    with pytest.raises(SchemaError):
        FilterResult(flagged=["doc0"], scores={})  # must be a set
    with pytest.raises(SchemaError):
        FilterResult(flagged=set(), scores={"doc0": "high"})
    ctx = FilterContext(texts={"doc0": "text"}, vectors=None, config={})
    assert ctx.thresholds == {}
