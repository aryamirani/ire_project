"""Step 2: NQ loading, question join, selection and corpus building on a tiny synthetic BEIR zip (no network)."""

import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ire.data.nq import (
    BeirParquet,
    build_question_pool,
    choose_passages,
    iter_beir_corpus,
    load_beir_qrels,
    load_beir_queries,
    load_nq_open,
    normalize_question,
    scan_corpus,
    select_placeholder,
)
from ire.data.prepare import OutputPaths, prepare, verify_outputs
from ire.schemas import CorpusRecord, QuestionAllRecord, QuestionRecord, read_jsonl

N_DOCS = 40
# qid -> (BEIR query text, gold doc ids)
QUERIES = {f"test{i}": (f"who is person number {i}", [f"doc{2 * i}"] + ([f"doc{2 * i + 1}"] if i % 3 == 0 else []))
           for i in range(12)}
QUERIES["test12"] = ("a question nq open does not have", ["doc30"])   # no answer -> not eligible
QUERIES["test13"] = ("who is missing from the corpus", ["doc999"])   # gold not in corpus -> not eligible
QUERIES["test14"] = ("who wrote hamlet", ["doc31"])                  # answer only in nq_open train


def _nq_open_rows(split: str) -> list[dict]:
    if split == "train":
        return [{"question": "Who wrote Hamlet?", "answer": ["William Shakespeare", "Shakespeare"]}]
    rows = [{"question": f"Who is person number {i}?", "answer": [f"Person {i}"]} for i in range(12)]
    rows.append({"question": "who is person number 0", "answer": ["P. Zero", "Person 0"]})  # duplicate question
    rows.append({"question": "who is missing from the corpus", "answer": ["Nobody"]})
    return rows


@pytest.fixture
def raw(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """A BEIR-layout nq.zip plus nq_open validation/train parquet files."""
    zip_path = tmp_path / "raw" / "nq.zip"
    zip_path.parent.mkdir(parents=True)
    corpus = "".join(json.dumps({"_id": f"doc{i}", "title": f"Title {i}", "text": f"Passage text {i}.",
                                 "metadata": {}}) + "\n" for i in range(N_DOCS))
    queries = "".join(json.dumps({"_id": q, "text": t, "metadata": {}}) + "\n" for q, (t, _) in QUERIES.items())
    queries += json.dumps({"_id": "train0", "text": "a training query", "metadata": {}}) + "\n"  # not in test qrels
    qrels = "query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t1\n" for q, (_, docs) in QUERIES.items()
                                                    for d in docs)
    qrels += "test0\tdoc39\t0\n"  # score 0 is not relevant
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nq/corpus.jsonl", corpus)
        zf.writestr("nq/queries.jsonl", queries)
        zf.writestr("nq/qrels/test.tsv", qrels)
    paths = {}
    for split in ("validation", "train"):
        paths[split] = tmp_path / "raw" / f"nq_open_{split}.parquet"
        pq.write_table(pa.Table.from_pylist(_nq_open_rows(split)), paths[split])
    return zip_path, paths


def test_normalize_question() -> None:
    assert normalize_question("  Who wrote 'Hamlet'?  ") == "who wrote hamlet"
    assert normalize_question("WHO   WROTE\tHAMLET") == "who wrote hamlet"
    assert normalize_question("who’s the “king”, really?") == "whos the king really"


def test_readers(raw) -> None:
    zip_path, nq_paths = raw
    queries = load_beir_queries(zip_path)
    assert queries["test14"] == "who wrote hamlet" and "train0" in queries
    qrels = load_beir_qrels(zip_path)
    assert qrels["test0"] == ["doc0", "doc1"]  # score-0 row for doc39 ignored
    answers, stats = load_nq_open(nq_paths)
    assert answers["who is person number 0"] == ["Person 0", "P. Zero"]  # union, first-seen order, no dupes
    assert answers["who wrote hamlet"] == ["William Shakespeare", "Shakespeare"]
    assert stats["rows"] == {"validation": 14, "train": 1} and stats["questions_with_several_rows"] == 1


def test_scan_and_pool(raw) -> None:
    zip_path, nq_paths = raw
    qrels = load_beir_qrels(zip_path)
    scan = scan_corpus(zip_path, {d for docs in qrels.values() for d in docs})
    assert scan.n_passages == N_DOCS and not scan.duplicate_ids
    assert "doc999" not in scan.gold_positions and scan.gold_positions["doc31"] == 31
    answers, stats = load_nq_open(nq_paths)
    pool, join = build_question_pool(load_beir_queries(zip_path), qrels, answers, scan.gold_positions,
                                     stats["_split_of"])
    assert join["queries"] == 16 and join["queries_with_qrels"] == 15  # train0 is not a test query
    assert join["queries_with_gold_in_corpus"] == 14                   # test13's gold is missing
    assert join["queries_matched_to_nq_open"] == 14                    # test12 has no answer
    assert join["matched_by_split"] == {"validation": 13, "train": 1}
    assert [q.qid for q in pool] == [f"test{i}" for i in range(12)] + ["test14"]
    assert all(isinstance(q, QuestionAllRecord) for q in pool)


def test_scan_detects_duplicates(tmp_path: Path) -> None:
    zip_path = tmp_path / "dup.zip"
    lines = [{"_id": "a", "title": "", "text": "x"}, {"_id": "b", "title": "", "text": "y"},
             {"_id": "a", "title": "", "text": "z"}]
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nq/corpus.jsonl", "".join(json.dumps(r) + "\n" for r in lines))
    scan = scan_corpus(zip_path, {"a"})
    assert scan.n_passages == 3 and scan.duplicate_ids == ["a"] and scan.gold_positions == {"a": 0}


def test_select_placeholder_balanced_and_seeded() -> None:
    pool = [QuestionAllRecord(qid=f"q{i}", question=f"question {i}", answers=["a"], gold_doc_ids=[f"d{i}"])
            for i in range(30)]
    sel = select_placeholder(pool, n_target=6, n_clean=6, seed=13)
    assert all(isinstance(q, QuestionRecord) for q in sel)
    cells = {}
    for q in sel:
        cells[(q.role, q.fold)] = cells.get((q.role, q.fold), 0) + 1
    assert cells == {("target", "A"): 3, ("target", "B"): 3, ("clean", "A"): 3, ("clean", "B"): 3}
    assert len({q.qid for q in sel}) == 12
    assert sel == select_placeholder(pool, n_target=6, n_clean=6, seed=13)      # deterministic
    assert sel != select_placeholder(pool, n_target=6, n_clean=6, seed=14)      # seed matters
    with pytest.raises(ValueError, match="pool has"):
        select_placeholder(pool[:5], n_target=6, n_clean=6)
    with pytest.raises(ValueError, match="evenly"):
        select_placeholder(pool, n_target=5, n_clean=6)


def test_choose_passages() -> None:
    keep = choose_passages(1000, {3, 500, 999}, 50, seed=13)
    assert len(keep) == 50 and {3, 500, 999} <= keep and all(0 <= i < 1000 for i in keep)
    assert keep == choose_passages(1000, {3, 500, 999}, 50, seed=13)
    with pytest.raises(ValueError):
        choose_passages(1000, set(range(10)), 5)
    with pytest.raises(ValueError):
        choose_passages(10, set(), 11)


def _prepare(raw, tmp_path: Path, mode: str, **kw) -> tuple[OutputPaths, dict]:
    zip_path, nq_paths = raw
    out = OutputPaths.in_dir(tmp_path / f"out_{mode}_{kw.get('dev_questions')}")
    build = prepare(zip_path, nq_paths, out, mode, n_target=4, n_clean=4, seed=13, **kw)
    return out, build


def test_prepare_random_mode(raw, tmp_path: Path) -> None:
    out, build = _prepare(raw, tmp_path, "random", n_passages=20)
    assert build["eligible_questions"] == 13 and build["corpus_written"] == 20
    corpus = read_jsonl(out.corpus, CorpusRecord)
    questions = read_jsonl(out.questions, QuestionRecord)
    assert len(questions) == 8 and len(read_jsonl(out.questions_all, QuestionAllRecord)) == 13
    ids = {r.doc_id for r in corpus}
    assert all(d in ids for q in questions for d in q.gold_doc_ids)
    assert build["corpus_gold_passages"] + build["corpus_random_passages"] == 20
    result = verify_outputs(out, "random", 20, n_target=4, n_clean=4, pool_gold_required=False)
    assert result["ok"], result
    assert result["counts"]["role_fold"] == {"clean/A": 2, "clean/B": 2, "target/A": 2, "target/B": 2}
    # corpus keeps the BEIR order and the contract format
    first = json.loads(out.corpus.read_text(encoding="utf-8").splitlines()[0])
    assert set(first) == {"doc_id", "title", "text", "is_poison"} and first["is_poison"] is False


def test_prepare_is_deterministic(raw, tmp_path: Path) -> None:
    out1, _ = _prepare(raw, tmp_path / "a", "random", n_passages=20)
    out2, _ = _prepare(raw, tmp_path / "b", "random", n_passages=20)
    for p1, p2 in zip((out1.corpus, out1.questions, out1.questions_all), (out2.corpus, out2.questions, out2.questions_all)):
        assert p1.read_bytes() == p2.read_bytes()


def test_prepare_full_mode(raw, tmp_path: Path) -> None:
    out, build = _prepare(raw, tmp_path, "full")
    assert build["corpus_written"] == N_DOCS
    result = verify_outputs(out, "full", None, raw_unique_passages=N_DOCS, n_target=4, n_clean=4)
    assert result["ok"] and result["checks"]["questions_all_gold_in_corpus"]


def test_prepare_dev_subset(raw, tmp_path: Path) -> None:
    out, build = _prepare(raw, tmp_path, "random", n_passages=25, dev_questions=10)
    qall = read_jsonl(out.questions_all, QuestionAllRecord)
    assert len(qall) == 10 and {q.qid for q in read_jsonl(out.questions, QuestionRecord)} <= {q.qid for q in qall}
    result = verify_outputs(out, "random", 25, n_target=4, n_clean=4, pool_gold_required=True)
    assert result["ok"], result  # every gold of all 10 questions is in the corpus


def test_verify_catches_problems(raw, tmp_path: Path) -> None:
    out, _ = _prepare(raw, tmp_path, "random", n_passages=20)
    lines = out.corpus.read_text(encoding="utf-8").splitlines()
    gold = read_jsonl(out.questions, QuestionRecord)[0].gold_doc_ids[0]
    kept = [ln for ln in lines if json.loads(ln)["doc_id"] != gold]
    out.corpus.write_text("\n".join(kept + [kept[0]]) + "\n", encoding="utf-8")  # drop a gold, add a duplicate
    result = verify_outputs(out, "random", 20, n_target=4, n_clean=4, pool_gold_required=False)
    assert not result["ok"]
    assert not result["checks"]["no_duplicate_doc_ids"] and not result["checks"]["all_question_gold_in_corpus"]


def test_subset_mode_is_a_clear_todo(raw, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="steps 3-5"):
        _prepare(raw, tmp_path, "subset")


def test_parquet_source_matches_zip(raw, tmp_path: Path) -> None:
    """The Hugging Face layout (corpus/queries parquet + qrels TSV) gives byte-identical outputs to the zip."""
    zip_path, nq_paths = raw
    hf = tmp_path / "hf"
    (hf / "qrels").mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        corpus = [json.loads(ln) for ln in zf.read("nq/corpus.jsonl").decode("utf-8").splitlines()]
        queries = [json.loads(ln) for ln in zf.read("nq/queries.jsonl").decode("utf-8").splitlines()]
        (hf / "qrels" / "test.tsv").write_bytes(zf.read("nq/qrels/test.tsv"))
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ("_id", "title", "text")} for r in corpus]),
                   hf / "corpus.parquet", row_group_size=7)  # several row groups: exercises batch streaming
    pq.write_table(pa.Table.from_pylist([{"_id": q["_id"], "title": "", "text": q["text"]} for q in queries]),
                   hf / "queries.parquet")
    src = BeirParquet(hf / "corpus.parquet", hf / "queries.parquet", hf / "qrels")
    assert list(iter_beir_corpus(src)) == [{k: r[k] for k in ("_id", "title", "text")} for r in corpus]
    out_zip, _ = _prepare(raw, tmp_path / "z", "random", n_passages=20)
    out_hf = OutputPaths.in_dir(tmp_path / "p")
    prepare(src, nq_paths, out_hf, "random", 20, seed=13, n_target=4, n_clean=4)
    for a, b in zip((out_zip.corpus, out_zip.questions, out_zip.questions_all),
                    (out_hf.corpus, out_hf.questions, out_hf.questions_all)):
        assert a.read_bytes() == b.read_bytes()


def test_prepare_uses_frozen_question_set(raw, tmp_path: Path) -> None:
    """With Arya's frozen selection, questions.jsonl is that set (in its order) and the random corpus holds its gold."""
    full, _ = _prepare(raw, tmp_path / "full", "full")
    pool = read_jsonl(full.questions_all, QuestionAllRecord)
    picked = list(reversed(pool))[:8]
    cells = [("target", "A"), ("target", "B"), ("clean", "A"), ("clean", "B")]
    frozen = {"questions_all_sha256": "x", "entries": [
        {"qid": q.qid, "role": cells[i % 4][0], "fold": cells[i % 4][1], "answer_type": "other"}
        for i, q in enumerate(picked)]}
    out, build = _prepare(raw, tmp_path / "frozen", "random", n_passages=20, question_set=frozen)
    qs = read_jsonl(out.questions, QuestionRecord)
    assert [q.qid for q in qs] == [q.qid for q in picked]
    assert [(q.role, q.fold) for q in qs] == [cells[i % 4] for i in range(8)]
    assert build["questions_source"] == "frozen question_set"
    assert verify_outputs(out, "random", 20, n_target=4, n_clean=4, pool_gold_required=False)["ok"]
