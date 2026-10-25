"""Step 5: SPLADE index and exact sparse retriever.

Most tests use a fake sparse encoder (hashed bag of words over a 64-term vocabulary) and need no model. The
SPLADE tests load naver/splade-cocondenser-ensembledistil on CPU and are skipped if it is not in the HF cache.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from ire.config import load_config
from ire.retrieval.splade import (
    PROGRESS,
    SpladeEncoder,
    SpladeRetriever,
    SpladeSettings,
    build_splade_index,
    index_is_current,
)
from ire.schemas import CorpusRecord, PoisonRecord, write_jsonl

DOCS = [
    ("d0", "Hamlet", "Hamlet is a tragedy written by William Shakespeare around 1600."),
    ("d1", "Macbeth", "Macbeth is a tragedy by Shakespeare about a Scottish general."),
    ("d2", "Zebra", "Zebras are African equines with black and white striped coats."),
    ("d3", "Volcano", "A volcano is a rupture in the crust of a planet."),
    ("d4", "Python (programming language)", "Python is a programming language created by Guido van Rossum."),
    ("d5", "Shakespeare", "William Shakespeare was an English playwright, poet and actor."),
    ("d6", "Kilimanjaro", "Mount Kilimanjaro is a dormant volcano in Tanzania."),
]
SETTINGS = SpladeSettings(model="fake-splade")


class FakeSparseEncoder:
    """Hashed bag of words: weight = log(1 + count) per hashed term; deterministic and non-negative."""

    name, revision, vocab_size = "fake-splade", "fake-rev-1", 64

    def __init__(self, fail_after_calls: int | None = None) -> None:
        self.calls = 0
        self.fail_after_calls = fail_after_calls

    def _matrix(self, texts) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, text in enumerate(texts):
            counts: dict[int, int] = {}
            for w in re.findall(r"\w+", text.lower()):
                j = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.vocab_size
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i), cols.append(j), vals.append(np.log1p(c))
        return sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(texts), self.vocab_size))

    def encode_documents(self, texts):
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("simulated crash")
        return self._matrix(texts)

    def encode_queries(self, texts):
        return self._matrix(texts)

    def has_content(self, texts):
        return [bool(re.search(r"\w", t)) for t in texts]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [CorpusRecord(doc_id=d, title=t, text=x, is_poison=False) for d, t, x in DOCS])
    return path


@pytest.fixture
def index(corpus: Path, tmp_path: Path) -> SpladeRetriever:
    build_splade_index(corpus, tmp_path / "splade", FakeSparseEncoder(), SETTINGS)
    return SpladeRetriever(tmp_path / "splade", encoder=FakeSparseEncoder(), mmap=False)


# ---------------------------------------------------------------- config


@pytest.mark.parametrize("env", ["local", "ada"])
def test_lengths_and_pooling_in_config(env: str) -> None:
    s = SpladeSettings.from_config(load_config(env))
    assert (s.model, s.doc_max_length, s.query_max_length, s.pooling) == (
        "naver/splade-cocondenser-ensembledistil", 256, 64, "max_log1p_relu")


def test_pruning_is_refused() -> None:
    class Cfg(dict):
        def __getitem__(self, key):
            return {"splade": {"prune_doc_terms": True}, "models.splade": "x"}[key]

    with pytest.raises(ValueError, match="exact"):
        SpladeSettings.from_config(Cfg())


# ---------------------------------------------------------------- index + retriever (fake encoder)


def test_storage(index: SpladeRetriever, tmp_path: Path) -> None:
    names = sorted(p.name for p in (tmp_path / "splade").iterdir())
    assert names == ["doc_ids.npy", "doc_matrix.data.npy", "doc_matrix.indices.npy", "doc_matrix.indptr.npy",
                     "inverted.data.npy", "inverted.indices.npy", "inverted.indptr.npy", "manifest.json"]
    assert index.docs.shape == (len(DOCS), 64) and index.inverted.shape == (64, len(DOCS))
    assert index.docs.data.min() > 0 and index.docs.dtype == np.float32
    assert (index.inverted.T != index.docs).nnz == 0  # the inverted index is exactly the transpose
    assert index.manifest["nnz"] == index.docs.nnz and index.manifest["nnz_per_doc"]["empty_rows"] == 0


def test_search_is_exact_dot_product(index: SpladeRetriever) -> None:
    queries = ["who wrote hamlet", "dormant volcano tanzania", "programming language python", "william shakespeare"]
    q = index.query_vectors(queries).toarray()
    full = index.docs.toarray() @ q.T                                   # manual dot products, [docs, queries]
    for i, hits in enumerate(index.search(queries, k=len(DOCS))):
        expected = sorted(((index.doc_ids[j], float(full[j, i])) for j in range(len(DOCS)) if full[j, i] > 0),
                          key=lambda x: (-x[1], index.doc_ids.index(x[0])))
        assert [d for d, _ in hits] == [d for d, _ in expected]
        np.testing.assert_allclose([s for _, s in hits], [s for _, s in expected], rtol=1e-6)
        np.testing.assert_allclose(index.score_docs(queries[i], [d for d, _ in hits]), [s for _, s in hits],
                                   rtol=1e-6)


def test_only_matching_docs_and_k(index: SpladeRetriever) -> None:
    (hits,) = index.search(["zebras"], k=100)
    assert hits and all(s > 0 for _, s in hits) and len(hits) < len(DOCS)
    assert index.search(["zebras"], k=0) == [[]] and index.search([], k=5) == []
    missing = [d for d, _, _ in DOCS if d not in {h for h, _ in hits}]
    assert missing and index.score_docs("zebras", missing).tolist() == [0.0] * len(missing)


@pytest.mark.parametrize("query", ["", "   ", "?!"])
def test_queries_without_tokens_return_nothing(index: SpladeRetriever, query: str) -> None:
    assert index.search([query], k=10) == [[]]
    assert index.score_docs(query, ["d0"]).tolist() == [0.0]


def test_ties_ordered_by_index_and_boundary(tmp_path: Path) -> None:
    path = tmp_path / "ties.jsonl"
    write_jsonl(path, [CorpusRecord(doc_id=f"t{i}", title="", text="same words", is_poison=False) for i in range(6)])
    build_splade_index(path, tmp_path / "ti", FakeSparseEncoder(), SETTINGS)
    r = SpladeRetriever(tmp_path / "ti", encoder=FakeSparseEncoder(), mmap=False)
    assert [d for d, _ in r.search(["same"], k=6)[0]] == [f"t{i}" for i in range(6)]
    assert [d for d, _ in r.search(["same"], k=3)[0]] == ["t0", "t1", "t2"]  # ties cut by index, not arbitrarily


def test_deterministic_and_unknown_ids(index: SpladeRetriever) -> None:
    qs = ["shakespeare tragedy", "volcano"]
    assert index.search(qs, k=5) == index.search(qs, k=5)
    with pytest.raises(KeyError):
        index.score_docs("volcano", ["nope"])


def test_poison_shards_appended(corpus: Path, tmp_path: Path) -> None:
    shard = tmp_path / "poison" / "poison_fact.jsonl"
    write_jsonl(shard, [PoisonRecord(doc_id="poison_q1_fact_1", qid="q1", attack="fact", title="",
                                     text="zebras zebras striped", is_poison=True, target_answer="x")])
    manifest = build_splade_index(corpus, tmp_path / "p", FakeSparseEncoder(), SETTINGS, extra_shards=[shard])
    assert (manifest["n_docs"], manifest["n_poison"]) == (len(DOCS) + 1, 1)
    r = SpladeRetriever(tmp_path / "p", encoder=FakeSparseEncoder(), mmap=False)
    top = r.search(["zebras striped"], k=1)[0][0][0]
    assert top == "poison_q1_fact_1" and r.is_poison(top)


def test_resume_after_crash(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_splade_index(corpus, out, FakeSparseEncoder(fail_after_calls=2), SETTINGS, checkpoint_every=3)
    assert json.loads((tmp_path / "resume.building" / PROGRESS).read_text(encoding="utf-8"))["rows_done"] == 6
    enc = FakeSparseEncoder()
    manifest = build_splade_index(corpus, out, enc, SETTINGS, checkpoint_every=3)
    assert manifest["resumed_from_row"] == 6 and enc.calls == 1
    build_splade_index(corpus, tmp_path / "fresh", FakeSparseEncoder(), SETTINGS, checkpoint_every=3)
    a = SpladeRetriever(out, encoder=FakeSparseEncoder(), mmap=False).docs
    b = SpladeRetriever(tmp_path / "fresh", encoder=FakeSparseEncoder(), mmap=False).docs
    assert (a != b).nnz == 0 and not (tmp_path / "resume.building").exists()


def test_index_is_current(corpus: Path, tmp_path: Path) -> None:
    build_splade_index(corpus, tmp_path / "s", FakeSparseEncoder(), SETTINGS)
    assert index_is_current(tmp_path / "s", corpus, [], SETTINGS, "fake-rev-1")
    assert not index_is_current(tmp_path / "s", corpus, [], replace(SETTINGS, doc_max_length=128), "fake-rev-1")
    assert not index_is_current(tmp_path / "s", corpus, [], SETTINGS, "other-rev")


def test_encoder_revision_must_match(index: SpladeRetriever) -> None:
    other = FakeSparseEncoder()
    other.revision = "different"
    index._encoder = other
    with pytest.raises(ValueError, match="rebuild with --force"):
        index.search(["volcano"], k=1)


# ---------------------------------------------------------------- real SPLADE (skipped if not cached)


def _splade_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("naver/splade-cocondenser-ensembledistil", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_splade = pytest.mark.skipif(not _splade_cached(), reason="SPLADE model not in the local HF cache")


@pytest.fixture(scope="module")
def splade() -> SpladeEncoder:
    return SpladeEncoder(SpladeSettings(model="naver/splade-cocondenser-ensembledistil", encode_dtype="float32"),
                         device="cpu", batch_size=4)


@needs_splade
def test_pooling_matches_textbook_formula(splade: SpladeEncoder) -> None:
    import torch

    texts = ["Hamlet is a tragedy by William Shakespeare.", "who wrote hamlet", "a much longer passage " * 5]
    enc = splade.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.inference_mode():
        logits = splade.model(**enc).logits
        ref = (torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)).amax(dim=1)
    ours = splade.encode_documents(texts).toarray()
    np.testing.assert_allclose(ours, ref.numpy(), atol=1e-5)
    assert ours.min() >= 0 and (ours > 0).sum(1).min() > 0


@needs_splade
def test_query_truncated_to_64_tokens(splade: SpladeEncoder) -> None:
    long_q = " ".join(f"word{i}" for i in range(200))
    ids = splade.tokenizer(long_q, add_special_tokens=False)["input_ids"][:62]   # 64 incl. [CLS]/[SEP]
    short_q = splade.tokenizer.decode(ids)
    a, b = splade.encode_queries([long_q]).toarray(), splade.encode_queries([short_q]).toarray()
    np.testing.assert_allclose(a, b, atol=1e-5)


@needs_splade
def test_real_splade_ranks_sensibly(splade: SpladeEncoder, corpus: Path, tmp_path: Path) -> None:
    settings = SpladeSettings(model="naver/splade-cocondenser-ensembledistil", encode_dtype="float32")
    build_splade_index(corpus, tmp_path / "real", splade, settings)
    r = SpladeRetriever(tmp_path / "real", encoder=splade, mmap=False)
    assert r.search(["who wrote hamlet"], k=1)[0][0][0] in {"d0", "d5"}
    assert r.search(["mountain in tanzania"], k=1)[0][0][0] == "d6"
    assert splade.has_content(["", "  ", "the", "?"]) == [False, False, True, True]
    assert r.search([""], k=5) == [[]]
