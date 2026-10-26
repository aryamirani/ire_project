"""Step 4: dense index, retriever and vector API.

Most tests use a deterministic fake encoder (hashed bag of words), so they need no model and no network. The E5
tests use the real intfloat/e5-small-v2 on CPU and are skipped if that model is not in the local HF cache.
"""

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ire.config import load_config
from ire.retrieval.common import doc_text
from ire.retrieval.dense import (
    PROGRESS,
    DenseRetriever,
    DenseSettings,
    E5Encoder,
    build_dense_index,
    index_is_current,
)
from ire.schemas import CorpusRecord, FilterContext, PoisonRecord, write_jsonl

DOCS = [
    ("d0", "Hamlet", "Hamlet is a tragedy written by William Shakespeare around 1600."),
    ("d1", "Macbeth", "Macbeth is a tragedy by Shakespeare about a Scottish general."),
    ("d2", "Zebra", "Zebras are African equines with black and white striped coats."),
    ("d3", "Volcano", "A volcano is a rupture in the crust of a planet."),
    ("d4", "Python (programming language)", "Python is a programming language created by Guido van Rossum."),
    ("d5", "Shakespeare", "William Shakespeare was an English playwright, poet and actor."),
    ("d6", "Kilimanjaro", "Mount Kilimanjaro is a dormant volcano in Tanzania."),
]
SETTINGS = DenseSettings(model="fake-encoder")


class FakeEncoder:
    """Hashed bag-of-words vectors: deterministic, L2-normalised, shares words -> high inner product."""

    name, revision, dim = "fake-encoder", "fake-rev-1", 64

    def __init__(self, fail_after_calls: int | None = None) -> None:
        self.calls = 0
        self.fail_after_calls = fail_after_calls

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        v[0] += 1e-3  # never all-zero
        return v / np.linalg.norm(v)

    def encode_passages(self, texts):
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("simulated crash")
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [CorpusRecord(doc_id=d, title=t, text=x, is_poison=False) for d, t, x in DOCS])
    return path


@pytest.fixture
def index(corpus: Path, tmp_path: Path) -> DenseRetriever:
    build_dense_index(corpus, tmp_path / "dense", FakeEncoder(), SETTINGS)
    return DenseRetriever(tmp_path / "dense", encoder=FakeEncoder(), mmap=False)


# ---------------------------------------------------------------- prefixes (config + encoder guard)

@pytest.mark.parametrize("env", ["local", "ada"])
def test_e5_prefixes_are_in_config(env: str) -> None:
    cfg = load_config(env)
    assert "e5" in cfg["models.dense"]
    assert cfg["dense.query_prefix"] == "query: " and cfg["dense.passage_prefix"] == "passage: "
    settings = DenseSettings.from_config(cfg)
    assert (settings.query_prefix, settings.passage_prefix) == ("query: ", "passage: ")


@pytest.mark.parametrize("field", ["query_prefix", "passage_prefix"])
def test_e5_encoder_refuses_missing_prefix(field: str) -> None:
    settings = replace(DenseSettings.from_config(load_config("local")), **{field: ""})
    with pytest.raises(ValueError, match="needs the E5 prefixes"):
        E5Encoder(settings, device="cpu")  # raises before any model is loaded


# ---------------------------------------------------------------- index + retriever (fake encoder)

def test_search_and_storage(index: DenseRetriever, tmp_path: Path) -> None:
    (hits,) = index.search(["zebras striped coats"], k=3)
    assert hits[0][0] == "d2" and all(isinstance(d, str) and isinstance(s, float) for d, s in hits)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert index.search(["volcano"], k=100)[0].__len__() == len(DOCS)  # k clamped to the index size
    assert index.search([], k=5) == [] and index.search(["x"], k=0) == [[]]
    files = sorted(p.name for p in (tmp_path / "dense").iterdir())
    assert files == ["doc_ids.npy", "doc_vectors.npy", "faiss.index", "manifest.json"]
    stored = np.load(tmp_path / "dense" / "doc_vectors.npy", mmap_mode="r")
    assert stored.dtype == np.float16 and stored.shape == (len(DOCS), FakeEncoder.dim)


def test_search_independent_of_query_batch(index: DenseRetriever) -> None:
    """A query's results must not depend on how many queries are searched together (FAISS uses BLAS for >= 20)."""
    rng = np.random.default_rng(0)
    qv = rng.normal(size=(25, FakeEncoder.dim)).astype(np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    batched = index.search_vectors(qv, k=5)
    single = [index.search_vectors(qv[i:i + 1], k=5)[0] for i in range(len(qv))]
    assert batched == single


def test_get_doc_vector_reads_stored_rows(index: DenseRetriever) -> None:
    ids = ["d3", "d0", "d6"]
    vecs = index.get_doc_vector(ids)
    assert vecs.dtype == np.float32 and vecs.shape == (3, FakeEncoder.dim)
    fresh = FakeEncoder().encode_passages([doc_text(*next((t, x) for d, t, x in DOCS if d == i)) for i in ids])
    assert np.all(np.sum(vecs * fresh, axis=1) > 0.999)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-3)
    assert index.get_doc_vector([]).shape == (0, FakeEncoder.dim)
    with pytest.raises(KeyError):
        index.get_doc_vector(["nope"])


def test_get_query_vector_and_score_docs(index: DenseRetriever) -> None:
    q = index.get_query_vector(["william shakespeare"])
    assert q.shape == (1, FakeEncoder.dim) and abs(np.linalg.norm(q) - 1) < 1e-5
    (hits,) = index.search(["william shakespeare"], k=len(DOCS))
    np.testing.assert_allclose(index.score_docs("william shakespeare", [d for d, _ in hits]),
                               [s for _, s in hits], rtol=1e-5, atol=1e-6)


def test_vector_api_fits_filter_context(index: DenseRetriever) -> None:
    ctx = FilterContext(texts={}, vectors=index, config={})
    assert ctx.vectors.get_doc_vector(["d1"]).shape == (1, FakeEncoder.dim)


def test_poison_shards_appended(corpus: Path, tmp_path: Path) -> None:
    shard = tmp_path / "poison" / "poison_fact.jsonl"
    write_jsonl(shard, [PoisonRecord(doc_id="poison_q1_fact_1", qid="q1", attack="fact", title="",
                                     text="Zebras striped coats zebra zebra", is_poison=True, target_answer="x")])
    manifest = build_dense_index(corpus, tmp_path / "p", FakeEncoder(), SETTINGS, extra_shards=[shard])
    assert (manifest["n_docs"], manifest["n_poison"]) == (len(DOCS) + 1, 1)
    r = DenseRetriever(tmp_path / "p", encoder=FakeEncoder(), mmap=False)
    ids = [d for d, _ in r.search(["zebras striped coats"], k=2)[0]]
    assert "poison_q1_fact_1" in ids and r.is_poison("poison_q1_fact_1") and not r.is_poison("d2")


def test_resume_after_crash(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_dense_index(corpus, out, FakeEncoder(fail_after_calls=2), SETTINGS, checkpoint_every=3)
    progress = json.loads((tmp_path / "resume.building" / PROGRESS).read_text(encoding="utf-8"))
    assert progress["rows_done"] == 6 and not out.exists()
    enc = FakeEncoder()
    manifest = build_dense_index(corpus, out, enc, SETTINGS, checkpoint_every=3)
    assert manifest["resumed_from_row"] == 6 and enc.calls == 1  # only the last checkpoint was encoded again
    build_dense_index(corpus, tmp_path / "fresh", FakeEncoder(), SETTINGS, checkpoint_every=3)
    a = np.load(out / "doc_vectors.npy")
    b = np.load(tmp_path / "fresh" / "doc_vectors.npy")
    assert np.array_equal(a, b) and not (tmp_path / "resume.building").exists()


def test_index_is_current(corpus: Path, tmp_path: Path) -> None:
    build_dense_index(corpus, tmp_path / "d", FakeEncoder(), SETTINGS)
    assert index_is_current(tmp_path / "d", corpus, [], SETTINGS, FakeEncoder.revision)
    assert not index_is_current(tmp_path / "d", corpus, [], replace(SETTINGS, max_length=256), FakeEncoder.revision)
    assert not index_is_current(tmp_path / "d", corpus, [], SETTINGS, "other-revision")
    assert not index_is_current(tmp_path / "missing", corpus, [], SETTINGS, FakeEncoder.revision)


def test_hnsw_index_matches_flat_on_tiny_corpus(corpus: Path, tmp_path: Path) -> None:
    build_dense_index(corpus, tmp_path / "flat", FakeEncoder(), SETTINGS)
    build_dense_index(corpus, tmp_path / "hnsw", FakeEncoder(), replace(SETTINGS, index_type="hnsw", hnsw_m=8))
    flat = DenseRetriever(tmp_path / "flat", encoder=FakeEncoder(), mmap=False)
    hnsw = DenseRetriever(tmp_path / "hnsw", encoder=FakeEncoder(), mmap=False, ef_search=16)
    queries = ["who wrote hamlet", "volcano tanzania", "programming language"]
    assert [h[0][0] for h in flat.search(queries, k=3)] == [h[0][0] for h in hnsw.search(queries, k=3)]


# ---------------------------------------------------------------- real E5 (skipped if not cached)

def _e5_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("intfloat/e5-small-v2", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_e5 = pytest.mark.skipif(not _e5_cached(), reason="intfloat/e5-small-v2 not in the local HF cache")


@pytest.fixture(scope="module")
def e5() -> E5Encoder:
    return E5Encoder(DenseSettings(model="intfloat/e5-small-v2"), device="cpu", batch_size=4)


@needs_e5
def test_e5_prefix_changes_vectors(e5: E5Encoder) -> None:
    text = "Hamlet is a tragedy by William Shakespeare."
    with_p, without_p = e5.encode_passages([text])[0], e5._encode([text])[0]
    assert abs(np.linalg.norm(with_p) - 1) < 1e-4 and float(with_p @ without_p) < 0.99
    assert float(e5.encode_queries(["who wrote hamlet"])[0] @ e5._encode(["who wrote hamlet"])[0]) < 0.99


@needs_e5
def test_e5_index_reencode_matches_and_ranks_sensibly(e5: E5Encoder, corpus: Path, tmp_path: Path) -> None:
    settings = DenseSettings(model="intfloat/e5-small-v2")
    build_dense_index(corpus, tmp_path / "e5", e5, settings)
    r = DenseRetriever(tmp_path / "e5", encoder=e5, mmap=False)
    assert r.search(["who wrote hamlet"], k=1)[0][0][0] in {"d0", "d5"}
    assert r.search(["mountain in tanzania"], k=1)[0][0][0] == "d6"
    ids = [d for d, _, _ in DOCS]
    fresh = e5.encode_passages([doc_text(t, x) for _, t, x in DOCS[::-1]])[::-1]  # other batch order
    assert np.min(np.sum(r.get_doc_vector(ids) * fresh, axis=1)) > 0.999


def test_ties_ordered_by_doc_id_even_beyond_the_fetch_margin(tmp_path: Path) -> None:
    """40 identical passages (more than k + margin): FAISS must widen its candidate list until every tie is in."""
    path = tmp_path / "ties.jsonl"
    ids = [f"t{i:02d}" for i in range(39, -1, -1)]  # index order is the reverse of doc_id order
    write_jsonl(path, [CorpusRecord(doc_id=d, title="", text="identical text", is_poison=False) for d in ids]
                + [CorpusRecord(doc_id="other", title="", text="volcano tanzania", is_poison=False)])
    build_dense_index(path, tmp_path / "ties", FakeEncoder(), SETTINGS)
    r = DenseRetriever(tmp_path / "ties", encoder=FakeEncoder(), mmap=False)
    assert [d for d, _ in r.search(["identical text"], k=2)[0]] == ["t00", "t01"]
    assert [d for d, _ in r.search(["identical text"], k=41)[0]] == sorted(ids) + ["other"]
