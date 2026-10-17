"""BM25 index and retriever (Step 3), built on bm25s with PyStemmer and English stopwords (no Java).

    from ire.retrieval.bm25 import BM25Retriever, build_index
    build_index(corpus_path, out_dir, extra_shards=[...])          # one-off
    bm25 = BM25Retriever(out_dir)                                   # cheap: memory-mapped arrays
    bm25.search(["who wrote hamlet"], k=150)  -> [[("doc12", 17.3), ...]]
    bm25.score_docs("who wrote hamlet", ["doc12", "poison_q1_fact_1"])  -> array of scores

Design notes
- Indexed text is `title + " " + text`. Tokenisation (lowercase, bm25s token pattern, bm25s English stopwords,
  Snowball English stemmer) and BM25 parameters come from `bm25:` in configs/experiment.yaml and are stored in the
  index manifest; loading checks them, so local and Ada indexes are built the same way.
- Results use the original string doc_ids. Only documents sharing at least one query term are returned (score > 0),
  so a row can be shorter than k when few documents match; a query with no indexable terms (empty, only stopwords,
  only unknown words) returns []. Ties are ordered by position in the index, so output is fully deterministic.
- `score_docs` scores any documents under the *current* index statistics (IDF, average length); comparing it with
  an index built without poison measures the "mask after scoring" approximation from the plan (section 2.1).
- Poison status is never stored: it is `is_poison_id(doc_id)` (the `poison_` prefix).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import Stemmer
from bm25s.tokenization import Tokenizer

from ire.retrieval.common import (  # noqa: F401  (doc_text, find_poison_shards, iter_documents re-exported)
    MANIFEST,
    dir_size,
    doc_text,
    find_poison_shards,
    index_dir_for,
    inputs_fingerprint,
    iter_documents,
    load_documents,
    read_manifest,
    swap_in,
)
from ire.schemas import is_poison_id

DOC_IDS = "doc_ids.json"
EMPTY_TOKEN_ID = 0  # bm25s gives every empty document/query the "" token; it must never match anything


@dataclass(frozen=True)
class BM25Settings:
    """Tokenisation and scoring settings (from `bm25:` in experiment.yaml)."""

    method: str = "lucene"
    k1: float = 0.9
    b: float = 0.4
    lower: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"
    stopwords: str = "en"
    stemmer: str = "english"

    @classmethod
    def from_config(cls, cfg: Any) -> BM25Settings:
        bm = cfg["bm25"]
        return cls(method=bm["method"], k1=float(bm["k1"]), b=float(bm["b"]), lower=bool(bm["lower"]),
                   token_pattern=bm["token_pattern"], stopwords=bm["stopwords"], stemmer=bm["stemmer"])

    def tokenizer(self) -> Tokenizer:
        return Tokenizer(lower=self.lower, splitter=self.token_pattern, stopwords=self.stopwords,
                         stemmer=Stemmer.Stemmer(self.stemmer))


def fingerprint(corpus_path: Path, extra_shards: Sequence[Path], settings: BM25Settings) -> dict[str, Any]:
    """What an index was built from: content hashes of the corpus and shards, and the settings."""
    return {**inputs_fingerprint(corpus_path, extra_shards), "settings": asdict(settings),
            "bm25s_version": bm25s.__version__}


def index_is_current(index_dir: Path, corpus_path: Path, extra_shards: Sequence[Path],
                     settings: BM25Settings) -> bool:
    """True if index_dir holds a finished index built from exactly these inputs and settings."""
    manifest = read_manifest(index_dir)
    return manifest is not None and manifest.get("fingerprint") == fingerprint(corpus_path, extra_shards, settings)


def build_index(corpus_path: Path, out_dir: Path, extra_shards: Sequence[Path] | None = None,
                settings: BM25Settings | None = None, extra_docs: Iterable[tuple[str, str]] = (),
                max_docs: int | None = None) -> dict[str, Any]:
    """Build a BM25 index of corpus_path (+ poison shards) into out_dir and return its manifest.

    extra_docs / max_docs exist for throw-away test indexes (unique-token test); real indexes leave them unset.
    The index is written to a sibling temp folder and swapped in at the end, so a crash never leaves a half index.
    """
    settings = settings or BM25Settings()
    shards = list(extra_shards or [])
    t0 = time.time()
    doc_ids, texts = load_documents(corpus_path, shards, max_docs, extra_docs)
    t_read = time.time() - t0

    tokenizer = settings.tokenizer()
    token_ids = tokenizer.tokenize(texts, return_as="ids", show_progress=False)
    del texts
    model = bm25s.BM25(k1=settings.k1, b=settings.b, method=settings.method)
    model.index(token_ids, show_progress=False)
    t_index = time.time() - t0 - t_read

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover of an interrupted build (our own temp folder)
    tmp.mkdir(parents=True)
    model.save(tmp, show_progress=False)
    tokenizer.save_vocab(tmp)
    tokenizer.save_stopwords(tmp)
    (tmp / DOC_IDS).write_text(json.dumps(doc_ids), encoding="utf-8")
    n_poison = sum(1 for d in doc_ids if is_poison_id(d))
    manifest = {
        "fingerprint": fingerprint(corpus_path, shards, settings) if max_docs is None and not extra_docs else None,
        "settings": asdict(settings),
        "n_docs": len(doc_ids), "n_poison": n_poison, "n_clean": len(doc_ids) - n_poison,
        "n_vocab": len(tokenizer.get_vocab_dict()),
        "timings_s": {"read": round(t_read, 2), "tokenize_and_index": round(t_index, 2),
                      "total": round(time.time() - t0, 2)},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["index_bytes"] = dir_size(tmp)

    swap_in(tmp, out_dir)
    return manifest


class BM25Retriever:
    """Search a saved BM25 index. Loading memory-maps the score arrays, so it is cheap.

    Use mmap=False for throw-away indexes that are deleted in the same process (Windows locks mapped files).
    """

    def __init__(self, index_dir: Path, settings: BM25Settings | None = None, mmap: bool = True) -> None:
        self.index_dir = Path(index_dir)
        self.manifest = read_manifest(self.index_dir)
        if self.manifest is None:
            raise FileNotFoundError(f"no BM25 index at {self.index_dir} (run scripts/build_index.py --retriever bm25)")
        built_with = BM25Settings(**self.manifest["settings"])
        if settings is not None and settings != built_with:
            raise ValueError(f"index at {self.index_dir} was built with {asdict(built_with)}, config says "
                             f"{asdict(settings)}; rebuild with --force")
        self.settings = built_with  # queries must be tokenised exactly like the documents were
        self.model = bm25s.BM25.load(self.index_dir, mmap=mmap, show_progress=False)
        self.tokenizer = self.settings.tokenizer()
        self.tokenizer.load_vocab(str(self.index_dir))
        self.tokenizer.load_stopwords(str(self.index_dir))
        self.doc_ids: list[str] = json.loads((self.index_dir / DOC_IDS).read_text(encoding="utf-8"))
        self._row_of: dict[str, int] | None = None  # doc_id -> row, built on first score_docs call

    @classmethod
    def from_config(cls, cfg: Any) -> BM25Retriever:
        return cls(default_index_dir(cfg), BM25Settings.from_config(cfg))

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @staticmethod
    def is_poison(doc_id: str) -> bool:
        return is_poison_id(doc_id)

    def _query_token_ids(self, queries: Sequence[str]) -> list[list[int]]:
        ids = self.tokenizer.tokenize(list(queries), update_vocab=False, return_as="ids", show_progress=False)
        return [[t for t in q if t != EMPTY_TOKEN_ID] for q in ids]  # unknown words and stopwords vanish

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Top-k (doc_id, score) per query, score descending, ties by index position; only score > 0."""
        token_ids = self._query_token_ids(queries)
        out: list[list[tuple[str, float]]] = [[] for _ in queries]
        todo = [i for i, q in enumerate(token_ids) if q]
        if not todo or k <= 0:
            return out
        rows, scores = self.model.retrieve([token_ids[i] for i in todo], k=min(k, self.n_docs),
                                           show_progress=False, n_threads=0)
        for i, row, score in zip(todo, rows, scores):
            order = np.lexsort((row, -score))  # score descending, then index position ascending
            out[i] = [(self.doc_ids[int(row[j])], float(score[j])) for j in order if score[j] > 0]
        return out

    def score_docs(self, query: str, doc_ids: Sequence[str]) -> np.ndarray:
        """BM25 score of each doc_id for `query` under the current index statistics (0.0 if no term matches).

        Raises KeyError for a doc_id that is not in the index.
        """
        if self._row_of is None:
            self._row_of = {d: i for i, d in enumerate(self.doc_ids)}
        rows = [self._row_of[d] for d in doc_ids]
        (tokens,) = self._query_token_ids([query])
        if not tokens:
            return np.zeros(len(rows), dtype=np.float32)
        return np.asarray(self.model.get_scores(tokens))[rows]


def default_index_dir(cfg: Any) -> Path:
    """`<indexes_dir>/<env>/bm25` (local and Ada indexes never mix)."""
    return index_dir_for(cfg, "bm25")


def unique_token_test(corpus_path: Path, work_dir: Path, settings: BM25Settings | None = None,
                      max_docs: int | None = None, token: str = "zorblaxquintavel") -> dict[str, Any]:
    """Build a throw-away index (corpus, or its first max_docs passages, + one passage containing a made-up word),
    search for the word, and report its rank. The real index and corpus are never touched; work_dir is deleted."""
    doc_id = "unique_token_test_doc"
    passage = f"This test passage mentions the invented word {token} exactly once."
    try:
        manifest = build_index(corpus_path, work_dir / "bm25_unique_token", settings=settings,
                               extra_docs=[(doc_id, passage)], max_docs=max_docs)
        # mmap=False: Windows cannot delete memory-mapped files, and this index is deleted right after
        hits = BM25Retriever(work_dir / "bm25_unique_token", settings, mmap=False).search([token], k=10)[0]
        rank = next((r for r, (d, _) in enumerate(hits, start=1) if d == doc_id), None)
        return {"token": token, "rank": rank, "passed": rank == 1, "n_hits": len(hits),
                "n_docs_in_test_index": manifest["n_docs"]}
    finally:
        shutil.rmtree(work_dir / "bm25_unique_token", ignore_errors=True)
