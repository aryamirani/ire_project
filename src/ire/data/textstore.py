"""Document text store (Step 9): doc_id -> (title, text) for the corpus plus every poison shard, in SQLite.

    from ire.data.textstore import TextStore
    store = TextStore.open_for(cfg)            # builds / refreshes <data_dir>/data/text_store.sqlite if needed
    store.record("doc12")                      # ("Hamlet", "Hamlet is a tragedy ...")
    store["doc12"]                             # "Hamlet Hamlet is a tragedy ..." (title + " " + text, as indexed)
    store.records(["doc12", "poison_q1_fact_1"])

On Ada the corpus has 2.68M passages; SQLite keeps them on disk (one indexed lookup per doc_id) instead of a Python
dict. The store records the fingerprint of its inputs (corpus sha256 + poison shard hashes) and is rebuilt when they
change. It is a read-only Mapping[str, str], so it can be FilterContext.texts. Unknown doc_ids raise KeyError.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ire.retrieval.common import doc_text, find_poison_shards, inputs_fingerprint
from ire.schemas import CorpusRecord, PoisonRecord, iter_jsonl

SCHEMA_VERSION = 1


class TextStore(Mapping[str, str]):
    """Read-only doc_id -> text mapping backed by SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no text store at {self.path} (TextStore.build or TextStore.open_for)")
        self._con = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        self.fingerprint = json.loads(self._meta("fingerprint"))
        self._len = int(self._meta("n_docs"))

    def _meta(self, key: str) -> str:
        row = self._con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise RuntimeError(f"{self.path}: missing meta {key!r} (rebuild the store)")
        return row[0]

    # ---- building

    @staticmethod
    def build(path: Path, corpus_path: Path, shards: Sequence[Path] = (), batch: int = 50_000) -> TextStore:
        """Write a new store (temp file, then rename): every corpus passage, then every poison passage."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".building")
        tmp.unlink(missing_ok=True)
        con = sqlite3.connect(tmp)
        con.execute("PRAGMA journal_mode = OFF")
        con.execute("PRAGMA synchronous = OFF")
        con.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL) WITHOUT ROWID")
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        def rows() -> Iterator[tuple[str, str, str]]:
            for rec in iter_jsonl(corpus_path, CorpusRecord):
                yield rec.doc_id, rec.title, rec.text
            for shard in shards:
                for prec in iter_jsonl(shard, PoisonRecord):
                    yield prec.doc_id, prec.title, prec.text

        n = 0
        chunk: list[tuple[str, str, str]] = []
        for r in rows():
            chunk.append(r)
            if len(chunk) == batch:
                con.executemany("INSERT INTO docs VALUES (?, ?, ?)", chunk)  # duplicate doc_id -> IntegrityError
                n += len(chunk)
                chunk = []
        if chunk:
            con.executemany("INSERT INTO docs VALUES (?, ?, ?)", chunk)
            n += len(chunk)
        fp = {**inputs_fingerprint(corpus_path, shards), "schema_version": SCHEMA_VERSION}
        con.executemany("INSERT INTO meta VALUES (?, ?)", [("fingerprint", json.dumps(fp)), ("n_docs", str(n))])
        con.commit()
        con.close()
        os.replace(tmp, path)
        return TextStore(path)

    @classmethod
    def open_for(cls, cfg: Any, shards: Sequence[Path] | None = None, path: Path | None = None) -> TextStore:
        """The store for the active corpus + poison shards (default: every poison_*.jsonl in poison_dir),
        rebuilt if missing or built from other inputs."""
        path = Path(path or cfg.path("text_store"))
        shards = list(find_poison_shards(cfg.path("poison_dir")) if shards is None else shards)
        corpus = cfg.path("corpus")
        want = {**inputs_fingerprint(corpus, shards), "schema_version": SCHEMA_VERSION}
        if path.is_file():
            store = cls(path)
            if store.fingerprint == want:
                return store
            store.close()
        return cls.build(path, corpus, shards)

    # ---- reading

    def record(self, doc_id: str) -> tuple[str, str]:
        row = self._con.execute("SELECT title, text FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return row[0], row[1]

    def records(self, doc_ids: Sequence[str]) -> list[tuple[str, str]]:
        """(title, text) for each doc_id, in the given order. KeyError names the first unknown id."""
        found: dict[str, tuple[str, str]] = {}
        uniq = list(dict.fromkeys(doc_ids))
        for start in range(0, len(uniq), 500):
            part = uniq[start:start + 500]
            q = f"SELECT doc_id, title, text FROM docs WHERE doc_id IN ({','.join('?' * len(part))})"
            for d, title, text in self._con.execute(q, part):
                found[d] = (title, text)
        missing = [d for d in uniq if d not in found]
        if missing:
            raise KeyError(missing[0])
        return [found[d] for d in doc_ids]

    def __getitem__(self, doc_id: str) -> str:
        return doc_text(*self.record(doc_id))

    def __contains__(self, doc_id: object) -> bool:
        return isinstance(doc_id, str) and self._con.execute(
            "SELECT 1 FROM docs WHERE doc_id = ?", (doc_id,)).fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        for (d,) in self._con.execute("SELECT doc_id FROM docs"):
            yield d

    def __len__(self) -> int:
        return self._len

    def close(self) -> None:
        self._con.close()
