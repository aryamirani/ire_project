"""Generation cache (Step 9): SQLite (WAL) store of LLM answers.

key       = sha1(qid + "|" + "|".join(top5_ids))          -- exactly as in the plan (order of the 5 ids matters)
namespace = sha1 of everything that can change an answer (model id + revision, rendered prompt template, passage
            format, truncation, generation settings; see ire.generation.llm.Generator.namespace_settings).
An answer is only reused inside its own namespace, so a changed model or prompt never returns old answers.
Rows are written one transaction per batch: a killed run loses at most the batch in flight.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def cache_key(qid: str, top5_ids: Sequence[str]) -> str:
    """sha1(qid + "|" + "|".join(top5_ids)) as in IMPLEMENTATION_PLAN.md step 9."""
    return hashlib.sha1((qid + "|" + "|".join(top5_ids)).encode("utf-8")).hexdigest()


def namespace_id(settings: dict[str, Any]) -> str:
    """sha1 of the canonical JSON of the namespace settings."""
    return hashlib.sha1(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class GenerationCache:
    """answers(namespace, key) -> answer, raw output, qid, top5, prompt sha1, timestamp."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.execute("PRAGMA synchronous = NORMAL")
        with self._con:
            self._con.execute("""CREATE TABLE IF NOT EXISTS namespaces (
                namespace TEXT PRIMARY KEY, settings TEXT NOT NULL, created_utc TEXT NOT NULL)""")
            self._con.execute("""CREATE TABLE IF NOT EXISTS answers (
                namespace TEXT NOT NULL, key TEXT NOT NULL, qid TEXT NOT NULL, top5 TEXT NOT NULL,
                answer TEXT NOT NULL, raw TEXT NOT NULL, prompt_sha1 TEXT NOT NULL, created_utc TEXT NOT NULL,
                PRIMARY KEY (namespace, key))""")

    def register_namespace(self, namespace: str, settings: dict[str, Any]) -> None:
        with self._con:
            self._con.execute("INSERT OR IGNORE INTO namespaces VALUES (?, ?, ?)",
                              (namespace, json.dumps(settings, sort_keys=True), _now()))

    def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        uniq = list(dict.fromkeys(keys))
        for start in range(0, len(uniq), 500):
            part = uniq[start:start + 500]
            q = (f"SELECT key, answer, raw, qid, top5, prompt_sha1, created_utc FROM answers "
                 f"WHERE namespace = ? AND key IN ({','.join('?' * len(part))})")
            for key, answer, raw, qid, top5, psha, created in self._con.execute(q, [namespace, *part]):
                out[key] = {"answer": answer, "raw": raw, "qid": qid, "top5": json.loads(top5),
                            "prompt_sha1": psha, "created_utc": created}
        return out

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        return self.get_many(namespace, [key]).get(key)

    def put_many(self, namespace: str, rows: Sequence[dict[str, Any]]) -> int:
        """Insert rows (keys: key, qid, top5, answer, raw, prompt_sha1) in ONE transaction; existing keys are kept."""
        now = _now()
        with self._con:
            cur = self._con.executemany(
                "INSERT OR IGNORE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(namespace, r["key"], r["qid"], json.dumps(list(r["top5"])), r["answer"], r["raw"],
                  r["prompt_sha1"], now) for r in rows])
        return cur.rowcount

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return self._con.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        return self._con.execute("SELECT COUNT(*) FROM answers WHERE namespace = ?", (namespace,)).fetchone()[0]

    def close(self) -> None:
        self._con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
