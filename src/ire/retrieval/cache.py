"""Top-150 retrieval cache (Step 7): build, resume, validate and load `top150_<retriever>.jsonl`.

    from ire.retrieval.cache import load_cache
    cache = load_cache(["bm25", "dense"])          # {"bm25": {qid: [(doc_id, score), ...]}, "dense": {...}}

Files live in `<cache_dir>/<env>/` (cache_dir from paths.yaml). Each `top150_<r>.jsonl` holds one
RetrievalCacheRecord per question (section 2 contract), in question-file order; rows are sorted by score
descending with ties by doc_id. Next to it, `top150_<r>.jsonl.manifest.json` (the sidecar) records the cache KEY
(retriever, questions sha256, depth, cache format version and the index fingerprint copied from the index manifest;
for rrf the keys of both component caches plus rrf k/depth) and build information.

The loader recomputes the key from the CURRENT index manifests and questions file and refuses a cache whose key
differs (e.g. after poison ingestion or a new question set): StaleCacheError.

top150_rrf.jsonl is fused from the cached bm25 and dense rows and is an UNMASKED, NO-POISON REFERENCE only
(sidecar "unmasked_reference": true). load_cache refuses it unless allow_unmasked_reference=True: experiments
with poison must mask the bm25 and dense lists and call fuse() at pool time (see HANDOFF "Interface changes").
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ire.retrieval.common import index_dir_for, read_manifest
from ire.retrieval.rrf import fuse
from ire.schemas import QuestionRecord, RetrievalCacheRecord, iter_jsonl, read_jsonl

CACHE_VERSION = 1           # bump if the search/ordering code changes in a way that changes cached rows
BASE_RETRIEVERS = ("bm25", "dense", "splade")
RRF = "rrf"
ALL_RETRIEVERS = (*BASE_RETRIEVERS, RRF)
UNMASKED_MSG = ("top150_rrf.jsonl is an unmasked, no-poison reference only. For any setting with poison, mask the "
                "bm25 and dense lists and call fuse() instead (ire.retrieval.rrf.fuse). Pass "
                "allow_unmasked_reference=True only for no-poison reference checks.")

Row = list[tuple[str, float]]


class CacheError(RuntimeError):
    """A cache is missing or cannot be built."""


class StaleCacheError(CacheError):
    """A cache was built from a different index or question set than the current one."""


class UnmaskedReferenceError(CacheError):
    """The fused rrf cache was requested without allow_unmasked_reference=True."""


# ---------------------------------------------------------------- paths and keys


def cache_dir_for(cfg: Any) -> Path:
    """`<cache_dir>/<env>` (local and Ada caches never mix)."""
    return cfg.path("cache_dir") / cfg.env


def cache_path(cache_dir: Path, retriever: str) -> Path:
    return Path(cache_dir) / f"top150_{retriever}.jsonl"


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def index_fingerprint(index_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(index_dir)
    if manifest is None or not manifest.get("fingerprint"):
        raise CacheError(f"no finished index at {Path(index_dir).as_posix()} (run scripts/build_index.py first)")
    return manifest["fingerprint"]


def base_key(retriever: str, questions_path: Path, depth: int, index_fp: dict[str, Any]) -> dict[str, Any]:
    """What a bm25/dense/splade cache depends on."""
    return {"retriever": retriever, "cache_version": CACHE_VERSION, "questions_sha256": sha256_file(questions_path),
            "depth": int(depth), "index_fingerprint": index_fp}


def rrf_key(questions_path: Path, component_keys: dict[str, dict[str, Any]], k: int, depth: int) -> dict[str, Any]:
    """What the fused rrf cache depends on: both component caches (their keys) and the fusion parameters."""
    return {"retriever": RRF, "cache_version": CACHE_VERSION, "questions_sha256": sha256_file(questions_path),
            "components": component_keys, "rrf": {"k": int(k), "depth": int(depth)}}


def current_key(cfg: Any, retriever: str, questions_path: Path) -> dict[str, Any]:
    """The key a cache for `retriever` must have right now (from the current index manifests and questions)."""
    if retriever == RRF:
        comps = {name: current_key(cfg, name, questions_path) for name in cfg["rrf.components"]}
        return rrf_key(questions_path, comps, int(cfg["rrf.k"]), int(cfg["rrf.depth"]))
    if retriever not in BASE_RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}; expected one of {ALL_RETRIEVERS}")
    return base_key(retriever, questions_path, int(cfg["top_k_cache"]), index_fingerprint(index_dir_for(cfg, retriever)))


def key_mismatch(expected: dict[str, Any], found: dict[str, Any]) -> list[str]:
    """Top-level key fields that differ (for a readable stale-cache message)."""
    return sorted(k for k in set(expected) | set(found) if expected.get(k) != found.get(k))


def read_sidecar(path: Path) -> dict[str, Any] | None:
    side = sidecar_path(path)
    return json.loads(side.read_text(encoding="utf-8")) if side.is_file() else None


# ---------------------------------------------------------------- writing (resumable, atomic)


def _line(qid: str, retriever: str, row: Row) -> str:
    rec = RetrievalCacheRecord(qid=qid, retriever=retriever, top150=list(row))  # validates the contract
    return json.dumps(rec.to_dict(), ensure_ascii=False) + "\n"


def _check_row(retriever: str, qid: str, row: Row, depth: int, exact_len: int | None) -> None:
    keys = [(-s, d) for d, s in row]
    if keys != sorted(keys):
        raise CacheError(f"{retriever} row for {qid} is not sorted by score desc / doc_id asc")
    if len(row) > depth or (exact_len is not None and len(row) != exact_len):
        raise CacheError(f"{retriever} row for {qid} has {len(row)} entries "
                         f"(expected {'exactly ' + str(exact_len) if exact_len is not None else 'at most ' + str(depth)})")


def write_cache(out_path: Path, key: dict[str, Any], questions: Sequence[QuestionRecord],
                compute_rows: Callable[[Sequence[QuestionRecord]], list[Row]], depth: int,
                exact_len: int | None = None, batch_size: int = 32, force: bool = False,
                extra_sidecar: dict[str, Any] | None = None, log: Any = None) -> dict[str, Any]:
    """Write one cache file. Skips if the finished file already has this key (unless force).

    Rows are appended to `<file>.partial` one question at a time; a restart with the same key keeps the complete
    lines already written and continues after them. At the end the partial file is renamed into place and the
    sidecar written. Returns the sidecar dict (with "skipped": True if nothing had to be done).
    """
    out_path = Path(out_path)
    retriever = key["retriever"]
    side = sidecar_path(out_path)
    existing = read_sidecar(out_path)
    if not force and out_path.is_file() and existing and existing.get("key") == key:
        if log:
            log.info("%s cache is current: nothing to do (--force rebuilds)", retriever)
        return {**existing, "skipped": True}

    t0 = time.time()
    partial = out_path.with_name(out_path.name + ".partial")
    partial_key = partial.with_name(partial.name + ".key.json")
    qids = [q.qid for q in questions]
    done: list[str] = []
    if not force and partial.is_file() and partial_key.is_file() and \
            json.loads(partial_key.read_text(encoding="utf-8")) == key:
        data = partial.read_bytes()
        cut = data.rfind(b"\n") + 1
        if cut < len(data):  # drop a half-written last line
            with partial.open("r+b") as fh:
                fh.truncate(cut)
        done = [json.loads(line)["qid"] for line in data[:cut].decode("utf-8").splitlines() if line.strip()]
        if done != qids[:len(done)]:
            done = []  # not a prefix of this question list: start over
    if not done:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"")
        partial_key.write_text(json.dumps(key), encoding="utf-8")
    resumed_from = len(done)
    if log and resumed_from:
        log.info("%s: resuming after %d of %d questions", retriever, resumed_from, len(qids))

    todo = list(questions[resumed_from:])
    with partial.open("a", encoding="utf-8", newline="\n") as fh:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            rows = compute_rows(batch)
            for q, row in zip(batch, rows):
                _check_row(retriever, q.qid, row, depth, exact_len)
                fh.write(_line(q.qid, retriever, row))
                fh.flush()
            if log:
                log.info("%s: %d / %d questions", retriever, resumed_from + start + len(batch), len(qids))
    build_s = time.time() - t0

    lengths = [len(rec.top150) for rec in iter_jsonl(partial, RetrievalCacheRecord)]
    if len(lengths) != len(qids):
        raise CacheError(f"{retriever}: wrote {len(lengths)} rows for {len(qids)} questions")
    sidecar = {
        "key": key, "retriever": retriever, "unmasked_reference": retriever == RRF,
        "n_questions": len(qids), "depth": depth,
        "row_length": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths),
                       "short_rows": sum(1 for n in lengths if n < depth)},
        "build_time_s": round(build_s, 2), "resumed_from_question": resumed_from,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra_sidecar or {}),
    }
    side.unlink(missing_ok=True)            # never leave an old sidecar next to a new file
    os.replace(partial, out_path)
    tmp_side = side.with_name(side.name + ".tmp")
    tmp_side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    os.replace(tmp_side, side)
    partial_key.unlink(missing_ok=True)
    return {**sidecar, "skipped": False}


# ---------------------------------------------------------------- loading


def load_cache_file(path: Path, expected_key: dict[str, Any], qids: Sequence[str] | None = None,
                    allow_unmasked_reference: bool = False) -> dict[str, Row]:
    """Read one cache file after checking its sidecar against expected_key (and the question ids, if given)."""
    path = Path(path)
    sidecar = read_sidecar(path)
    if not path.is_file() or sidecar is None:
        raise CacheError(f"no cache at {path.as_posix()} (run scripts/build_cache.py)")
    if sidecar.get("unmasked_reference") and not allow_unmasked_reference:
        raise UnmaskedReferenceError(UNMASKED_MSG)
    if sidecar.get("key") != expected_key:
        diff = key_mismatch(expected_key, sidecar.get("key") or {})
        raise StaleCacheError(f"{path.name} is stale: {', '.join(diff)} {'differs' if len(diff) == 1 else 'differ'} "
                              f"from the current index / questions (e.g. after poison ingestion or a new question "
                              f"set). Rebuild it: scripts/build_cache.py --retriever {expected_key.get('retriever')}")
    rows = {rec.qid: [(d, s) for d, s in rec.top150] for rec in iter_jsonl(path, RetrievalCacheRecord)}
    if qids is not None and list(rows) != list(qids):
        raise StaleCacheError(f"{path.name} rows do not match the question ids of the questions file")
    return rows


def load_cache(retrievers: Sequence[str], allow_unmasked_reference: bool = False, cfg: Any = None,
               questions_path: Path | None = None) -> dict[str, dict[str, Row]]:
    """{retriever: {qid: [(doc_id, score), ...]}} for the active environment, validated against the CURRENT index
    manifests and questions file (default: questions.jsonl). Raises StaleCacheError / UnmaskedReferenceError."""
    if cfg is None:
        from ire.config import load_config

        cfg = load_config()
    questions_path = Path(questions_path or cfg.path("questions"))
    if RRF in retrievers and not allow_unmasked_reference:
        raise UnmaskedReferenceError(UNMASKED_MSG)
    qids = [q.qid for q in read_jsonl(questions_path, QuestionRecord)]
    out = {}
    for r in retrievers:
        out[r] = load_cache_file(cache_path(cache_dir_for(cfg), r), current_key(cfg, r, questions_path), qids,
                                 allow_unmasked_reference)
    return out


def rrf_component_rows(cfg: Any, questions_path: Path, qids: Sequence[str]) -> dict[str, dict[str, Row]]:
    """The cached component rows rrf is fused from. Raises CacheError if a component cache is missing or stale:
    rrf is never built from fresh retrieval, so it always matches its component files."""
    try:
        return {c: load_cache_file(cache_path(cache_dir_for(cfg), c), current_key(cfg, c, questions_path), qids)
                for c in cfg["rrf.components"]}
    except CacheError as exc:
        raise CacheError(f"refusing to build rrf: the {'/'.join(cfg['rrf.components'])} caches must exist and be "
                         f"current first ({exc})") from exc


def fuse_rows(bm25_rows: dict[str, Row], dense_rows: dict[str, Row], questions: Sequence[QuestionRecord],
              k: int, depth: int) -> list[Row]:
    """RRF rows from the cached component rows (no re-retrieval)."""
    return [fuse(bm25_rows[q.qid], dense_rows[q.qid], k=k, depth=depth) for q in questions]
