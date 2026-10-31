"""Data contracts from docs/IMPLEMENTATION_PLAN.md section 2, with validation and JSONL read/write helpers.

Records (one JSON object per line):
    CorpusRecord          corpus.jsonl            {"doc_id", "title", "text", "is_poison": false}
    PoisonRecord          poison_*.jsonl          {"doc_id", "qid", "attack", "title", "text", "is_poison": true,
                                                   "target_answer"}
    QuestionRecord        questions.jsonl         {"qid", "question", "answers", "gold_doc_ids", "role", "fold"}
    QuestionAllRecord     questions_all.jsonl     {"qid", "question", "answers", "gold_doc_ids"}  (added in step 2)
    RetrievalCacheRecord  top150_<retriever>.jsonl {"qid", "retriever", "top150": [[doc_id, score], ...]}
In-memory interfaces: VectorAPI, FilterContext, FilterResult, FilterFn, GenerationOutput.

Validation is strict: every field is required, unknown fields are rejected, types must match exactly
(bool is not accepted as int/float and vice versa). Any violation raises SchemaError. Changing a format
must be recorded in docs/HANDOFF.md under "Interface changes".
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, TypedDict, TypeVar, get_args, get_origin, get_type_hints

import numpy as np

ATTACKS: tuple[str, ...] = ("fact", "hijack", "url", "diverse", "repeat", "stealth", "heldout")
RETRIEVERS: tuple[str, ...] = ("bm25", "dense", "splade", "rrf")
ROLES: tuple[str, ...] = ("target", "clean")
FOLDS: tuple[str, ...] = ("A", "B")
TOP_K_CACHE: int = 150
POISON_PREFIX: str = "poison_"

Attack = Literal["fact", "hijack", "url", "diverse", "repeat", "stealth", "heldout"]
Retriever = Literal["bm25", "dense", "splade", "rrf"]
Role = Literal["target", "clean"]
Fold = Literal["A", "B"]

_POISON_ID = re.compile(rf"^{POISON_PREFIX}(?P<qid>.+)_(?P<attack>{'|'.join(ATTACKS)})_(?P<n>\d+)$")


class SchemaError(ValueError):
    """A record does not match its data contract."""


# ---------------------------------------------------------------- poison IDs

def is_poison_id(doc_id: str) -> bool:
    """True if `doc_id` names a poison passage (`poison_<qid>_<attack>_<n>`)."""
    return doc_id.startswith(POISON_PREFIX)


def make_poison_id(qid: str, attack: str, n: int) -> str:
    """Build `poison_<qid>_<attack>_<n>`."""
    if attack not in ATTACKS:
        raise SchemaError(f"unknown attack {attack!r}; expected one of {ATTACKS}")
    return f"{POISON_PREFIX}{qid}_{attack}_{n}"


def parse_poison_id(doc_id: str) -> tuple[str, str, int]:
    """Split `poison_<qid>_<attack>_<n>` into (qid, attack, n); raises SchemaError if malformed."""
    match = _POISON_ID.match(doc_id)
    if not match:
        raise SchemaError(f"malformed poison doc_id {doc_id!r}; expected poison_<qid>_<attack>_<n> "
                          f"with attack in {ATTACKS}")
    return match["qid"], match["attack"], int(match["n"])


# ---------------------------------------------------------------- type checking

def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Literal:
        return "one of " + "|".join(map(str, get_args(tp)))
    if origin in (list, tuple):
        return f"{origin.__name__}[{', '.join(_type_name(a) for a in get_args(tp))}]"
    return getattr(tp, "__name__", str(tp))


def _matches(value: Any, tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is Literal:
        return isinstance(value, str) and value in get_args(tp)
    if origin is list:
        (item_tp,) = get_args(tp)
        return isinstance(value, list) and all(_matches(v, item_tp) for v in value)
    if origin is tuple:  # JSON has no tuples: a list of the right length is accepted
        args = get_args(tp)
        return (isinstance(value, (list, tuple)) and len(value) == len(args)
                and all(_matches(v, a) for v, a in zip(value, args)))
    if tp is bool:
        return type(value) is bool
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))
    if tp is str:
        return isinstance(value, str)
    raise TypeError(f"unsupported schema type {tp!r}")


# ---------------------------------------------------------------- records

R = TypeVar("R", bound="Record")


@dataclass(frozen=True)
class Record:
    """Base class: field types are checked on construction; `from_dict` also checks the key set."""

    _hints: ClassVar[dict[type, dict[str, Any]]] = {}

    @classmethod
    def _field_types(cls) -> dict[str, Any]:
        if cls not in Record._hints:
            hints = get_type_hints(cls)
            Record._hints[cls] = {f.name: hints[f.name] for f in dataclasses.fields(cls)}
        return Record._hints[cls]

    def __post_init__(self) -> None:
        for name, tp in self._field_types().items():
            value = getattr(self, name)
            if not _matches(value, tp):
                raise SchemaError(f"{type(self).__name__}.{name}: expected {_type_name(tp)}, "
                                  f"got {type(value).__name__} {value!r:.80}")
        self._check()

    def _check(self) -> None:
        """Value-level rules beyond types (overridden by subclasses)."""

    @classmethod
    def from_dict(cls: type[R], data: Mapping[str, Any]) -> R:
        """Validate a parsed JSON object and build the record. Raises SchemaError."""
        if not isinstance(data, Mapping):
            raise SchemaError(f"{cls.__name__}: expected a JSON object, got {type(data).__name__}")
        expected = set(cls._field_types())
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        if missing:
            raise SchemaError(f"{cls.__name__}: missing field(s) {missing}")
        if extra:
            raise SchemaError(f"{cls.__name__}: unknown field(s) {extra} (contract changes go in docs/HANDOFF.md)")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready dict (tuples become lists when serialised)."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CorpusRecord(Record):
    """One clean passage of `corpus.jsonl`. doc_id is the BEIR `_id`."""

    doc_id: str
    title: str
    text: str
    is_poison: bool

    def _check(self) -> None:
        if not self.doc_id:
            raise SchemaError("CorpusRecord.doc_id must be non-empty")
        if is_poison_id(self.doc_id):
            raise SchemaError(f"CorpusRecord.doc_id {self.doc_id!r} must not start with {POISON_PREFIX!r}")
        if self.is_poison:
            raise SchemaError(f"CorpusRecord {self.doc_id!r}: is_poison must be false")


@dataclass(frozen=True)
class PoisonRecord(Record):
    """One poison passage of Arya's `poison_*.jsonl`. doc_id must be `poison_<qid>_<attack>_<n>`."""

    doc_id: str
    qid: str
    attack: Attack
    title: str
    text: str
    is_poison: bool
    target_answer: str

    def _check(self) -> None:
        qid, attack, _ = parse_poison_id(self.doc_id)
        if qid != self.qid or attack != self.attack:
            raise SchemaError(f"PoisonRecord {self.doc_id!r}: doc_id says qid={qid!r} attack={attack!r}, "
                              f"but fields say qid={self.qid!r} attack={self.attack!r}")
        if not self.is_poison:
            raise SchemaError(f"PoisonRecord {self.doc_id!r}: is_poison must be true")
        if not self.target_answer:
            raise SchemaError(f"PoisonRecord {self.doc_id!r}: target_answer must be non-empty")


@dataclass(frozen=True)
class QuestionRecord(Record):
    """One question of `questions.jsonl`."""

    qid: str
    question: str
    answers: list[str]
    gold_doc_ids: list[str]
    role: Role
    fold: Fold

    def _check(self) -> None:
        if not self.qid or not self.question:
            raise SchemaError(f"QuestionRecord {self.qid!r}: qid and question must be non-empty")


@dataclass(frozen=True)
class QuestionAllRecord(Record):
    """One question of `questions_all.jsonl`: the eligible pool the 400-question set is drawn from.

    No role/fold yet. Only questions with >= 1 gold passage in the (full) corpus and >= 1 answer string.
    """

    qid: str
    question: str
    answers: list[str]
    gold_doc_ids: list[str]

    def _check(self) -> None:
        if not self.qid or not self.question:
            raise SchemaError(f"QuestionAllRecord {self.qid!r}: qid and question must be non-empty")
        if not self.answers or not self.gold_doc_ids:
            raise SchemaError(f"QuestionAllRecord {self.qid!r}: needs >= 1 answer and >= 1 gold_doc_id")


@dataclass(frozen=True)
class RetrievalCacheRecord(Record):
    """One row of `top150_<retriever>.jsonl`: raw (doc_id, score) pairs sorted by score descending."""

    qid: str
    retriever: Retriever
    top150: list[tuple[str, float]]

    def __post_init__(self) -> None:
        super().__post_init__()
        # normalise JSON lists [doc_id, score] to tuples so records compare equal after a round trip
        object.__setattr__(self, "top150", [(d, float(s)) for d, s in self.top150])

    def _check(self) -> None:
        if len(self.top150) > TOP_K_CACHE:
            raise SchemaError(f"RetrievalCacheRecord {self.qid!r}: {len(self.top150)} entries > {TOP_K_CACHE}")
        ids = [d for d, _ in self.top150]
        if len(set(ids)) != len(ids):
            raise SchemaError(f"RetrievalCacheRecord {self.qid!r}: duplicate doc_ids in top150")
        scores = [s for _, s in self.top150]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise SchemaError(f"RetrievalCacheRecord {self.qid!r}: top150 is not sorted by score descending")


# ---------------------------------------------------------------- JSONL helpers

def iter_jsonl(path: str | Path, record_type: type[R]) -> Iterator[R]:
    """Stream validated records from a JSONL file. Errors name the file and line number."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield record_type.from_dict(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path.name} line {lineno}: invalid JSON ({exc.msg})") from exc
            except SchemaError as exc:
                raise SchemaError(f"{path.name} line {lineno}: {exc}") from exc


def read_jsonl(path: str | Path, record_type: type[R]) -> list[R]:
    """Read and validate a whole JSONL file."""
    return list(iter_jsonl(path, record_type))


def write_jsonl(path: str | Path, records: Iterable[Record], append: bool = False) -> int:
    """Write records (already validated on construction) as JSONL; returns the number written.

    append=False writes atomically (temp file + rename), so a crash never leaves a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if append else path.with_name(path.name + ".tmp")
    n = 0
    with target.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            if not isinstance(rec, Record):
                raise SchemaError(f"write_jsonl expects Record instances, got {type(rec).__name__}")
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    if not append:
        os.replace(target, path)
    return n


# ---------------------------------------------------------------- in-memory interfaces

class VectorAPI(Protocol):
    """Dense vectors from the dense retriever's encoder (Step 4); rows are L2-normalised."""

    def get_doc_vector(self, doc_ids: list[str]) -> np.ndarray:
        """[n, d] passage vectors read from the stored matrix (no re-encoding)."""
        ...

    def get_query_vector(self, texts: list[str]) -> np.ndarray:
        """[n, d] query vectors (with the model's query prefix)."""
        ...


@dataclass
class FilterContext:
    """Everything a filter may need besides the pool: texts, vectors, config, fold-specific thresholds."""

    texts: Mapping[str, str]                 # doc_id -> passage text
    vectors: VectorAPI | None                # None if the filter does not use vectors
    config: Any                              # ire.config.Config (or a mapping in tests)
    thresholds: Mapping[str, Any] = field(default_factory=dict)  # calibrated on the opposite fold


@dataclass
class FilterResult:
    """Output of `apply_filter`: flagged doc_ids and per-doc scores (filter-specific meaning)."""

    flagged: set[str]
    scores: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.flagged, set) or not all(isinstance(d, str) for d in self.flagged):
            raise SchemaError("FilterResult.flagged must be a set of doc_id strings")
        if not isinstance(self.scores, dict) or not all(
                isinstance(k, str) and _matches(v, float) for k, v in self.scores.items()):
            raise SchemaError("FilterResult.scores must be a dict of doc_id -> finite float")


# apply_filter(pool, qid, question, ctx) -> FilterResult; pool = ranked doc_ids (top-50 after masking)
FilterFn = Callable[[list[str], str, str, FilterContext], FilterResult]


class GenerationOutput(TypedDict):
    """Return value of `generate(qid, question, top5_doc_ids)`."""

    answer: str
    cached: bool
