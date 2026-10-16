"""BEIR NQ + nq_open: stream the corpus, join questions to answers, select questions and passages (Step 2).

Two equivalent sources of BEIR NQ: the original zip (read in place, never extracted) or the Hugging Face copy
(parquet + TSV). The corpus is always streamed (2.68M passages do not fit comfortably in 16 GB as Python
objects). Every selection is seeded and deterministic.
"""

from __future__ import annotations

import csv
import io
import json
import random
import unicodedata
import zipfile
from collections.abc import Container, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ire.schemas import (
    CorpusRecord,
    QuestionAllRecord,
    QuestionRecord,
    SchemaError,
    is_poison_id,
    write_jsonl,
)

# ---------------------------------------------------------------- reading the BEIR zip


def normalize_question(text: str) -> str:
    """Join key for BEIR queries vs nq_open: NFKC, lowercase, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return " ".join(text.split())


@dataclass(frozen=True)
class BeirZip:
    """The original BEIR `nq.zip` (members `*/corpus.jsonl`, `*/queries.jsonl`, `*/qrels/<split>.tsv`), read in place."""

    path: Path

    def _member(self, zf: zipfile.ZipFile, suffix: str) -> str:
        names = [n for n in zf.namelist() if n.endswith(suffix)]
        if len(names) != 1:
            raise FileNotFoundError(f"expected exactly one '*{suffix}' in {self.path}, found {names}")
        return names[0]

    def _lines(self, suffix: str) -> Iterator[str]:
        with zipfile.ZipFile(self.path) as zf, zf.open(self._member(zf, suffix)) as raw:
            yield from io.TextIOWrapper(raw, encoding="utf-8")

    def iter_corpus(self) -> Iterator[dict[str, Any]]:
        for line in self._lines("corpus.jsonl"):
            if line.strip():
                yield json.loads(line)

    def iter_queries(self) -> Iterator[dict[str, Any]]:
        for line in self._lines("queries.jsonl"):
            if line.strip():
                yield json.loads(line)

    def qrels_lines(self, split: str) -> Iterator[str]:
        return self._lines(f"qrels/{split}.tsv")


@dataclass(frozen=True)
class BeirParquet:
    """The Hugging Face copy of BEIR NQ (`BeIR/nq` corpus + queries parquet, `BeIR/nq-qrels` TSV files)."""

    corpus_path: Path
    queries_path: Path
    qrels_dir: Path

    def iter_corpus(self, batch_size: int = 20_000) -> Iterator[dict[str, Any]]:
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(self.corpus_path).iter_batches(batch_size, columns=["_id", "title", "text"]):
            yield from batch.to_pylist()  # streamed batch by batch, never the whole table

    def iter_queries(self) -> Iterator[dict[str, Any]]:
        import pyarrow.parquet as pq

        yield from pq.read_table(self.queries_path, columns=["_id", "text"]).to_pylist()

    def qrels_lines(self, split: str) -> Iterator[str]:
        with (self.qrels_dir / f"{split}.tsv").open("r", encoding="utf-8") as fh:
            yield from fh


BeirSource = BeirZip | BeirParquet


def _source(src: BeirSource | Path) -> BeirSource:
    return BeirZip(Path(src)) if isinstance(src, (str, Path)) else src


def iter_beir_corpus(src: BeirSource | Path) -> Iterator[dict[str, Any]]:
    """Yield the raw BEIR corpus objects (`_id`, `title`, `text`, ...) in file order. A Path means a BEIR zip."""
    return _source(src).iter_corpus()


def load_beir_queries(src: BeirSource | Path) -> dict[str, str]:
    """qid -> query text, in file order."""
    return {str(obj["_id"]): obj["text"] for obj in _source(src).iter_queries()}


def load_beir_qrels(src: BeirSource | Path, split: str = "test") -> dict[str, list[str]]:
    """qid -> relevant doc_ids (score > 0), in file order, without duplicates."""
    out: dict[str, list[str]] = {}
    reader = csv.reader(_source(src).qrels_lines(split), delimiter="	")
    header = next(reader)
    if header[:2] != ["query-id", "corpus-id"]:
        raise ValueError(f"unexpected qrels header {header}")
    for row in reader:
        if len(row) >= 3 and int(row[2]) > 0:
            docs = out.setdefault(row[0], [])
            if row[1] not in docs:
                docs.append(row[1])
    return out


def load_nq_open(parquet_paths: Mapping[str, Path]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """normalised question -> answers (union over rows and splits, first-seen order), plus stats."""
    import pyarrow.parquet as pq

    answers: dict[str, list[str]] = {}
    split_of: dict[str, str] = {}
    stats: dict[str, Any] = {"rows": {}, "questions_with_several_rows": 0}
    seen_rows: dict[str, int] = {}
    for split, path in parquet_paths.items():
        rows = pq.read_table(path, columns=["question", "answer"]).to_pylist()
        stats["rows"][split] = len(rows)
        for row in rows:
            key = normalize_question(row["question"])
            seen_rows[key] = seen_rows.get(key, 0) + 1
            split_of.setdefault(key, split)
            bucket = answers.setdefault(key, [])
            for ans in row["answer"]:
                ans = ans.strip()
                if ans and ans not in bucket:
                    bucket.append(ans)
    stats["unique_questions"] = len(answers)
    stats["questions_with_several_rows"] = sum(1 for n in seen_rows.values() if n > 1)
    stats["_split_of"] = split_of  # internal: which split a question was first seen in
    return answers, stats


# ---------------------------------------------------------------- corpus scan


@dataclass
class CorpusScan:
    """Result of one streaming pass over the corpus."""

    n_passages: int = 0
    gold_positions: dict[str, int] = field(default_factory=dict)  # gold doc_id -> line index (first occurrence)
    duplicate_ids: list[str] = field(default_factory=list)
    poison_like_ids: list[str] = field(default_factory=list)       # ids starting with "poison_" (must be none)


def scan_corpus(src: BeirSource | Path, gold_ids: Container[str]) -> CorpusScan:
    """Count passages, find the line index of every gold id, and detect duplicate / poison-like ids."""
    scan = CorpusScan()
    seen: set[str] = set()
    for i, obj in enumerate(iter_beir_corpus(src)):
        scan.n_passages = i + 1  # lines, including any duplicates
        doc_id = str(obj["_id"])
        if doc_id in seen:
            scan.duplicate_ids.append(doc_id)
            continue
        seen.add(doc_id)
        if is_poison_id(doc_id):
            scan.poison_like_ids.append(doc_id)
        if doc_id in gold_ids:
            scan.gold_positions[doc_id] = i
    return scan


# ---------------------------------------------------------------- questions


def build_question_pool(queries: Mapping[str, str], qrels: Mapping[str, Sequence[str]],
                        answers_by_norm: Mapping[str, list[str]], ids_in_corpus: Container[str],
                        split_of: Mapping[str, str] | None = None) -> tuple[list[QuestionAllRecord], dict[str, Any]]:
    """Eligible questions: in qrels, >= 1 gold passage in the corpus and >= 1 nq_open answer.

    Order follows `queries`. Returns the pool and join/coverage counts.
    """
    stats = {"queries": len(queries), "queries_with_qrels": 0, "queries_with_gold_in_corpus": 0,
             "queries_matched_to_nq_open": 0, "matched_by_split": {}, "eligible": 0}
    pool: list[QuestionAllRecord] = []
    for qid, text in queries.items():
        if qid not in qrels:
            continue  # not a test-split query
        stats["queries_with_qrels"] += 1
        gold = [d for d in qrels[qid] if d in ids_in_corpus]
        key = normalize_question(text)
        answers = answers_by_norm.get(key, [])
        if gold:
            stats["queries_with_gold_in_corpus"] += 1
        if answers:
            stats["queries_matched_to_nq_open"] += 1
            if split_of is not None:
                split = split_of.get(key, "?")
                stats["matched_by_split"][split] = stats["matched_by_split"].get(split, 0) + 1
        if gold and answers:
            pool.append(QuestionAllRecord(qid=qid, question=text, answers=list(answers), gold_doc_ids=gold))
    stats["eligible"] = len(pool)
    return pool, stats


def select_placeholder(pool: Sequence[QuestionAllRecord], n_target: int = 200, n_clean: int = 200,
                       folds: Sequence[str] = ("A", "B"), seed: int = 13) -> list[QuestionRecord]:
    """Seeded sample of n_target + n_clean questions; each role is split evenly over the folds.

    With the defaults: target/A, target/B, clean/A, clean/B have 100 questions each.
    """
    if len(pool) < n_target + n_clean:
        raise ValueError(f"pool has {len(pool)} questions, need {n_target + n_clean}")
    for n in (n_target, n_clean):
        if n % len(folds):
            raise ValueError(f"{n} questions cannot be split evenly over folds {list(folds)}")
    chosen = random.Random(seed).sample(list(pool), n_target + n_clean)
    out: list[QuestionRecord] = []
    for role, group in (("target", chosen[:n_target]), ("clean", chosen[n_target:])):
        per_fold = len(group) // len(folds)
        for i, q in enumerate(group):
            out.append(QuestionRecord(qid=q.qid, question=q.question, answers=list(q.answers),
                                      gold_doc_ids=list(q.gold_doc_ids), role=role, fold=folds[i // per_fold]))
    return out


def select_extra_questions(pool: Sequence[QuestionAllRecord], exclude_qids: Container[str], n: int,
                           seed: int = 13) -> list[QuestionAllRecord]:
    """Seeded sample of n more pool questions not in exclude_qids (used by --dev-subset)."""
    rest = [q for q in pool if q.qid not in exclude_qids]
    if len(rest) < n:
        raise ValueError(f"only {len(rest)} questions left, need {n}")
    return random.Random(seed).sample(rest, n)


# ---------------------------------------------------------------- passages


def choose_passages(n_total: int, gold_positions: set[int], n_passages: int, seed: int = 13) -> set[int]:
    """Line indices to keep: every gold position plus seeded uniform random others, n_passages in total."""
    n_fill = n_passages - len(gold_positions)
    if n_fill < 0:
        raise ValueError(f"{len(gold_positions)} gold passages already exceed n_passages={n_passages}")
    if n_passages > n_total:
        raise ValueError(f"n_passages={n_passages} > corpus size {n_total}")
    rng = random.Random(seed)
    chosen = set(gold_positions)
    while len(chosen) < n_passages:
        chosen.add(rng.randrange(n_total))
    return chosen


def write_corpus(src: BeirSource | Path, out_path: Path, keep: set[int] | None = None) -> int:
    """Stream the BEIR corpus to `corpus.jsonl` (contract format); keep only line indices in `keep` if given.

    Duplicate ids (if any) keep their first occurrence. Returns the number of passages written.
    """
    def records() -> Iterator[CorpusRecord]:
        seen: set[str] = set()
        for i, obj in enumerate(iter_beir_corpus(src)):
            if keep is not None and i not in keep:
                continue
            doc_id = str(obj["_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                yield CorpusRecord(doc_id=doc_id, title=obj.get("title") or "", text=obj.get("text") or "",
                                   is_poison=False)
            except SchemaError as exc:
                raise SchemaError(f"corpus line {i + 1}: {exc}") from exc

    return write_jsonl(out_path, records())
