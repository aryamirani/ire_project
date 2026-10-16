"""Build corpus.jsonl / questions_all.jsonl / questions.jsonl from the raw files, then verify them (Step 2).

No network here: `scripts/prepare_data.py` downloads the raw files and calls `prepare()` and `verify_outputs()`.

Corpus modes:
    full    every BEIR NQ passage (~2.68M)
    random  all gold passages of the selected questions + seeded uniform random others, n_passages in total
            (local profile: 400 placeholder questions, 5,000 passages; --dev-subset: 1,000 questions, 50,000)
    subset  ~500k: all gold + hard negatives from each retriever's top-100.
            TODO(steps 3-5): needs the retrievers; raises NotImplementedError until then.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ire.data.nq import (
    BeirSource,
    build_question_pool,
    choose_passages,
    load_beir_qrels,
    load_beir_queries,
    load_nq_open,
    scan_corpus,
    select_extra_questions,
    select_placeholder,
    write_corpus,
)
from ire.schemas import CorpusRecord, QuestionAllRecord, QuestionRecord, iter_jsonl, read_jsonl, write_jsonl

CORPUS_MODES: tuple[str, ...] = ("full", "random", "subset")
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputPaths:
    corpus: Path
    questions_all: Path
    questions: Path

    @classmethod
    def in_dir(cls, folder: Path) -> OutputPaths:
        return cls(folder / "corpus.jsonl", folder / "questions_all.jsonl", folder / "questions.jsonl")


def prepare(src: BeirSource | Path, nq_open_paths: Mapping[str, Path], out: OutputPaths, mode: str,
            n_passages: int | None = None, seed: int = 13, n_target: int = 200, n_clean: int = 200,
            folds: Sequence[str] = ("A", "B"), dev_questions: int | None = None,
            log: logging.Logger | None = None) -> dict[str, Any]:
    """Join, select and write the three files. Returns build counts (join stats, scan, selection).

    dev_questions: if set, questions_all.jsonl holds only that many questions (the placeholder set plus seeded
    extra ones) and the random corpus covers all their gold passages (the --dev-subset layout).
    """
    log = log or _log
    if mode not in CORPUS_MODES:
        raise ValueError(f"corpus mode must be one of {CORPUS_MODES}, got {mode!r}")
    if mode == "subset":
        raise NotImplementedError("corpus mode 'subset' (~500k: gold + hard negatives from each retriever's "
                                  "top-100) needs the retrievers of steps 3-5. TODO: implement after step 5.")
    if mode == "random" and not n_passages:
        raise ValueError("corpus mode 'random' needs n_passages")

    log.info("reading BEIR queries + test qrels and nq_open answers")
    queries = load_beir_queries(src)
    qrels = load_beir_qrels(src, "test")
    answers, nq_stats = load_nq_open(nq_open_paths)
    split_of = nq_stats.pop("_split_of")
    gold_all = {d for docs in qrels.values() for d in docs}

    log.info("pass 1/2 over the corpus: counting passages and locating %d gold ids", len(gold_all))
    scan = scan_corpus(src, gold_all)
    log.info("corpus has %d passages (%d duplicate ids)", scan.n_passages, len(scan.duplicate_ids))
    if scan.poison_like_ids:
        raise ValueError(f"clean corpus has ids starting with 'poison_': {scan.poison_like_ids[:5]}")

    pool, join = build_question_pool(queries, qrels, answers, scan.gold_positions, split_of)
    placeholder = select_placeholder(pool, n_target, n_clean, folds, seed)
    if dev_questions:
        chosen = {q.qid for q in placeholder}
        as_pool = [QuestionAllRecord(qid=q.qid, question=q.question, answers=q.answers, gold_doc_ids=q.gold_doc_ids)
                   for q in placeholder]
        questions_all = as_pool + select_extra_questions(pool, chosen, dev_questions - len(placeholder), seed)
    else:
        questions_all = pool

    keep: set[int] | None = None
    n_gold_kept = None
    if mode == "random":
        corpus_questions = questions_all if dev_questions else placeholder
        gold_pos = {scan.gold_positions[d] for q in corpus_questions for d in q.gold_doc_ids}
        n_gold_kept = len(gold_pos)
        keep = choose_passages(scan.n_passages, gold_pos, int(n_passages), seed)

    log.info("pass 2/2: writing %s (mode=%s)", out.corpus.name, mode)
    n_written = write_corpus(src, out.corpus, keep)
    write_jsonl(out.questions_all, questions_all)
    write_jsonl(out.questions, placeholder)
    log.info("wrote %d passages, %d questions_all, %d questions", n_written, len(questions_all), len(placeholder))

    return {
        "raw_corpus_passages": scan.n_passages,
        "raw_duplicate_doc_ids": len(scan.duplicate_ids),
        "qrels_test_gold_ids": len(gold_all),
        "qrels_gold_ids_found_in_corpus": len(scan.gold_positions),
        "beir_test_queries": join["queries"],
        "queries_with_qrels": join["queries_with_qrels"],
        "queries_with_gold_in_corpus": join["queries_with_gold_in_corpus"],
        "queries_matched_to_nq_open": join["queries_matched_to_nq_open"],
        "matched_by_nq_open_split": join["matched_by_split"],
        "nq_open_rows": nq_stats["rows"],
        "nq_open_unique_questions": nq_stats["unique_questions"],
        "nq_open_questions_with_several_rows": nq_stats["questions_with_several_rows"],
        "eligible_questions": len(pool),
        "questions_all_written": len(questions_all),
        "questions_written": len(placeholder),
        "corpus_written": n_written,
        "corpus_gold_passages": n_gold_kept,
        "corpus_random_passages": None if n_gold_kept is None else n_written - n_gold_kept,
    }


def verify_outputs(out: OutputPaths, mode: str, n_passages: int | None, raw_unique_passages: int | None = None,
                   n_target: int = 200, n_clean: int = 200, folds: Sequence[str] = ("A", "B"),
                   pool_gold_required: bool = True) -> dict[str, Any]:
    """Re-read the three files through the schemas and run the Step 2 checks.

    pool_gold_required: whether every questions_all gold id must be in corpus.jsonl (true for full mode and the
    dev subset; false for the local random corpus, which only holds the gold of the 400 placeholder questions).
    """
    ids: set[str] = set()
    n_corpus = dup_ids = 0
    for rec in iter_jsonl(out.corpus, CorpusRecord):
        n_corpus += 1
        if rec.doc_id in ids:
            dup_ids += 1
        ids.add(rec.doc_id)
    qs = read_jsonl(out.questions, QuestionRecord)
    qall = read_jsonl(out.questions_all, QuestionAllRecord)

    missing_q = sorted({d for q in qs for d in q.gold_doc_ids if d not in ids})
    missing_all = sorted({d for q in qall for d in q.gold_doc_ids if d not in ids})
    role_fold = Counter(f"{q.role}/{q.fold}" for q in qs)
    per_cell = {"target": n_target // len(folds), "clean": n_clean // len(folds)}
    expected_cells = {f"{r}/{f}": per_cell[r] for r in ("target", "clean") for f in folds}
    if mode == "full":
        size_ok = raw_unique_passages is not None and n_corpus == raw_unique_passages
    else:
        size_ok = n_corpus == n_passages
    pool_qids = {q.qid for q in qall}

    checks = {
        "corpus_size_matches_mode": size_ok,
        "no_duplicate_doc_ids": dup_ids == 0,
        "no_duplicate_qids": len({q.qid for q in qs}) == len(qs) and len(pool_qids) == len(qall),
        "all_question_gold_in_corpus": not missing_q,
        "questions_all_gold_in_corpus": not missing_all,
        "fold_balance": dict(role_fold) == expected_cells,
        "questions_subset_of_questions_all": all(q.qid in pool_qids for q in qs),
    }
    required = dict(checks)
    if not pool_gold_required:
        required.pop("questions_all_gold_in_corpus")
    return {
        "counts": {
            "corpus_passages": n_corpus,
            "corpus_duplicate_doc_ids": dup_ids,
            "questions": len(qs),
            "questions_all": len(qall),
            "role_fold": dict(sorted(role_fold.items())),
            "question_gold_ids": len({d for q in qs for d in q.gold_doc_ids}),
            "question_gold_missing_in_corpus": len(missing_q),
            "questions_all_gold_missing_in_corpus": len(missing_all),
        },
        "checks": checks,
        "required_checks": sorted(required),
        "ok": all(required.values()),
    }


def sample_records(path: Path, n: int = 3, max_text: int = 160) -> list[dict[str, Any]]:
    """First n JSON objects of a JSONL file, long strings shortened (for printing)."""
    import json

    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(out) == n:
                break
            obj = json.loads(line)
            out.append({k: (v[:max_text] + "...") if isinstance(v, str) and len(v) > max_text else v
                        for k, v in obj.items()})
    return out
