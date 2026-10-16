"""Retrieval sanity metrics for index builds (Steps 3-5): gold-passage Recall@k and Hit@k.

These only check that an index works; the project's real metrics (nDCG, ASR, ...) are Vansh's (eval/).
"""

from __future__ import annotations

import random
from collections.abc import Container, Sequence
from typing import Any

from ire.schemas import QuestionAllRecord, QuestionRecord

Question = QuestionRecord | QuestionAllRecord


def questions_with_gold_in_index(questions: Sequence[Question], ids_in_index: Container[str]) -> list[tuple[str, str, list[str]]]:
    """(qid, question, gold ids present in the index) for questions with >= 1 gold passage in the index."""
    out = []
    for q in questions:
        gold = [d for d in q.gold_doc_ids if d in ids_in_index]
        if gold:
            out.append((q.qid, q.question, gold))
    return out


def sample_eval_set(items: Sequence[Any], n: int, seed: int = 13) -> list[Any]:
    """All items if there are at most n, else a seeded sample of n (kept in input order)."""
    if len(items) <= n:
        return list(items)
    keep = set(random.Random(seed).sample(range(len(items)), n))
    return [x for i, x in enumerate(items) if i in keep]


def recall_at_k(results: Sequence[Sequence[tuple[str, float]]], gold: Sequence[Sequence[str]],
                ks: Sequence[int]) -> dict[str, float]:
    """Mean over questions of |gold in top-k| / |gold| (recall@k) and of [any gold in top-k] (hit@k)."""
    if len(results) != len(gold) or not gold:
        raise ValueError(f"need one result list per question (got {len(results)} results, {len(gold)} questions)")
    out: dict[str, float] = {}
    for k in ks:
        rec = hit = 0.0
        for ranked, g in zip(results, gold):
            top = {d for d, _ in ranked[:k]}
            found = sum(1 for d in g if d in top)
            rec += found / len(g)
            hit += 1.0 if found else 0.0
        out[f"recall@{k}"] = round(rec / len(gold), 4)
        out[f"hit@{k}"] = round(hit / len(gold), 4)
    return out
