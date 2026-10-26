"""Query-and-phrase filter (plan section 4.1; Arya's lane). Same API as every filter (IMPLEMENTATION_PLAN 2):

    apply_filter(pool, qid, question, ctx) -> FilterResult

1. Similarity signal: s(q, p) = cos(e(q), e(p)) with the dense encoder (ctx.vectors: get_query_vector /
   get_doc_vector). A passage is flagged if s > ctx.thresholds["sim_cutoff"], the (1 - tail) percentile of s over
   the clean questions' top-50 pools of the OPPOSITE fold (anchor tail 2.5%; sweep 1 / 2.5 / 5 / 10%). Cutoffs
   come from `calibrate_cutoffs` / scripts/calibrate_query_phrase.py; `thresholds_for` picks the opposite fold.
2. Phrase rule: flagged if the passage text matches any pattern of the frozen list
   configs/frozen/injection_phrases.json (normalised text, re.search).
3. Ablation only: distance of each passage from the pool centroid (`centroid_distance`). Never used as a rule.

config `query_phrase.rule`: either (default: flag if either signal fires) | similarity | phrase.
FilterResult.scores holds s(q, p) for every pool passage (the continuous score for ROC-AUC); `explain` returns
the per-passage details (similarity, phrase pattern ids, centroid distance).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ire.schemas import FOLDS, FilterContext, FilterResult

REPO_ROOT = Path(__file__).resolve().parents[3]
PHRASES_PATH = REPO_ROOT / "configs" / "frozen" / "injection_phrases.json"
RULES: tuple[str, ...] = ("either", "similarity", "phrase")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                         "„": '"', "–": "-", "—": "-", "−": "-", " ": " "})


# ---------------------------------------------------------------- phrase rule

def normalise(text: str) -> str:
    """NFKC, ASCII quotes/dashes, lowercase, single spaces (the frozen list's normalisation)."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_QUOTES).lower().split())


@lru_cache(maxsize=4)
def load_phrases(path: str | Path = PHRASES_PATH) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(pattern id, compiled regex) for every pattern of the frozen list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple((p["id"], re.compile(p["regex"], re.IGNORECASE)) for p in data["patterns"])


def phrase_hits(text: str, phrases: Sequence[tuple[str, re.Pattern[str]]] | None = None) -> list[str]:
    """Ids of the frozen patterns that match `text` ([] = no match)."""
    t = normalise(text)
    return [pid for pid, rx in (phrases or load_phrases()) if rx.search(t)]


# ---------------------------------------------------------------- similarity

def similarities(question: str, pool: Sequence[str], vectors: Any) -> np.ndarray:
    """s(q, p) for every pool passage (vectors are L2-normalised, so the dot product is the cosine)."""
    if not pool:
        return np.zeros(0, dtype=np.float32)
    q = np.asarray(vectors.get_query_vector([question]), dtype=np.float32)[0]
    d = np.asarray(vectors.get_doc_vector(list(pool)), dtype=np.float32)
    return d @ q


def centroid_distance(pool: Sequence[str], vectors: Any) -> np.ndarray:
    """Ablation: 1 - cos(p, centroid of the pool) for every pool passage."""
    if not pool:
        return np.zeros(0, dtype=np.float32)
    d = np.asarray(vectors.get_doc_vector(list(pool)), dtype=np.float32)
    c = d.mean(axis=0)
    c /= max(float(np.linalg.norm(c)), 1e-12)
    return 1.0 - d @ c


def calibrate_cutoffs(sims_by_fold: Mapping[str, Sequence[float]], tails: Sequence[float]) -> dict[str, dict[str, float]]:
    """fold -> {tail: (1 - tail) percentile of s} over that fold's clean pool similarities.

    Keys are tails formatted with repr-free strings ("0.025"), so the JSON file is stable.
    """
    out: dict[str, dict[str, float]] = {}
    for fold, sims in sims_by_fold.items():
        arr = np.asarray(sims, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"no similarities for fold {fold}")
        out[fold] = {tail_key(t): float(np.quantile(arr, 1.0 - t)) for t in tails}
    return out


def tail_key(tail: float) -> str:
    return f"{float(tail):g}"


def opposite_fold(fold: str) -> str:
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}")
    return FOLDS[1 - FOLDS.index(fold)]


def thresholds_for(table: Mapping[str, Any], retriever: str, report_fold: str, tail: float | None = None) -> dict:
    """FilterContext.thresholds for a run reported on `report_fold`: cutoff calibrated on the opposite fold.

    table: the content of configs/query_phrase_thresholds.json; tail defaults to the table's anchor.
    """
    tail = table["anchor_tail"] if tail is None else tail
    calib = opposite_fold(report_fold)
    cutoff = table["cutoffs"][retriever][calib][tail_key(tail)]
    return {"sim_cutoff": float(cutoff), "calibrated_on_fold": calib, "tail": float(tail), "retriever": retriever}


def load_thresholds(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the filter

def _rule(ctx: FilterContext) -> str:
    cfg = ctx.config
    rule = (cfg.get("query_phrase.rule", "either") if hasattr(cfg, "get") else "either") or "either"
    if rule not in RULES:
        raise ValueError(f"query_phrase.rule must be one of {RULES}, got {rule!r}")
    return rule


def explain(pool: list[str], question: str, ctx: FilterContext) -> list[dict[str, Any]]:
    """Per-passage details: similarity, phrase pattern ids, centroid distance (ablation), flags."""
    rule = _rule(ctx)
    cutoff = ctx.thresholds.get("sim_cutoff")
    sims = similarities(question, pool, ctx.vectors) if ctx.vectors is not None else None
    cdist = centroid_distance(pool, ctx.vectors) if ctx.vectors is not None else None
    out = []
    for i, doc_id in enumerate(pool):
        hits = phrase_hits(ctx.texts[doc_id])
        s = None if sims is None else float(sims[i])
        sim_flag = s is not None and cutoff is not None and s > float(cutoff)
        flagged = {"either": sim_flag or bool(hits), "similarity": sim_flag, "phrase": bool(hits)}[rule]
        out.append({"doc_id": doc_id, "rank": i + 1, "similarity": s, "similarity_flag": sim_flag,
                    "phrase_hits": hits, "centroid_distance": None if cdist is None else float(cdist[i]),
                    "flagged": flagged})
    return out


def apply_filter(pool: list[str], qid: str, question: str, ctx: FilterContext) -> FilterResult:
    """Flag passages unusually close to the question (similarity) or matching the frozen phrase list."""
    rule = _rule(ctx)
    flagged: set[str] = set()
    scores: dict[str, float] = {}
    if rule in ("either", "similarity"):
        if ctx.vectors is None:
            raise ValueError("query_phrase filter needs ctx.vectors (ire.vectors.default_vectors())")
        if "sim_cutoff" not in ctx.thresholds:
            raise KeyError("query_phrase filter needs ctx.thresholds['sim_cutoff'] from the opposite fold "
                           "(ire.defenses.query_phrase.thresholds_for)")
        cutoff = float(ctx.thresholds["sim_cutoff"])
        sims = similarities(question, pool, ctx.vectors)
        for doc_id, s in zip(pool, sims):
            scores[doc_id] = float(s)
            if s > cutoff:
                flagged.add(doc_id)
    elif ctx.vectors is not None:
        scores = {d: float(s) for d, s in zip(pool, similarities(question, pool, ctx.vectors))}
    if rule in ("either", "phrase"):
        phrases = load_phrases()
        flagged |= {d for d in pool if phrase_hits(ctx.texts[d], phrases)}
    return FilterResult(flagged=flagged, scores=scores)
