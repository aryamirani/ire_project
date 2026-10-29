"""Perplexity filter and per-doc score cache (plan section 4.3; IMPLEMENTATION_PLAN step 11).

Scores (GPT-2 small, `models.perplexity`):
    split each passage at its token midpoint into "pre" and "post";
    f(R)  = mean negative log-likelihood per token of R, scored on its own after GPT-2's <|endoftext|> token;
    PD(p) = f(pre) - f(post)       two-tailed (a fluency break between the halves moves it either way);
    PM(p) = max(f(pre), f(post))   one-tailed (high = unusually unfluent).
Passages with fewer than `perplexity.min_tokens` (20) GPT-2 tokens are never flagged (`too_short`).
The passage text is what the filters see everywhere: ctx.texts[doc_id] (title + " " + text).

Cache: SQLite `caches/perplexity.sqlite`, one row per (namespace, doc_id) with
    {"doc_id", "ppl_pre", "ppl_post", "pd", "pm", "nll_pre", "nll_post", "n_tokens", "too_short"}
(ppl = exp(nll); PD and PM are computed on f, the mean NLL, as in the plan's formulas). The namespace is the model
id + revision + split rule + min_tokens, so a different model never reuses old scores.

Thresholds (configs/perplexity_thresholds.json, written by scripts/calibrate_perplexity.py = job A8): for every
alpha in `perplexity.alpha_sweep`, from 1,000 seeded random clean corpus passages:
    PD band [q(alpha / 2), q(1 - alpha / 2)]   PM cutoff q(1 - alpha)
`perplexity.rule`: pd | pm | either (default: flag if either rule fires). ctx.thresholds["alpha"] picks the
alpha (default `perplexity.alpha`, anchor 0.025). FilterResult.scores[doc_id] = PM (0.0 for too_short).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ire.schemas import FilterContext, FilterResult

SPLIT_RULE = "token_midpoint_v1"
RULES: tuple[str, ...] = ("pd", "pm", "either")


# ---------------------------------------------------------------- scoring

class PerplexityScorer:
    """GPT-2 half-passage scorer. Loads the model lazily; counts how many passages it actually scored."""

    def __init__(self, model: str, device: str = "cuda", min_tokens: int = 20, batch_tokens: int = 8192,
                 max_tokens: int = 1023, max_vram_gb: float | None = None) -> None:
        self.model_name, self.device_req, self.max_vram_gb = model, device, max_vram_gb
        # batches are bounded by padded tokens, not passages: the [tokens x 50,257] logits dominate GPU memory
        # (32 long passages needed > 6 GiB on an 11 GB card); 8,192 tokens keep it near 1.6 GiB
        self.min_tokens, self.batch_tokens, self.max_tokens = min_tokens, batch_tokens, max_tokens
        self._model = self._tok = None
        self.revision: str | None = None
        self._path: Path | None = None
        self.n_scored = 0

    @classmethod
    def from_config(cls, cfg: Any) -> PerplexityScorer:
        return cls(cfg["models.perplexity"], device=cfg["compute.device"], min_tokens=int(cfg["perplexity.min_tokens"]),
                   max_vram_gb=cfg.get("compute.max_vram_gb"))

    def _resolve(self) -> Path:
        """Local snapshot path and pinned revision, without loading any weights."""
        if self.revision is None:
            from ire.retrieval.common import resolve_model

            self._path, self.revision = resolve_model(self.model_name)
        return self._path

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ire.retrieval.common import setup_device

        path = self._resolve()
        # float32: NLLs are compared across passages; the VRAM cap makes an overflow a clean OutOfMemoryError
        self.device, _ = setup_device(self.device_req, False, self.max_vram_gb)
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(self.device).eval()
        self.torch = torch

    def tokenizer(self):
        self._load()
        return self._tok

    def namespace(self) -> str:
        self._resolve()  # revision only: a lookup that is all cache hits loads no model
        spec = {"model": self.model_name, "revision": self.revision, "split": SPLIT_RULE,
                "min_tokens": self.min_tokens, "max_tokens": self.max_tokens}
        return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def _batches(self, order: list[int], lengths: list[int]) -> list[list[int]]:
        """Consecutive groups of `order` (sorted by length) whose padded size stays within batch_tokens."""
        out, cur = [], []
        for i in order:
            width = lengths[i] + 1
            if cur and (len(cur) + 1) * width > self.batch_tokens:
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    def _mean_nll(self, seqs: list[list[int]]) -> list[float]:
        """Mean NLL per token of each token sequence, each scored after <|endoftext|> (batched, right-padded)."""
        torch = self.torch
        eos = self._tok.eos_token_id
        seqs = [s[: self.max_tokens] for s in seqs]
        lengths = [len(s) for s in seqs]
        order = sorted(range(len(seqs)), key=lambda i: lengths[i])
        res = [0.0] * len(seqs)
        batches = self._batches(order, lengths)
        k = 0
        while k < len(batches):
            idx = batches[k]
            try:
                per = self._batch_mean_nll([[eos] + seqs[i] for i in idx], eos)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                # halve the budget and re-split the passages not scored yet (same order, smaller batches)
                self.batch_tokens = max(1, self.batch_tokens // 2)
                batches[k:] = self._batches([i for b in batches[k:] for i in b], lengths)
                continue
            for j, i in enumerate(idx):
                res[i] = per[j]
            k += 1
        return res

    def _batch_mean_nll(self, rows: list[list[int]], pad: int) -> list[float]:
        """Mean NLL of each right-padded row. The loss is computed one row at a time on views of the logits: the
        batched form (a reshape of a sliced tensor) copied the [tokens x 50,257] logits and needed ~3x their size,
        which ran out of memory on a 6 GB card. Same per-position values and the same masked mean."""
        torch = self.torch
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, : len(r)] = torch.tensor(r)
            mask[j, : len(r)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            logits = self._model(input_ids=ids, attention_mask=mask).logits
            nll = torch.empty((len(rows), width - 1), dtype=torch.float32, device=self.device)
            for j in range(len(rows)):
                nll[j] = torch.nn.functional.cross_entropy(logits[j, :-1].float(), ids[j, 1:], reduction="none")
            del logits
        m = mask[:, 1:].float()
        per = (nll * m).sum(1) / m.sum(1)
        return [float(x) for x in per]

    def score_texts(self, items: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
        """[(doc_id, text)] -> score records (see the module docstring)."""
        self._load()
        recs: list[dict[str, Any]] = []
        halves: list[list[int]] = []
        owners: list[int] = []
        for doc_id, text in items:
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            rec: dict[str, Any] = {"doc_id": doc_id, "n_tokens": len(ids), "too_short": len(ids) < self.min_tokens}
            recs.append(rec)
            if not rec["too_short"]:
                mid = len(ids) // 2
                halves += [ids[:mid], ids[mid:]]
                owners.append(len(recs) - 1)
        nlls = self._mean_nll(halves) if halves else []
        for k, ri in enumerate(owners):
            pre, post = nlls[2 * k], nlls[2 * k + 1]
            recs[ri].update({"nll_pre": pre, "nll_post": post, "ppl_pre": math.exp(pre), "ppl_post": math.exp(post),
                             "pd": pre - post, "pm": max(pre, post)})
        for rec in recs:
            if rec["too_short"]:
                rec.update({"nll_pre": None, "nll_post": None, "ppl_pre": None, "ppl_post": None, "pd": None,
                            "pm": None})
        self.n_scored += len(items)
        return recs


# ---------------------------------------------------------------- cache

class PerplexityCache:
    """SQLite (WAL) store: scores(namespace, doc_id) -> JSON score record."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._con.execute("PRAGMA journal_mode = WAL")
        with self._con:
            self._con.execute("""CREATE TABLE IF NOT EXISTS scores (
                namespace TEXT NOT NULL, doc_id TEXT NOT NULL, record TEXT NOT NULL, PRIMARY KEY (namespace, doc_id))""")

    def get_many(self, namespace: str, doc_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        uniq = list(dict.fromkeys(doc_ids))
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(uniq), 500):
            part = uniq[start:start + 500]
            q = f"SELECT doc_id, record FROM scores WHERE namespace = ? AND doc_id IN ({','.join('?' * len(part))})"
            for d, rec in self._con.execute(q, [namespace, *part]):
                out[d] = json.loads(rec)
        return out

    def put_many(self, namespace: str, records: Iterable[Mapping[str, Any]]) -> None:
        with self._con:
            self._con.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                  [(namespace, r["doc_id"], json.dumps(r)) for r in records])

    def count(self, namespace: str | None = None) -> int:
        if namespace is None:
            return self._con.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        return self._con.execute("SELECT COUNT(*) FROM scores WHERE namespace = ?", (namespace,)).fetchone()[0]

    def close(self) -> None:
        self._con.close()


class PerplexityScores:
    """Scorer + cache: `get(doc_ids, texts)` returns cached records and scores only the misses."""

    def __init__(self, scorer: PerplexityScorer, cache: PerplexityCache, chunk: int = 2000) -> None:
        self.scorer, self.cache, self.chunk = scorer, cache, chunk

    @classmethod
    def from_config(cls, cfg: Any) -> PerplexityScores:
        return cls(PerplexityScorer.from_config(cfg), PerplexityCache(cfg.path("perplexity_cache")))

    def get(self, doc_ids: Sequence[str], texts: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        ns = self.scorer.namespace()
        have = self.cache.get_many(ns, doc_ids)
        missing = [d for d in dict.fromkeys(doc_ids) if d not in have]
        for start in range(0, len(missing), self.chunk):
            part = missing[start:start + self.chunk]
            recs = self.scorer.score_texts([(d, texts[d]) for d in part])
            self.cache.put_many(ns, recs)
            have.update({r["doc_id"]: r for r in recs})
        return have


_default: PerplexityScores | None = None


def default_scores(cfg: Any) -> PerplexityScores:
    global _default
    if _default is None:
        _default = PerplexityScores.from_config(cfg)
    return _default


# ---------------------------------------------------------------- thresholds

def alpha_key(alpha: float) -> str:
    return f"{float(alpha):g}"


def calibrate(records: Sequence[Mapping[str, Any]], alphas: Sequence[float]) -> dict[str, dict[str, float]]:
    """alpha -> {"pd_low", "pd_high", "pm_high"} from clean score records (too_short ones are skipped)."""
    pd = np.array([r["pd"] for r in records if not r["too_short"]], dtype=np.float64)
    pm = np.array([r["pm"] for r in records if not r["too_short"]], dtype=np.float64)
    if pd.size == 0:
        raise ValueError("no scorable passages to calibrate on")
    return {alpha_key(a): {"pd_low": float(np.quantile(pd, a / 2)), "pd_high": float(np.quantile(pd, 1 - a / 2)),
                           "pm_high": float(np.quantile(pm, 1 - a))} for a in alphas}


def flags(rec: Mapping[str, Any], th: Mapping[str, float], rule: str = "either") -> dict[str, bool]:
    """Which rules fire for one score record under one alpha's thresholds."""
    if rec["too_short"]:
        return {"pd": False, "pm": False, "flagged": False}
    pd_fire = rec["pd"] < th["pd_low"] or rec["pd"] > th["pd_high"]
    pm_fire = rec["pm"] > th["pm_high"]
    if rule not in RULES:
        raise ValueError(f"perplexity.rule must be one of {RULES}, got {rule!r}")
    return {"pd": pd_fire, "pm": pm_fire,
            "flagged": {"pd": pd_fire, "pm": pm_fire, "either": pd_fire or pm_fire}[rule]}


def load_thresholds(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the filter

def _cfg(ctx: FilterContext, key: str, default: Any) -> Any:
    cfg = ctx.config
    return cfg.get(key, default) if hasattr(cfg, "get") else default


def _setup(ctx: FilterContext) -> tuple[dict[str, float], str, Any]:
    alpha = float(ctx.thresholds.get("alpha", _cfg(ctx, "perplexity.alpha", 0.025)))
    table = ctx.thresholds.get("table")
    if table is None:
        table = load_thresholds(ctx.config.path("perplexity_thresholds"))
    th = table["thresholds"][alpha_key(alpha)]
    rule = str(_cfg(ctx, "perplexity.rule", "either"))
    scores = ctx.thresholds.get("scores") or default_scores(ctx.config)
    if table.get("namespace") and table["namespace"] != scores.scorer.namespace():
        # thresholds are percentiles of THIS scorer's PD / PM; another GPT-2 revision or split rule needs new ones
        raise ValueError(f"perplexity thresholds were calibrated for scorer namespace {table['namespace']}, the "
                         f"scorer is {scores.scorer.namespace()}: re-run scripts/calibrate_perplexity.py")
    return th, rule, scores


def explain(pool: list[str], question: str, ctx: FilterContext) -> list[dict[str, Any]]:
    """Per-passage score record plus which rule fired."""
    th, rule, scores = _setup(ctx)
    recs = scores.get(pool, ctx.texts)
    return [{**recs[d], "rank": i + 1, **flags(recs[d], th, rule)} for i, d in enumerate(pool)]


def apply_filter(pool: list[str], qid: str, question: str, ctx: FilterContext) -> FilterResult:
    """Flag passages whose PD falls outside the clean band or whose PM exceeds the clean cutoff."""
    th, rule, scores = _setup(ctx)
    recs = scores.get(pool, ctx.texts)
    flagged = {d for d in pool if flags(recs[d], th, rule)["flagged"]}
    return FilterResult(flagged=flagged, scores={d: (0.0 if recs[d]["too_short"] else float(recs[d]["pm"]))
                                                 for d in pool})
