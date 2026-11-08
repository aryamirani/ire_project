"""Sweep runner (plan sections 2, 4, 6, 8): setting -> pool -> filter -> first 5 survivors -> LLM -> run records.

A *setting* is (retriever, attack, rate, filter, filter parameter). For each setting the runner processes every
question (the attacked target questions plus all clean questions, so collateral damage and false positives are
measured), writes one JSONL file of run records, and is resumable at setting granularity: a finished setting's
file exists and is skipped on re-run (written atomically). Answers go through the generation cache, so a top-5
seen before (same qid, same ordered passages) is never generated twice.

Run record (plan section 8, extended; every key is always present):
    qid, setting, retriever, attack, rate, filter, of, param, fold, role,
    pool, pool_scores, flagged, top5, n_pool, n_poison_pool, n_poison_flagged, n_clean_flagged, n_poison_top5,
    n_gold_dropped, backfill, filter_ms, ndcg5, ndcg10, recall5, recall10, ndcg5_pool, ndcg10_pool,
    answer, asr (null for clean questions and the no-poison setting), em, f1, acc
`filter` is none | query_phrase | cluster | perplexity | random | oracle; for random, `of` names the detector whose
number of flagged passages is matched and `param` is that detector's parameter.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import traceback
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ire.defenses.controls import oracle_filter, random_filter
from ire.eval import metrics
from ire.retrieval.pool import candidate_pool
from ire.schemas import PoisonRecord, QuestionRecord, is_poison_id, iter_jsonl, read_jsonl

DETECTORS = ("query_phrase", "cluster", "perplexity")
FILTERS = ("none", *DETECTORS, "random", "oracle")
ANCHOR_KEY = {"query_phrase": "tail", "cluster": "tau_c", "perplexity": "alpha"}
ANCHOR_CFG = {"query_phrase": "query_phrase.anchor_tail", "cluster": "cluster.tau_c", "perplexity": "perplexity.alpha"}


def param_tuple(params: dict[str, float] | None) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((params or {}).items()))


@dataclass(frozen=True)
class Setting:
    retriever: str
    attack: str | None
    rate: int | None
    filter: str
    param: tuple[tuple[str, float], ...] = ()
    of: str | None = None
    tier: str = "headline"

    @property
    def sid(self) -> str:
        parts = [self.retriever, self.attack or "none", f"r{self.rate or 0}", self.filter]
        if self.of:
            parts.append(f"of-{self.of}")
        parts += [f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in self.param]
        return "__".join(parts)

    @property
    def detector(self) -> Setting | None:
        """For a random control: the detector setting whose flagged counts it matches."""
        if self.filter != "random":
            return None
        return Setting(self.retriever, self.attack, self.rate, str(self.of), self.param, None, self.tier)


def anchor_setting(retriever: str, attack: str | None, rate: int | None, filter_name: str, cfg: Any,
                   of: str | None = None) -> Setting:
    """The setting at the anchor parameter of a detector (or its random control with of=detector)."""
    det = of or filter_name
    if det in ANCHOR_KEY:
        param = param_tuple({ANCHOR_KEY[det]: float(cfg[ANCHOR_CFG[det]])})
    else:
        param = ()
    return Setting(retriever, attack, rate, filter_name, param, of)


def load_sweep(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def expand(sweep: dict[str, Any], tiers: Sequence[str], retrievers: Sequence[str] | None = None,
           attacks: Sequence[str] | None = None) -> list[Setting]:
    """All settings of the chosen tiers, in tier order, without duplicates. `attacks` keeps the no-poison setting."""
    order = {t: i for i, t in enumerate(sweep["tiers"])}
    for t in tiers:
        if t not in order:
            raise ValueError(f"unknown tier {t!r}; known: {list(order)}")
    out: list[Setting] = []
    seen: set[str] = set()
    for b in sorted((b for b in sweep["blocks"] if b["tier"] in tiers), key=lambda b: order[b["tier"]]):
        f, of = b["filter"], b.get("of")
        if f not in FILTERS:
            raise ValueError(f"unknown filter {f!r} in block {b}")
        if (f == "random") != (of is not None) or (of is not None and of not in DETECTORS):
            raise ValueError(f"'of' is required for random controls (and only for them), a detector name: {b}")
        for r in (retrievers or sweep["retrievers"]):
            for a in b.get("attacks", [None]):
                if attacks is not None and a is not None and a not in attacks:
                    continue
                for rate in b.get("rates", [None]):
                    if (a is None) != (rate is None):
                        raise ValueError(f"attack and rate must both be null or both set: {b}")
                    for p in b.get("params", [{}]):
                        s = Setting(r, a, rate, f, param_tuple(p), of, b["tier"])
                        if s.sid not in seen:
                            seen.add(s.sid)
                            out.append(s)
    return out


def load_targets(cfg: Any) -> tuple[dict[tuple[str, str], list[str]], dict[str, set[str]]]:
    """({(qid, attack): [target strings]}, {attack: {qids that have poison of that attack}}) from the poison shards."""
    targets: dict[tuple[str, str], set[str]] = {}
    qids: dict[str, set[str]] = {}
    for shard in sorted(Path(cfg.path("poison_dir")).glob("poison_*.jsonl")):
        for rec in iter_jsonl(shard, PoisonRecord):
            targets.setdefault((rec.qid, rec.attack), set()).add(rec.target_answer)
            qids.setdefault(rec.attack, set()).add(rec.qid)
    return {k: sorted(v) for k, v in targets.items()}, qids


# ---------------------------------------------------------------- pure record builders (unit tested)

def retrieval_part(s: Setting, q: QuestionRecord, pool: Sequence[str], flagged: Collection[str],
                   pool_scores: Sequence[float] | None, filter_ms: float, final_k: int = 5) -> dict[str, Any]:
    """Everything known before the LLM answers: pool, flagged, top-5 and the retrieval metrics."""
    flagged = set(flagged)
    survivors = [d for d in pool if d not in flagged]
    top5 = survivors[:final_k]
    gold = q.gold_doc_ids
    n_poison_pool = sum(is_poison_id(d) for d in pool)
    n_poison_flagged = sum(is_poison_id(d) for d in pool if d in flagged)
    return {
        "qid": q.qid, "setting": s.sid, "retriever": s.retriever, "attack": s.attack, "rate": s.rate,
        "filter": s.filter, "of": s.of, "param": dict(s.param), "fold": q.fold, "role": q.role,
        "pool": list(pool), "pool_scores": None if pool_scores is None else [round(float(x), 5) for x in pool_scores],
        "flagged": [d for d in pool if d in flagged], "top5": top5,
        "n_pool": len(pool), "n_poison_pool": n_poison_pool, "n_poison_flagged": n_poison_flagged,
        "n_clean_flagged": len(flagged & set(pool)) - n_poison_flagged,
        "n_poison_top5": sum(is_poison_id(d) for d in top5),
        "n_gold_dropped": metrics.gold_dropped(pool, flagged, gold),
        "backfill": metrics.backfill(pool, flagged, final_k),
        "filter_ms": round(float(filter_ms), 3),
        "ndcg5": metrics.ndcg_at_k(survivors, gold, 5), "ndcg10": metrics.ndcg_at_k(survivors, gold, 10),
        "recall5": metrics.recall_at_k(survivors, gold, 5), "recall10": metrics.recall_at_k(survivors, gold, 10),
        "ndcg5_pool": metrics.ndcg_at_k(pool, gold, 5), "ndcg10_pool": metrics.ndcg_at_k(pool, gold, 10),
    }


def answer_part(rec: dict[str, Any], q: QuestionRecord, answer: str, targets: Collection[str]) -> dict[str, Any]:
    """Add the answer and its metrics. asr is null for clean questions and for the no-poison setting."""
    asr = None if rec["attack"] is None or q.role != "target" else metrics.asr_any(answer, targets)
    rec.update(answer=answer, asr=asr, em=metrics.exact_match(answer, q.answers),
               f1=round(metrics.f1_score(answer, q.answers), 5), acc=metrics.contains_gold(answer, q.answers))
    return rec


def read_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- the runner

class SweepRunner:
    """Loads the heavy objects lazily (caches, texts, vectors, filters, LLM) and runs settings one at a time."""

    def __init__(self, cfg: Any, out_dir: Path, questions_per_role: int | None = None,
                 batch_size: int | None = None, chunk: int = 512) -> None:
        self.cfg, self.out_dir, self.batch_size, self.chunk = cfg, Path(out_dir), batch_size, chunk
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.seed, self.pool_size, self.final_k = int(cfg["seed"]), int(cfg["pool_size"]), int(cfg["final_k"])
        qs = read_jsonl(cfg.path("questions"), QuestionRecord)
        if questions_per_role:
            rng = random.Random(f"{self.seed}:subset")
            keep: list[QuestionRecord] = []
            for role in ("target", "clean"):
                pick = sorted((q for q in qs if q.role == role), key=lambda q: q.qid)
                keep += rng.sample(pick, min(questions_per_role, len(pick)))
            qs = keep
        self.questions = qs
        self.targets, self.attack_qids = load_targets(cfg)
        self._ready = False
        self._memo: dict[tuple, tuple[frozenset[str], list[float], float]] = {}
        self.stats = {"settings_run": 0, "settings_skipped": 0, "records": 0, "generated": 0, "cache_hits": 0}

    # ---- heavy objects
    def load(self) -> None:
        if self._ready:
            return
        from ire.data.textstore import TextStore
        from ire.defenses import cluster, perplexity, query_phrase
        from ire.generation.llm import Generator
        from ire.retrieval.cache import load_cache
        from ire.vectors import default_vectors

        cfg = self.cfg
        self.cache = load_cache(["bm25", "dense", "splade"], cfg=cfg)   # raises StaleCacheError if out of date
        self.texts = TextStore.open_for(cfg)
        self.vectors = default_vectors()
        self.mods = {"query_phrase": query_phrase, "cluster": cluster, "perplexity": perplexity}
        self.qp_table = query_phrase.load_thresholds(cfg.thresholds_path("query_phrase_thresholds"), cfg)
        self.ppl_table = perplexity.load_thresholds(cfg.thresholds_path("perplexity_thresholds"))
        self.ppl_scores = perplexity.default_scores(cfg)
        self.generator = Generator.from_config(cfg, texts=self.texts)
        self._ready = True

    # ---- helpers
    def path(self, s: Setting) -> Path:
        return self.out_dir / f"{s.sid}.jsonl"

    def questions_for(self, s: Setting) -> list[QuestionRecord]:
        """Attacked target questions (those with poison of this attack) + every clean question."""
        keep = self.attack_qids.get(s.attack) if s.attack is not None else None
        return [q for q in self.questions if q.role == "clean" or keep is None or q.qid in keep]

    def _detect(self, s: Setting, q: QuestionRecord, pool: list[str]) -> tuple[frozenset[str], list[float], float]:
        key = (s.retriever, s.filter, s.param, q.qid, q.fold, hashlib.sha1("|".join(pool).encode()).hexdigest())
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        from ire.schemas import FilterContext

        cfg, p = self.cfg, dict(s.param)
        if s.filter == "query_phrase":
            tail = float(p.get("tail", cfg["query_phrase.anchor_tail"]))
            th = self.mods["query_phrase"].thresholds_for(self.qp_table, s.retriever, q.fold, tail=tail)
        elif s.filter == "cluster":
            th = {"tau_c": float(p.get("tau_c", cfg["cluster.tau_c"]))}
        else:
            th = {"alpha": float(p.get("alpha", cfg["perplexity.alpha"])), "table": self.ppl_table,
                  "scores": self.ppl_scores}
        ctx = FilterContext(texts=self.texts, vectors=self.vectors, config=cfg, thresholds=th)
        t = time.perf_counter()
        res = self.mods[s.filter].apply_filter(pool, q.qid, q.question, ctx)
        ms = 1000 * (time.perf_counter() - t)
        out = (frozenset(res.flagged), [float(res.scores.get(d, 0.0)) for d in pool], ms)
        if len(self._memo) > 400_000:
            self._memo.clear()
        self._memo[key] = out
        return out

    def run_setting(self, s: Setting, log: Any = print) -> Path:
        out = self.path(s)
        if out.exists():
            self.stats["settings_skipped"] += 1
            return out
        det_counts: dict[str, int] = {}
        if s.filter == "random":
            det = s.detector
            assert det is not None
            det_counts = {r["qid"]: len(r["flagged"]) for r in read_records(self.run_setting(det, log))}
        self.load()
        t0 = time.time()
        qs = self.questions_for(s)
        recs: list[dict[str, Any]] = []
        for q in qs:
            pool = candidate_pool(q.qid, s.retriever, s.attack, s.rate, self.cache, self.pool_size, self.cfg)
            scores: list[float] | None = None
            ms = 0.0
            if s.filter == "none":
                flagged: Collection[str] = frozenset()
            elif s.filter == "oracle":
                flagged = oracle_filter(pool)
            elif s.filter == "random":
                flagged = random_filter(pool, det_counts[q.qid], self.seed, f"{s.sid}:{q.qid}")
            else:
                flagged, scores, ms = self._detect(s, q, pool)
            recs.append(retrieval_part(s, q, pool, flagged, scores, ms, self.final_k))
        by_qid = {q.qid: q for q in qs}
        for i in range(0, len(recs), self.chunk):
            part = recs[i:i + self.chunk]
            outs = self.generator.generate_batch([(r["qid"], by_qid[r["qid"]].question, r["top5"]) for r in part],
                                                 batch_size=self.batch_size)
            for r, o in zip(part, outs):
                answer_part(r, by_qid[r["qid"]], o["answer"], self.targets.get((r["qid"], s.attack), []))
                self.stats["cache_hits" if o["cached"] else "generated"] += 1
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, out)
        self.stats["settings_run"] += 1
        self.stats["records"] += len(recs)
        log(f"  done {s.sid}: {len(recs)} questions in {time.time() - t0:.0f}s "
            f"(generated {self.stats['generated']}, cached {self.stats['cache_hits']} so far)")
        return out

    def run_all(self, settings: Iterable[Setting], max_settings: int | None = None, log: Any = print) -> list[str]:
        """Run settings in order; a failing setting is logged and skipped. Returns the failed setting ids."""
        failed: list[str] = []
        todo = [s for s in settings]
        for n, s in enumerate(todo, start=1):
            if max_settings is not None and self.stats["settings_run"] >= max_settings:
                break
            log(f"[{n}/{len(todo)}] {s.sid}")
            try:
                self.run_setting(s, log)
            except Exception as exc:  # noqa: BLE001 - recorded; the sweep continues with the next setting
                traceback.print_exc()
                failed.append(f"{s.sid}: {type(exc).__name__}: {exc}")
        return failed
