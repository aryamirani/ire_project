"""Poison ingestion support (Step 8): shard validation and the checks around ingestion.

This module does not create poison (that is Arya's lane); it validates her `poison_*.jsonl` shards and measures
what ingesting them does to the indexes:
- `load_shards`          validate every record (PoisonRecord), reject duplicate ids and unknown qids, count.
- `masking_equivalence`  plan section 2.1: top-150 of the poisoned index with every poison_ id dropped vs the
                         clean index's top-150. Dense and SPLADE must be identical; BM25 differs slightly (IDF).
- `poison_reach`         rank of each poison passage in its own target question's (unmasked) results.
- `top_k_changed`        fraction of questions whose unmasked top-k differs from the clean top-k.
- clean reference        the clean index's top-150 for every question, saved before the first ingestion so the
                         masking check still has a clean baseline after the index has changed.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ire.schemas import PoisonRecord, SchemaError, is_poison_id, iter_jsonl

Row = list[tuple[str, float]]


class PoisonShardError(ValueError):
    """The poison shards are invalid (schema, duplicate ids, unknown qids)."""


@dataclass
class ShardSet:
    shards: list[Path]
    records: list[PoisonRecord] = field(default_factory=list)

    @property
    def by_attack(self) -> dict[str, int]:
        return dict(sorted(Counter(r.attack for r in self.records).items()))

    @property
    def by_qid(self) -> dict[str, int]:
        return dict(sorted(Counter(r.qid for r in self.records).items()))

    def summary(self, roles: Mapping[str, str] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"files": [s.name for s in self.shards], "n_records": len(self.records),
                               "by_attack": self.by_attack, "n_target_qids": len(self.by_qid),
                               "per_qid": {"min": min(self.by_qid.values(), default=0),
                                           "max": max(self.by_qid.values(), default=0)}}
        if roles is not None:
            out["qids_with_role_clean"] = sorted(q for q in self.by_qid if roles.get(q) == "clean")
        return out


def load_shards(shards: Sequence[Path], question_roles: Mapping[str, str],
                corpus_ids: set[str] | None = None) -> ShardSet:
    """Read and validate every shard. Raises PoisonShardError listing what is wrong.

    question_roles: qid -> role from questions.jsonl; every poison qid must be in it.
    corpus_ids: clean doc_ids; a poison id equal to one of them is rejected (PoisonRecord ids start with
    "poison_" and CorpusRecord ids never do, so this is a second guard).
    """
    result = ShardSet(shards=list(shards))
    problems: list[str] = []
    seen: dict[str, str] = {}
    for shard in shards:
        try:
            records = list(iter_jsonl(shard, PoisonRecord))
        except SchemaError as exc:
            problems.append(str(exc))
            continue
        for rec in records:
            if rec.doc_id in seen:
                problems.append(f"{shard.name}: doc_id {rec.doc_id} already in {seen[rec.doc_id]}")
            elif corpus_ids is not None and rec.doc_id in corpus_ids:
                problems.append(f"{shard.name}: doc_id {rec.doc_id} is also a corpus doc_id")
            seen.setdefault(rec.doc_id, shard.name)
            if rec.qid not in question_roles:
                problems.append(f"{shard.name}: {rec.doc_id} targets qid {rec.qid}, which is not in questions.jsonl")
            result.records.append(rec)
    if problems:
        raise PoisonShardError(f"{len(problems)} problem(s) in the poison shards, e.g.:\n  " + "\n  ".join(problems[:10]))
    return result


# ---------------------------------------------------------------- checks


def masking_equivalence(poisoned: Mapping[str, Row], clean: Mapping[str, Row], qids: Sequence[str],
                        score_tol: float = 1e-5) -> dict[str, Any]:
    """For each qid: drop every poison_ id from the poisoned top-k and compare with the clean top-k.

    Reported: questions whose masked list equals the clean list's prefix of the same length (ids and order);
    max |score difference| on those prefixes; for clean docs in both lists the absolute rank difference
    (max / mean); fraction of questions with the same clean top-5 and top-50; Jaccard overlap at 50.
    `exact` is True when every question is identical and every score within score_tol (relative).
    """
    identical = top5_same = top50_same = 0
    max_score_diff = 0.0
    rank_diffs: list[int] = []
    jaccards: list[float] = []
    for qid in qids:
        masked = [(d, s) for d, s in poisoned[qid] if not is_poison_id(d)]
        ref = clean[qid]
        m_ids, r_ids = [d for d, _ in masked], [d for d, _ in ref]
        same = m_ids == r_ids[:len(m_ids)]
        identical += same
        if same:
            for (_, a), (_, b) in zip(masked, ref):
                max_score_diff = max(max_score_diff, abs(a - b) / max(abs(b), 1e-12))
        r_rank = {d: i for i, d in enumerate(r_ids)}
        rank_diffs += [abs(i - r_rank[d]) for i, d in enumerate(m_ids) if d in r_rank]
        top5_same += m_ids[:5] == r_ids[:5]
        top50_same += m_ids[:50] == r_ids[:50]
        a, b = set(m_ids[:50]), set(r_ids[:50])
        jaccards.append(len(a & b) / len(a | b) if a | b else 1.0)
    n = len(qids)
    return {"n_questions": n, "identical_ids_and_order": identical, "max_rel_score_diff": max_score_diff,
            "rank_diff_max": max(rank_diffs, default=0),
            "rank_diff_mean": round(statistics.mean(rank_diffs), 4) if rank_diffs else 0.0,
            "frac_top5_unchanged": round(top5_same / n, 4) if n else None,
            "frac_top50_unchanged": round(top50_same / n, 4) if n else None,
            "jaccard_at_50_mean": round(statistics.mean(jaccards), 4) if jaccards else None,
            "jaccard_at_50_min": round(min(jaccards), 4) if jaccards else None,
            "exact": identical == n and max_score_diff <= score_tol}


def poison_reach(poisoned: Mapping[str, Row], records: Sequence[PoisonRecord],
                 ks: Sequence[int] = (10, 150)) -> dict[str, Any]:
    """Rank (1-based, None if absent) of each poison passage in its own target question's unmasked results, and
    the fraction found in the top-k, by attack."""
    ranks: dict[str, int | None] = {}
    for rec in records:
        ids = [d for d, _ in poisoned.get(rec.qid, [])]
        ranks[rec.doc_id] = ids.index(rec.doc_id) + 1 if rec.doc_id in ids else None
    by_attack: dict[str, dict[str, Any]] = {}
    for attack in sorted({r.attack for r in records}):
        group = [ranks[r.doc_id] for r in records if r.attack == attack]
        by_attack[attack] = {"n": len(group), **{f"frac_in_top{k}": round(sum(1 for x in group if x and x <= k)
                                                                         / len(group), 4) for k in ks}}
    found = [r for r in ranks.values() if r]
    return {"by_attack": by_attack, "ranks": ranks,
            **{f"any_in_top{k}": any(r <= k for r in found) for k in ks}}


def top_k_changed(poisoned: Mapping[str, Row], clean: Mapping[str, Row], qids: Sequence[str], k: int = 5) -> float:
    """Fraction of qids whose unmasked top-k ids (with poison) differ from the clean top-k ids."""
    if not qids:
        return 0.0
    changed = sum(1 for q in qids if [d for d, _ in poisoned[q][:k]] != [d for d, _ in clean[q][:k]])
    return round(changed / len(qids), 4)


# ---------------------------------------------------------------- clean reference (saved before ingestion)


def reference_path(index_dir: Path) -> Path:
    """`<indexes_dir>/<env>/masking_reference_<retriever>.json`, next to (not inside) the index folder, so a
    rebuild or ingestion of the index never deletes it."""
    index_dir = Path(index_dir)
    return index_dir.parent / f"masking_reference_{index_dir.name}.json"


def save_reference(index_dir: Path, clean_fingerprint: dict[str, Any], questions_sha256: str,
                   rows: Mapping[str, Row]) -> Path:
    path = reference_path(index_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"clean_fingerprint": clean_fingerprint, "questions_sha256": questions_sha256,
                               "rows": {q: [[d, s] for d, s in r] for q, r in rows.items()}}), encoding="utf-8")
    tmp.replace(path)
    return path


def load_reference(index_dir: Path, clean_fingerprint: dict[str, Any], questions_sha256: str
                   ) -> dict[str, Row] | None:
    """The saved clean top-150 rows if they belong to this clean index and question set, else None."""
    path = reference_path(index_dir)
    if not path.is_file():
        return None
    ref = json.loads(path.read_text(encoding="utf-8"))
    if ref.get("clean_fingerprint") != clean_fingerprint or ref.get("questions_sha256") != questions_sha256:
        return None
    return {q: [(d, float(s)) for d, s in r] for q, r in ref["rows"].items()}
