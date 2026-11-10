"""Sweep grid expansion and the pure run-record builders (src/ire/eval/sweep.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ire.eval.sweep import Setting, answer_part, expand, load_sweep, retrieval_part
from ire.schemas import QuestionRecord

REPO = Path(__file__).resolve().parents[1]
SWEEP = {
    "retrievers": ["bm25", "dense"], "tiers": ["headline", "rates"],
    "blocks": [
        {"tier": "headline", "filter": "none", "attacks": [None], "rates": [None]},
        {"tier": "headline", "filter": "query_phrase", "params": [{"tail": 0.025}], "attacks": ["fact"], "rates": [3]},
        {"tier": "headline", "filter": "random", "of": "query_phrase", "params": [{"tail": 0.025}],
         "attacks": ["fact"], "rates": [3]},
        {"tier": "rates", "filter": "none", "attacks": ["fact"], "rates": [1, 5]},
    ],
}


def test_expand_ids_and_random_detector():
    ss = expand(SWEEP, ["headline"])
    sids = [s.sid for s in ss]
    assert len(ss) == 6 and len(set(sids)) == 6
    assert "bm25__none__r0__none" in sids
    assert "dense__fact__r3__query_phrase__tail0.025" in sids
    rnd = next(s for s in ss if s.filter == "random" and s.retriever == "bm25")
    assert rnd.sid == "bm25__fact__r3__random__of-query_phrase__tail0.025"
    assert rnd.detector.sid == "bm25__fact__r3__query_phrase__tail0.025"
    assert len(expand(SWEEP, ["headline", "rates"])) == 10
    assert len(expand(SWEEP, ["headline"], attacks=["hijack"])) == 2      # only the no-poison settings remain


def test_expand_rejects_bad_blocks():
    bad = {**SWEEP, "blocks": [{"tier": "headline", "filter": "random", "attacks": ["fact"], "rates": [3]}]}
    with pytest.raises(ValueError):
        expand(bad, ["headline"])
    with pytest.raises(ValueError):
        expand(SWEEP, ["nope"])


def test_real_sweep_yaml_counts():
    sweep = load_sweep(REPO / "configs" / "sweep.yaml")
    assert len(expand(sweep, ["headline"])) == 68
    assert len(expand(sweep, sweep["tiers"])) == 384


Q = QuestionRecord(qid="q1", question="who?", answers=["Paris"], gold_doc_ids=["g1"], role="target", fold="A")
POOL = ["poison_q1_fact_1", "g1", "d2", "d3", "d4", "d5", "d6"]


def test_retrieval_part_oracle():
    s = Setting("bm25", "fact", 3, "oracle")
    rec = retrieval_part(s, Q, POOL, {"poison_q1_fact_1"}, None, 0.0)
    assert rec["top5"] == ["g1", "d2", "d3", "d4", "d5"]
    assert (rec["n_poison_pool"], rec["n_poison_flagged"], rec["n_clean_flagged"], rec["n_poison_top5"]) == (1, 1, 0, 0)
    assert rec["ndcg5"] == pytest.approx(1.0) and rec["ndcg5_pool"] < 1.0
    assert rec["flagged"] == ["poison_q1_fact_1"] and rec["pool_scores"] is None and rec["backfill"] == 0


def test_answer_part_asr_rules():
    s = Setting("bm25", "fact", 3, "none")
    rec = answer_part(retrieval_part(s, Q, POOL, set(), None, 0.0), Q, "It is Paris, ZOLTRAX", ["ZOLTRAX"])
    assert rec["asr"] == 1 and rec["acc"] == 1 and rec["em"] == 0 and rec["f1"] > 0
    assert rec["n_poison_top5"] == 1
    clean = QuestionRecord(qid="q9", question="x", answers=["y"], gold_doc_ids=["g"], role="clean", fold="B")
    assert answer_part(retrieval_part(s, clean, POOL, set(), None, 0.0), clean, "y", ["ZOLTRAX"])["asr"] is None
    nop = Setting("bm25", None, None, "none")
    assert answer_part(retrieval_part(nop, Q, POOL, set(), None, 0.0), Q, "Paris", [])["asr"] is None


def test_heldout_targets_are_the_hijack_questions_only():
    from ire.eval.sweep import collect_targets
    from ire.schemas import PoisonRecord

    def poison(qid, attack, n, target):
        return PoisonRecord(doc_id=f"poison_{qid}_{attack}_{n}", qid=qid, attack=attack, title="t", text="x",
                            is_poison=True, target_answer=target)

    recs = [poison("q1", "heldout", 1, "ZOLTRAX"), poison("q1", "heldout", 2, "ZOLTRAX"),
            poison("q2", "heldout", 1, "ZOLTRAX"), poison("q9", "heldout", 1, "1992"),   # a held-out fact passage
            poison("q9", "fact", 1, "1992"), poison("q1", "fact", 1, "1987")]
    targets, qids = collect_targets(recs, "ZOLTRAX")
    assert qids["heldout"] == {"q1", "q2"} and ("q9", "heldout") not in targets
    assert targets[("q1", "heldout")] == ["ZOLTRAX"]
    assert qids["fact"] == {"q1", "q9"} and targets[("q9", "fact")] == ["1992"]   # other attacks unchanged
