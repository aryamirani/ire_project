"""Analysis: summaries, paired comparison, headline family with Holm, cross-fitting (src/ire/eval/analysis.py)."""

from __future__ import annotations

import math

import pytest

from ire.eval import analysis as an

CFG = {"query_phrase.anchor_tail": 0.025, "cluster.tau_c": 0.85, "perplexity.alpha": 0.025}
QP = (("tail", 0.025),)


def rec(qid, role, fold, asr, ndcg5=0.5, n_poison_top5=0, filt="none", of=None, param=(), retriever="bm25",
        attack="fact", n_poison_pool=1, n_poison_flagged=0):
    return {"qid": qid, "role": role, "fold": fold, "asr": asr, "em": 0, "f1": 0.0, "acc": 1, "ndcg5": ndcg5,
            "ndcg10": ndcg5, "recall10": 1.0, "ndcg5_pool": ndcg5, "n_poison_top5": n_poison_top5,
            "n_poison_pool": n_poison_pool, "n_gold_dropped": 0, "backfill": 0, "filter_ms": 1.0, "n_pool": 50,
            "n_poison_flagged": n_poison_flagged, "n_clean_flagged": 0, "pool": [], "pool_scores": None,
            "retriever": retriever, "attack": attack, "rate": 3, "filter": filt, "of": of, "param": dict(param)}


def setting(asr_target, ndcg_clean, filt="none", param=(), n=40, flagged=0):
    rows = {f"t{i}": rec(f"t{i}", "target", "A" if i % 2 == 0 else "B", asr_target, filt=filt, param=param,
                         n_poison_flagged=flagged) for i in range(n)}
    rows.update({f"c{i}": rec(f"c{i}", "clean", "A" if i % 2 == 0 else "B", None, ndcg_clean, filt=filt, param=param,
                              n_poison_pool=0) for i in range(4)})
    return rows


def test_summarize_defense_and_detector_view():
    rows = {f"t{i}": rec(f"t{i}", "target", "A", int(i < 2), n_poison_flagged=int(i < 2)) for i in range(4)}
    rows["c0"] = rec("c0", "clean", "A", None, n_poison_top5=0, n_poison_pool=0)
    rows["c1"] = rec("c1", "clean", "A", None, n_poison_top5=1, n_poison_pool=0)
    s = an.summarize(rows, auc_boot=0)
    assert s["asr"] == pytest.approx(0.5) and s["n_target"] == 4 and s["n_clean"] == 2
    assert s["collateral_rate"] == pytest.approx(0.5)
    assert s["tpr"] == pytest.approx(0.5) and s["fpr"] == 0.0
    assert math.isnan(s["auc"])


def test_paired_difference():
    ref = {f"t{i}": rec(f"t{i}", "target", "A", int(i < 2)) for i in range(4)}
    oth = {f"t{i}": rec(f"t{i}", "target", "A", int(i == 1)) for i in range(4)}
    for d, nd in ((ref, 0.5), (oth, 0.4)):
        d.update({f"c{i}": rec(f"c{i}", "clean", "A", None, nd) for i in range(2)})
    p = an.paired(ref, oth, n_boot=100)
    assert p["asr_diff"] == pytest.approx(-0.25) and p["p_mcnemar"] == 1.0
    assert p["d_c_ndcg5"] == pytest.approx(-0.1)


def test_headline_family_helps():
    runs = {("bm25", "fact", 3, "none", None, ()): setting(1, 0.5),
            ("bm25", "fact", 3, "query_phrase", None, QP): setting(0, 0.5, "query_phrase", QP)}
    rows = an.headline(runs, CFG, ["bm25"])
    assert len(rows) == 1
    r = rows[0]
    assert r["detector"] == "query_phrase" and r["asr_diff"] == -1.0 and r["p_holm"] < 0.001
    assert r["helps"] is True and r["beats_random"] is False


def test_headline_not_helpful_when_clean_quality_drops():
    runs = {("bm25", "fact", 3, "none", None, ()): setting(1, 0.5),
            ("bm25", "fact", 3, "query_phrase", None, QP): setting(0, 0.3, "query_phrase", QP)}
    assert an.headline(runs, CFG, ["bm25"])[0]["helps"] is False


def test_cross_fit_prefers_param_without_clean_loss():
    p1, p2 = (("tau_c", 0.9),), (("tau_c", 0.7),)
    base = setting(1, 0.5)
    by_param = {p1: setting(0, 0.5, "cluster", p1), p2: setting(0, 0.0, "cluster", p2)}
    res = an.cross_fit(base, by_param, anchor=p2, bound=0.02)
    assert res["chosen_on_A_reported_on_B"] == p1 and res["chosen_on_B_reported_on_A"] == p1
    assert res["asr_none"] == 1.0 and res["asr_filtered"] == 0.0 and res["d_c_ndcg5"] == pytest.approx(0.0)
