"""Statistics (plan 5): Wilson, McNemar, paired bootstrap, Holm, ROC-AUC, kappa."""

from __future__ import annotations

import math

import pytest

from ire.eval import stats as s


def test_wilson():
    assert math.isnan(s.wilson(0, 0)[0])
    p, lo, hi = s.wilson(5, 10)
    assert p == 0.5 and lo < 0.5 < hi


def test_mcnemar_exact():
    assert s.mcnemar_exact(0, 0) == 1.0
    assert s.mcnemar_exact(5, 5) == 1.0
    assert s.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    r = s.paired_binary([1, 1, 0, 0], [0, 1, 0, 0])
    assert r["diff"] == -0.25 and r["only_a"] == 1 and r["only_b"] == 0 and r["p_mcnemar"] == 1.0


def test_holm():
    adj = s.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}


def test_paired_bootstrap_constant():
    b = s.paired_bootstrap([-0.1] * 20, n_boot=200)
    assert b["mean"] == pytest.approx(-0.1) and b["low"] == pytest.approx(-0.1) and b["high"] == pytest.approx(-0.1)
    assert math.isnan(s.paired_bootstrap([])["mean"])


def test_roc_auc():
    assert s.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert s.roc_auc([1, 0], [0.5, 0.5]) == 0.5
    assert math.isnan(s.roc_auc([1, 1], [0.1, 0.2]))
    g = [([0, 1], [0.1, 0.9]), ([0, 0, 1], [0.2, 0.3, 0.8])]
    b = s.bootstrap_auc(g, n_boot=50)
    assert b["auc"] == 1.0 and b["low"] == 1.0 and b["high"] == 1.0


def test_cohen_kappa():
    assert s.cohen_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert s.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
