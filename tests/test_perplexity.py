"""Step 11: perplexity filter, score cache and thresholds (src/ire/defenses/perplexity.py)."""

from __future__ import annotations

import numpy as np
import pytest

import ire.config  # noqa: F401  (loads .env, so HF_HOME points at the project's model cache)
from ire.defenses.perplexity import (
    PerplexityCache,
    PerplexityScorer,
    PerplexityScores,
    alpha_key,
    apply_filter,
    calibrate,
    explain,
    flags,
)
from ire.schemas import FilterContext, FilterResult


def rec(doc_id, pd, pm, too_short=False):
    return {"doc_id": doc_id, "pd": pd, "pm": pm, "too_short": too_short}


class FakeScorer:
    """Returns pre-set records; counts how many passages it was asked to score."""

    def __init__(self, table):
        self.table, self.n_scored = table, 0

    def namespace(self):
        return "ns-test"

    def score_texts(self, items):
        self.n_scored += len(items)
        return [dict(self.table[d]) for d, _ in items]


def test_calibrate_quantiles_and_flags():
    rng = np.random.default_rng(0)
    recs = [rec(f"d{i}", float(x), float(y)) for i, (x, y) in enumerate(zip(rng.normal(0, 1, 4000),
                                                                             rng.normal(4, 0.5, 4000)))]
    recs.append(rec("short", None, None, too_short=True))
    th = calibrate(recs, [0.025, 0.1])
    a = th[alpha_key(0.025)]
    pd = np.array([r["pd"] for r in recs[:-1]])
    pm = np.array([r["pm"] for r in recs[:-1]])
    assert a["pd_low"] == pytest.approx(np.quantile(pd, 0.0125)) and a["pd_high"] == pytest.approx(np.quantile(pd, 0.9875))
    assert a["pm_high"] == pytest.approx(np.quantile(pm, 0.975))
    assert np.mean(pd < a["pd_low"]) == pytest.approx(0.0125, abs=0.003)
    assert th["0.1"]["pm_high"] < a["pm_high"]
    rate = np.mean([flags(r, a, "either")["flagged"] for r in recs[:-1]])
    assert 0.03 < rate < 0.07                      # PD tails 2.5% + PM tail 2.5%, nearly independent here
    assert flags(rec("x", 5.0, 0.0), a, "pd") == {"pd": True, "pm": False, "flagged": True}
    assert flags(rec("x", 5.0, 0.0), a, "pm")["flagged"] is False
    assert flags(rec("x", 0.0, 9.0), a, "either")["flagged"] is True
    assert flags(rec("s", None, None, True), a, "either")["flagged"] is False   # too_short never flagged
    with pytest.raises(ValueError):
        flags(rec("x", 0.0, 0.0), a, "bogus")


def test_cache_round_trip_and_only_misses_scored(tmp_path):
    cache = PerplexityCache(tmp_path / "ppl.sqlite")
    table = {"a": rec("a", 0.1, 3.0), "b": rec("b", -2.5, 4.0), "c": rec("c", None, None, True)}
    scorer = FakeScorer(table)
    scores = PerplexityScores(scorer, cache)
    texts = {d: d for d in table}
    first = scores.get(["a", "b", "c"], texts)
    assert scorer.n_scored == 3 and first["b"]["pd"] == -2.5
    again = scores.get(["c", "a", "b", "a"], texts)
    assert scorer.n_scored == 3 and again == first          # cache hit: no scoring work
    assert cache.count("ns-test") == 3 and cache.count("other") == 0


def test_apply_filter_with_fixed_scores(tmp_path):
    table = {"clean": rec("clean", 0.0, 3.5), "break": rec("break", -3.0, 6.0), "loopy": rec("loopy", 0.1, 1.0),
             "short": rec("short", None, None, True)}
    th_table = {"thresholds": {"0.025": {"pd_low": -1.0, "pd_high": 1.0, "pm_high": 5.0}}}
    scores = PerplexityScores(FakeScorer(table), PerplexityCache(tmp_path / "p.sqlite"))
    ctx = FilterContext(texts={d: d for d in table}, vectors=None, config={"perplexity.rule": "either"},
                        thresholds={"alpha": 0.025, "table": th_table, "scores": scores})
    pool = ["clean", "break", "loopy", "short"]
    res = apply_filter(pool, "q", "x", ctx)
    assert isinstance(res, FilterResult)
    assert res.flagged == {"break"} and res.scores["short"] == 0.0 and res.scores["break"] == 6.0
    ctx.config["perplexity.rule"] = "pm"
    assert apply_filter(pool, "q", "x", ctx).flagged == {"break"}
    th_table["thresholds"]["0.025"]["pm_high"] = 0.5
    assert apply_filter(pool, "q", "x", ctx).flagged == {"clean", "break", "loopy"}  # never the too_short one
    detail = explain(pool, "x", ctx)
    assert [d["rank"] for d in detail] == [1, 2, 3, 4] and detail[3]["flagged"] is False


def _gpt2_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download

        from ire.retrieval.common import MODEL_SKIP_PATTERNS

        snapshot_download("openai-community/gpt2", local_files_only=True, ignore_patterns=MODEL_SKIP_PATTERNS)
        return True
    except Exception:  # noqa: BLE001
        return False


needs_gpt2 = pytest.mark.skipif(not _gpt2_cached(), reason="openai-community/gpt2 not in the local HF cache")

FLUENT = ("The Royal Mint is a government-owned mint that produces coins for the United Kingdom. It moved most of "
          "its production from London to a new site in Llantrisant, Wales, in the late 1960s and early 1970s, "
          "where it still operates today and exports coins to many countries around the world.")


@needs_gpt2
def test_gpt2_gibberish_second_half_and_short_text():
    scorer = PerplexityScorer("openai-community/gpt2", device="cpu", min_tokens=20)
    ids = scorer.tokenizer()(FLUENT, add_special_tokens=False)["input_ids"]
    half = scorer.tokenizer().decode(ids[: len(ids) // 2])
    gibberish = half + " zqx vlorp quantabrix mmfl ztt orpy glarnex wuv sploink qqar tib vexo prunk klaz dorvim"
    fluent, broken, short = scorer.score_texts([("f", FLUENT), ("g", gibberish), ("s", "Paris is in France.")])
    assert not fluent["too_short"] and short["too_short"] and short["pd"] is None
    assert broken["nll_post"] > fluent["nll_post"] + 2.0          # the gibberish half is far less fluent
    assert broken["pd"] < fluent["pd"] - 2.0                       # PD drops sharply (expected direction)
    assert broken["pm"] > fluent["pm"]
    assert broken["ppl_post"] == pytest.approx(np.exp(broken["nll_post"]))
    again = scorer.score_texts([("f", FLUENT)])[0]                 # deterministic
    assert again["pd"] == pytest.approx(fluent["pd"], abs=1e-5)


def test_batches_bounded_by_padded_tokens():
    sc = PerplexityScorer("unused", batch_tokens=100)
    lengths = [10, 10, 10, 40, 40, 99, 300]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches = sc._batches(order, lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:   # every batch fits, except a single sequence that is longer than the budget on its own
        assert len(b) == 1 or len(b) * (max(lengths[i] for i in b) + 1) <= 100


@needs_gpt2
def test_cache_hits_load_no_model(tmp_path):
    """Step 11 verify: a lookup whose doc_ids are all cached does no model work at all (no weights loaded)."""
    from ire.defenses.perplexity import PerplexityCache, PerplexityScorer, PerplexityScores

    scorer = PerplexityScorer("openai-community/gpt2", device="cpu")
    cache = PerplexityCache(tmp_path / "ppl.sqlite")
    ns = scorer.namespace()
    assert scorer._model is None and scorer.revision                      # namespace needs the revision only
    rec = {"doc_id": "d1", "ppl_pre": 10.0, "ppl_post": 12.0, "pd": -0.1, "pm": 2.5, "nll_pre": 2.3,
           "nll_post": 2.5, "n_tokens": 40, "too_short": False}
    cache.put_many(ns, [rec])
    got = PerplexityScores(scorer, cache).get(["d1"], {"d1": "unused text"})
    assert got["d1"]["pm"] == 2.5 and scorer._model is None and scorer.n_scored == 0


def test_filter_refuses_thresholds_from_another_scorer(tmp_path):
    """Thresholds are percentiles of one scorer's PD / PM; a table calibrated for another namespace is refused."""
    table = {"clean": rec("clean", 0.0, 3.5)}
    scores = PerplexityScores(FakeScorer(table), PerplexityCache(tmp_path / "p.sqlite"))
    th = {"namespace": "ns-other", "thresholds": {"0.025": {"pd_low": -1.0, "pd_high": 1.0, "pm_high": 5.0}}}
    ctx = FilterContext(texts={"clean": "clean"}, vectors=None, config={"perplexity.rule": "either"},
                        thresholds={"alpha": 0.025, "table": th, "scores": scores})
    with pytest.raises(ValueError, match="re-run scripts/calibrate_perplexity.py"):
        apply_filter(["clean"], "q", "x", ctx)
    th["namespace"] = "ns-test"                                           # FakeScorer's namespace
    assert apply_filter(["clean"], "q", "x", ctx).flagged == set()
