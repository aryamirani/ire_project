"""Step 1: Ada report acceptance checks (A0 rules on check_env.py fields; A1-A9 are stubs)."""

import copy

import pytest

from ire.ada_report import JOBS, STUBS, check_report
from ire.config import load_env_profile


def a0_report(env: str = "ada") -> dict:
    """A report with the structure scripts/check_env.py writes (synthetic, for testing the rules only)."""
    models = load_env_profile(env)["models"].values()
    return {
        "job": "A0", "env": env, "ok": True, "timestamp_utc": "2026-10-15T10:00:00+00:00",
        "git_commit": "a49ead8", "python": "3.13.7", "platform": "Linux-x86_64",
        "torch": {"importable": True, "version": "2.6.0+cu124", "cuda_available": True, "cuda_version": "12.4",
                  "gpus": [{"index": 0, "name": "Test GPU", "total_gib": 24.0, "free_gib": 23.5}]},
        "disk": {"path": "/data/ire", "free_gib": 500.0, "total_gib": 1000.0, "writable": True},
        "libraries": {"numpy": {"importable": True, "required": True, "version": "2.5.4"},
                      "pyserini": {"importable": False, "required": False, "error": "ImportError"}},
        "java": {"found": False, "required": False},
        "models_cached": {m: True for m in models},
        "hf_token_set": False, "runtime_s": 3.2, "rerun": "python scripts/check_env.py --report A0",
    }


def test_a0_passes_on_good_ada_report() -> None:
    result = check_report("A0", a0_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(env="local"),
    lambda r: r.update(job="A1"),
    lambda r: r.update(git_commit="unknown"),
    lambda r: r["torch"].update(cuda_available=False, gpus=[]),
    lambda r: r["libraries"]["numpy"].update(importable=False),
    lambda r: r["disk"].update(writable=False),
    lambda r: r["models_cached"].update({"Qwen/Qwen2.5-3B-Instruct": False}),
    lambda r: r.pop("models_cached"),
    lambda r: r.pop("timestamp_utc"),
], ids=["not-ok", "env-local", "wrong-job", "no-commit", "no-gpu", "missing-lib", "disk-readonly",
        "model-missing", "field-missing", "common-field-missing"])
def test_a0_fails(breaker) -> None:
    report = copy.deepcopy(a0_report())
    breaker(report)
    assert check_report("A0", report).status == "FAIL"


def test_warnings_do_not_fail() -> None:
    report = a0_report()
    report["python"] = "3.12.4"
    report["git_commit"] = "a49ead8-dirty"
    result = check_report("A0", report)
    assert result.status == "PASS"
    assert sum(not c.passed for c in result.checks) == 2


def test_local_report_is_only_a_dry_run() -> None:
    assert check_report("A0", a0_report("local")).status == "FAIL"
    result = check_report("A0", a0_report("local"), allow_local=True)
    assert result.status == "LOCAL_DRY_RUN"
    local = dict(a0_report("local"), job="A0_local")  # check_env.py --report A0_local
    assert check_report("A0", local, allow_local=True).status == "LOCAL_DRY_RUN"
    assert check_report("A0", dict(a0_report(), job="A0_local")).status == "FAIL"  # suffix only in dry runs


@pytest.mark.parametrize("job", [j for j in JOBS if j not in ("A0", "A1", "A2", "A3", "A4", "A5", "A6")])
def test_other_jobs_are_stubs(job: str) -> None:
    result = check_report(job, {"job": job, "env": "ada"})
    assert result.status == "NOT_IMPLEMENTED"
    assert f"step {STUBS[job][0]}" in result.note


def test_unknown_job_rejected() -> None:
    with pytest.raises(ValueError, match="unknown job"):
        check_report("A10", {})


def a1_report(mode: str = "full") -> dict:
    """A report with the structure scripts/prepare_data.py writes (synthetic, for testing the rules only)."""
    n = 2_681_468 if mode == "full" else 5000
    return {
        "job": "A1", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-16T10:00:00+00:00",
        "corpus_mode": mode, "dev_subset": False, "n_passages_requested": None if mode == "full" else 5000,
        "counts": {"raw_corpus_passages": 2_681_468, "raw_duplicate_doc_ids": 0, "corpus_passages": n,
                   "corpus_duplicate_doc_ids": 0, "question_gold_missing_in_corpus": 0,
                   "questions_all_gold_missing_in_corpus": 0 if mode == "full" else 3000,
                   "role_fold": {"clean/A": 100, "clean/B": 100, "target/A": 100, "target/B": 100}},
        "checks": {"no_duplicate_qids": True}, "ok": True,
        "files": {"corpus": {"rows": n}}, "sources": {},
    }


@pytest.mark.parametrize("mode", ["full", "random"])
def test_a1_passes(mode: str) -> None:
    result = check_report("A1", a1_report(mode))
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r["counts"].update(corpus_passages=500_000),                    # full mode but not all passages
    lambda r: r["files"]["corpus"].update(rows=10),
    lambda r: r["counts"].update(role_fold={"clean/A": 100, "clean/B": 100, "target/A": 99, "target/B": 101}),
    lambda r: r["counts"].update(question_gold_missing_in_corpus=1),
    lambda r: r["counts"].update(questions_all_gold_missing_in_corpus=2),     # full mode: pool gold must exist
    lambda r: r["counts"].update(corpus_duplicate_doc_ids=1),
    lambda r: r["checks"].update(no_duplicate_qids=False),
    lambda r: r.pop("counts"),
], ids=["not-ok", "size", "rows", "fold-balance", "gold-missing", "pool-gold-missing", "dup-docs", "dup-qids",
        "field-missing"])
def test_a1_fails(breaker) -> None:
    report = copy.deepcopy(a1_report("full"))
    breaker(report)
    assert check_report("A1", report).status == "FAIL"


def test_a1_random_mode_size_must_match_request() -> None:
    report = a1_report("random")
    report["n_passages_requested"] = 6000
    assert check_report("A1", report).status == "FAIL"


def a2_report() -> dict:
    """A report with the structure scripts/build_index.py writes (synthetic, for testing the rules only)."""
    return {
        "job": "A2", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-16T10:00:00+00:00",
        "retriever": "bm25", "ok": True, "checks": {"unique_token_rank_1": True},
        "corpus": {"rows": 2_681_468}, "n_docs": 2_681_468, "n_poison": 0, "build_time_s": 900.0,
        "index_bytes": 2 * 2**30,
        "eval": {"placeholder_questions": {"n_questions": 400, "recall@10": 0.55, "recall@100": 0.8},
                 "questions_all_sample": {"n_questions": 1000, "n_available": 2255, "recall@10": 0.52,
                                          "recall@100": 0.77}},
        "latency_ms": {"single_query_median_ms": 40.0}, "determinism": {"same_query_twice": True},
        "edge_queries": {"''": {"hits": 0, "error": None}},
        "unique_token": {"rank": 1, "n_docs_in_test_index": 50_001}, "libraries": {"bm25s": "0.3.13"},
    }


def test_a2_passes() -> None:
    result = check_report("A2", a2_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(retriever="dense"),
    lambda r: r.update(n_docs=2_681_467),                                         # index != corpus + poison
    lambda r: r["eval"]["questions_all_sample"].update(n_questions=500),          # not 1,000 questions
    lambda r: r["eval"]["questions_all_sample"].update(**{"recall@100": 0.05}),   # not above random
    lambda r: r["eval"]["questions_all_sample"].update(**{"recall@10": 0.9}),     # recall@10 > recall@100
    lambda r: r["eval"]["placeholder_questions"].update(n_questions=390),
    lambda r: r["unique_token"].update(rank=2),
    lambda r: r["determinism"].update(same_query_twice=False),
    lambda r: r["edge_queries"]["''"].update(error="ValueError: boom"),
    lambda r: r.pop("index_bytes"),
], ids=["not-ok", "wrong-retriever", "doc-count", "n-questions", "recall-low", "recall-order", "placeholder",
        "unique-token", "determinism", "edge-crash", "field-missing"])
def test_a2_fails(breaker) -> None:
    report = copy.deepcopy(a2_report())
    breaker(report)
    assert check_report("A2", report).status == "FAIL"


def test_a2_low_but_plausible_recall_is_only_a_warning() -> None:
    report = a2_report()
    report["eval"]["questions_all_sample"].update(**{"recall@10": 0.3, "recall@100": 0.5})
    result = check_report("A2", report)
    assert result.status == "PASS" and any(not c.passed and not c.fatal for c in result.checks)


def a3_report() -> dict:
    """A report with the structure build_index.py --retriever dense writes (synthetic, for testing the rules only)."""
    return {
        "job": "A3", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-17T10:00:00+00:00",
        "retriever": "dense", "ok": True, "checks": {"norms_within_1e-3": True},
        "corpus": {"rows": 2_681_468}, "n_docs": 2_681_468, "n_poison": 0, "build_time_s": 3600.0,
        "index_bytes": 12 * 2**30, "model": "intfloat/e5-base-v2", "model_revision": "b533fe4636f4a2507c08ddab40644d20b0006d6a",
        "dim": 768, "device": "cuda", "vram_peak_gib": {"build": 6.1, "overall": 6.1, "cap": None},
        "settings": {"query_prefix": "query: ", "passage_prefix": "passage: ", "index_type": "hnsw"},
        "eval": {"placeholder_questions": {"n_questions": 400, "recall@10": 0.7, "recall@100": 0.9},
                 "questions_all_sample": {"n_questions": 1000, "n_available": 2255, "recall@10": 0.68,
                                          "recall@100": 0.91}},
        "latency_ms": {"single_query_median_ms": 12.0}, "determinism": {"same_query_twice": True},
        "edge_queries": {"''": {"hits": 10, "error": None}},
        "prefix_test": {"differ": True}, "reencode_check": {"n": 200, "min_cosine": 0.9999, "passed": True},
        "norms": {"min": 0.9998, "max": 1.0002, "passed": True}, "self_retrieval": {"n": 100, "top1_rate": 0.99,
                                                                                    "passed": True},
        "libraries": {"torch": "2.6.0"},
    }


def test_a3_passes() -> None:
    result = check_report("A3", a3_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(retriever="bm25"),
    lambda r: r.update(model="intfloat/e5-small-v2"),                               # not the Ada model
    lambda r: r.update(model_revision="unknown"),
    lambda r: r["settings"].update(query_prefix=""),
    lambda r: r.update(n_docs=1),
    lambda r: r["eval"]["questions_all_sample"].update(n_questions=10),
    lambda r: r["eval"]["questions_all_sample"].update(**{"recall@100": 0.1}),
    lambda r: r["reencode_check"].update(min_cosine=0.99, passed=False),
    lambda r: r["norms"].update(passed=False),
    lambda r: r["prefix_test"].update(differ=False),
    lambda r: r["self_retrieval"].update(passed=False),
    lambda r: r["determinism"].update(same_query_twice=False),
    lambda r: r["edge_queries"]["''"].update(error="RuntimeError: boom"),
    lambda r: r.pop("vram_peak_gib"),
], ids=["not-ok", "wrong-retriever", "wrong-model", "no-revision", "no-prefix", "doc-count", "n-questions",
        "recall-low", "reencode", "norms", "prefix", "self-retrieval", "determinism", "edge-crash", "field-missing"])
def test_a3_fails(breaker) -> None:
    report = copy.deepcopy(a3_report())
    breaker(report)
    assert check_report("A3", report).status == "FAIL"


def test_a3_cpu_encoding_is_only_a_warning() -> None:
    report = a3_report()
    report["device"] = "cpu"
    result = check_report("A3", report)
    assert result.status == "PASS" and any(not c.passed and not c.fatal for c in result.checks)


def a4_report() -> dict:
    """A report with the structure build_index.py --retriever splade writes (synthetic, for testing the rules only)."""
    return {
        "job": "A4", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-25T10:00:00+00:00",
        "retriever": "splade", "ok": True, "checks": {"weights_non_negative": True},
        "corpus": {"rows": 2_681_468}, "n_docs": 2_681_468, "n_poison": 0, "build_time_s": 5400.0,
        "index_bytes": 6 * 2**30, "model": "naver/splade-cocondenser-ensembledistil",
        "model_revision": "49cf4c7b0db5b870a401ddf5e2669993ef3699c7", "device": "cuda",
        "settings": {"doc_max_length": 256, "query_max_length": 64, "pooling": "max_log1p_relu"},
        "vram_peak_gib": {"build": 3.0, "overall": 3.1, "cap": None},
        "eval": {"placeholder_questions": {"n_questions": 400, "recall@10": 0.7, "recall@100": 0.92},
                 "questions_all_sample": {"n_questions": 1000, "n_available": 2255, "recall@10": 0.69,
                                          "recall@100": 0.93}},
        "latency_ms": {"single_query_median_ms": 40.0}, "determinism": {"same_query_twice": True},
        "edge_queries": {"''": {"hits": 0, "error": None}, "'the of and is'": {"hits": 10, "error": None}},
        "sparsity": {"doc": {"mean": 150.2, "max": 400, "min": 10, "empty_rows": 0},
                     "query": {"mean": 46.0, "max": 100, "min": 20, "n": 400}},
        "non_negative": {"passed": True}, "exactness": {"passed": True, "max_rel_diff_search_vs_manual": 1e-7,
                                                       "max_abs_diff_score_docs_vs_search": 0.0},
        "self_retrieval": {"n": 100, "top1_rate": 1.0, "passed": True}, "libraries": {"scipy": "1.18.1"},
    }


def test_a4_passes() -> None:
    result = check_report("A4", a4_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(retriever="dense"),
    lambda r: r.update(model="naver/splade_v2_max"),
    lambda r: r.update(model_revision=""),
    lambda r: r["settings"].update(doc_max_length=512),                          # differs from experiment.yaml
    lambda r: r["settings"].update(query_max_length=32),
    lambda r: r.update(n_docs=5),
    lambda r: r["sparsity"]["doc"].update(mean=0),
    lambda r: r["eval"]["questions_all_sample"].update(**{"recall@100": 0.1}),
    lambda r: r["non_negative"].update(passed=False),
    lambda r: r["exactness"].update(passed=False),
    lambda r: r["self_retrieval"].update(passed=False),
    lambda r: r["determinism"].update(same_query_twice=False),
    lambda r: r["edge_queries"]["''"].update(hits=5),                            # empty query must return nothing
    lambda r: r["edge_queries"]["'the of and is'"].update(error="ValueError: boom"),
    lambda r: r.pop("sparsity"),
], ids=["not-ok", "wrong-retriever", "wrong-model", "no-revision", "doc-len", "query-len", "doc-count",
        "no-sparsity", "recall-low", "negative", "not-exact", "self-retrieval", "determinism", "empty-query",
        "edge-crash", "field-missing"])
def test_a4_fails(breaker) -> None:
    report = copy.deepcopy(a4_report())
    breaker(report)
    assert check_report("A4", report).status == "FAIL"


def test_a4_empty_doc_vectors_only_warn() -> None:
    report = a4_report()
    report["sparsity"]["doc"]["empty_rows"] = 3
    result = check_report("A4", report)
    assert result.status == "PASS" and any(not c.passed and not c.fatal for c in result.checks)


def a5_report() -> dict:
    """A report with the structure scripts/build_cache.py writes (synthetic, for testing the rules only)."""
    def r(name, length, n_poison):
        return {"rows": 400, "n_questions": 400, "row_length": {"min": length[0], "median": 150, "max": length[1],
                                                                 "short_rows": 0 if length[0] == 150 else 3},
                "all_rows_sorted": True, "fingerprint_matches_index": True, "index_n_poison": n_poison,
                "spot_check": {"n": 20, "ids_and_order_equal": 20, "score_docs_agree": 20, "passed": True},
                "unmasked_reference": name == "rrf"}
    return {"job": "A5", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-25T10:00:00+00:00",
            "ok": True, "checks": {"dense_spot_check": True}, "questions": {"path": "data/questions.jsonl", "n": 400},
            "depth": 150, "retrievers": {"bm25": r("bm25", (40, 150), 3000), "dense": r("dense", (150, 150), 3000),
                                         "splade": r("splade", (150, 150), 3000),
                                         "rrf": r("rrf", (150, 150), {"bm25": 3000, "dense": 3000})}}


def test_a5_passes() -> None:
    result = check_report("A5", a5_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(depth=100),
    lambda r: r["questions"].update(n=399),
    lambda r: r["retrievers"].pop("splade"),
    lambda r: r["retrievers"].update(rrf={"error": "refusing to build rrf"}),
    lambda r: r["retrievers"]["bm25"].update(rows=399),
    lambda r: r["retrievers"]["dense"]["row_length"].update(min=149),           # dense rows must be exactly 150
    lambda r: r["retrievers"]["splade"]["row_length"].update(max=151),
    lambda r: r["retrievers"]["bm25"].update(all_rows_sorted=False),
    lambda r: r["retrievers"]["dense"]["spot_check"].update(passed=False),
    lambda r: r["retrievers"]["splade"]["spot_check"].update(n=5),             # fewer than 10 random rows
    lambda r: r["retrievers"]["bm25"].update(fingerprint_matches_index=False),
    lambda r: r["retrievers"]["rrf"].update(unmasked_reference=False),
    lambda r: r.pop("retrievers"),
], ids=["not-ok", "depth", "n-questions", "missing-retriever", "rrf-error", "row-count", "dense-length",
        "too-long", "unsorted", "spot-check", "spot-check-small", "fingerprint", "rrf-flag", "field-missing"])
def test_a5_fails(breaker) -> None:
    report = copy.deepcopy(a5_report())
    breaker(report)
    assert check_report("A5", report).status == "FAIL"


def test_a5_no_poison_in_index_only_warns() -> None:
    report = a5_report()
    for name in ("bm25", "dense", "splade"):
        report["retrievers"][name]["index_n_poison"] = 0
    result = check_report("A5", report)
    assert result.status == "PASS"
    assert sum(1 for c in result.checks if not c.passed and not c.fatal) == 3


def a6_report() -> dict:
    """A report with the structure scripts/index_poison.py writes (synthetic, for testing the rules only)."""
    def r(name):
        m = ({"rank_diff_max": 4, "rank_diff_mean": 0.1, "frac_top5_unchanged": 0.99, "frac_top50_unchanged": 0.8,
              "jaccard_at_50_mean": 0.999, "n_questions": 50, "identical_ids_and_order": 30, "exact": False}
             if name == "bm25" else {"n_questions": 50, "identical_ids_and_order": 50, "exact": True,
                                     "max_rel_score_diff": 1e-7})
        return {"n_poison_before": 0, "n_poison_after": 3000, "n_docs_before": 2_681_468, "n_docs_after": 2_684_468,
                "index_bytes_before": 10**9, "index_bytes_after": 10**9 + 10**6,
                "reach": {"by_attack": {"fact": {"n": 1000, "frac_in_top10": 0.8}}, "any_in_top10": True},
                "top5_changed_frac": {"all": 0.4, "clean_role": 0.01}, "masking": m}
    return {"job": "A6", "env": "ada", "git_commit": "abc1234", "timestamp_utc": "2026-10-25T10:00:00+00:00",
            "ok": True, "checks": {"shards_valid": True}, "work_copy": False,
            "shards": {"files": ["poison_main.jsonl", "poison_adaptive.jsonl"], "n_records": 3000,
                       "by_attack": {"fact": 1000, "hijack": 1000, "diverse": 500, "repeat": 250, "stealth": 250},
                       "is_poison_recoverable_from_doc_id": "3000/3000", "qids_with_role_clean": []},
            "questions": {"n": 400}, "retrievers": {n: r(n) for n in ("bm25", "dense", "splade")}}


def test_a6_passes() -> None:
    result = check_report("A6", a6_report())
    assert result.status == "PASS", [c for c in result.failed]


@pytest.mark.parametrize("breaker", [
    lambda r: r.update(ok=False),
    lambda r: r.update(work_copy=True),                                        # Ada must ingest for real
    lambda r: r["shards"].update(n_records=0),
    lambda r: r["shards"].update(is_poison_recoverable_from_doc_id="2999/3000"),
    lambda r: r["retrievers"].pop("splade"),
    lambda r: r["retrievers"]["dense"].update(n_poison_after=10),
    lambda r: r["retrievers"]["bm25"].update(index_bytes_after=None),
    lambda r: r["retrievers"]["dense"]["masking"].update(exact=False),
    lambda r: r["retrievers"]["splade"]["masking"].update(identical_ids_and_order=49),
    lambda r: r["retrievers"]["splade"]["masking"].update(n_questions=20, identical_ids_and_order=20),
    lambda r: r["retrievers"]["bm25"]["masking"].pop("rank_diff_max"),
    lambda r: [x["reach"].update(any_in_top10=False) for x in r["retrievers"].values()],
    lambda r: r["retrievers"]["dense"].pop("top5_changed_frac"),
    lambda r: r.pop("shards"),
], ids=["not-ok", "work-copy", "no-records", "ids", "missing-retriever", "n-poison", "size", "dense-not-exact",
        "splade-not-identical", "too-few-questions", "bm25-not-reported", "no-reach", "no-top5", "field-missing"])
def test_a6_fails(breaker) -> None:
    report = copy.deepcopy(a6_report())
    breaker(report)
    assert check_report("A6", report).status == "FAIL"


def test_a6_fixture_sized_or_non_adaptive_shards_only_warn() -> None:
    report = a6_report()
    report["shards"].update(n_records=10, by_attack={"fact": 5, "hijack": 5},
                            is_poison_recoverable_from_doc_id="10/10")
    for r in report["retrievers"].values():
        r["n_poison_after"] = 10
    result = check_report("A6", report)
    assert result.status == "PASS" and sum(1 for c in result.checks if not c.passed and not c.fatal) == 2
