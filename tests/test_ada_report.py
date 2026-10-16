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


@pytest.mark.parametrize("job", [j for j in JOBS if j not in ("A0", "A1")])
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
