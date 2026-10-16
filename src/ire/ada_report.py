"""Acceptance checks for Ada job reports (IMPLEMENTATION_PLAN.md section 5).

`check_report(job, report)` runs the rules for one job on a parsed `reports/<job>.json` and returns a
ReportResult. A step may go from DONE-LOCAL to DONE only when the result status is PASS.

Status values:
    PASS             every check passed on a report with env=ada
    FAIL             at least one check failed
    NOT_IMPLEMENTED  the rules for this job are still a stub (they are filled in with the job's step)
    LOCAL_DRY_RUN    all checks passed but the report is env=local (only with allow_local=True);
                     useful to test the rules, never counts for DONE
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yaml

from ire.config import CONFIG_DIR, load_env_profile

def load_experiment() -> dict[str, Any]:
    """configs/experiment.yaml as a dict (question counts and folds for the A1 rules)."""
    with (CONFIG_DIR / "experiment.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


JOBS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    fatal: bool = True  # non-fatal checks are warnings: shown, but do not fail the report


@dataclass
class ReportResult:
    job: str
    status: str
    checks: list[Check] = field(default_factory=list)
    note: str = ""

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.fatal and not c.passed]


# ---------------------------------------------------------------- shared checks

COMMON_FIELDS: tuple[str, ...] = ("job", "env", "git_commit", "timestamp_utc")


def _common_checks(job: str, report: dict[str, Any], allow_local: bool) -> list[Check]:
    checks = []
    missing = [k for k in COMMON_FIELDS if k not in report]
    checks.append(Check("common fields present", not missing, f"missing: {missing}" if missing else ""))
    names = {job, f"{job}_local"} if allow_local else {job}  # local reports are written as <job>_local
    checks.append(Check("job name matches", report.get("job") in names,
                        f"report says job={report.get('job')!r}, expected {job!r}"))
    commit = str(report.get("git_commit", "unknown"))
    checks.append(Check("git commit recorded", commit not in ("", "unknown"), f"git_commit={commit!r}"))
    checks.append(Check("working tree clean on Ada", not commit.endswith("-dirty"),
                        "run from a clean checkout so the code version is known", fatal=False))
    return checks


# ---------------------------------------------------------------- A0 (fields written by scripts/check_env.py)

A0_FIELDS: tuple[str, ...] = ("ok", "python", "platform", "torch", "disk", "libraries", "java",
                              "models_cached", "hf_token_set", "runtime_s", "rerun")


def _rules_a0(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A0_FIELDS if k not in report]
    checks.append(Check("check_env fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    checks.append(Check("check_env RESULT OK", report["ok"] is True, f"ok={report['ok']!r}"))

    torch = report["torch"]
    checks.append(Check("torch importable", bool(torch.get("importable")), str(torch.get("error", ""))))
    gpus = torch.get("gpus") or []
    checks.append(Check("CUDA available with >= 1 GPU", bool(torch.get("cuda_available")) and len(gpus) >= 1,
                        "re-run inside a GPU session: bash ada/setup_env.sh --check-only"
                        if not gpus else ", ".join(f"{g['name']} ({g['total_gib']} GiB)" for g in gpus)))

    libs = report["libraries"]
    bad = [n for n, v in libs.items() if v.get("required") and not v.get("importable")]
    checks.append(Check("required libraries importable", not bad, f"not importable: {bad}" if bad else
                        f"{sum(1 for v in libs.values() if v.get('importable'))} importable"))

    disk = report["disk"]
    checks.append(Check("IRE_DATA_DIR writable", disk.get("writable") is True, str(disk.get("path", ""))))
    free = disk.get("free_gib")
    checks.append(Check("free disk reported", isinstance(free, (int, float)), f"free_gib={free!r}"))

    profile = "local" if report.get("env") == "local" else "ada"  # local only reaches here as a dry run
    expected_models = set(load_env_profile(profile)["models"].values())
    cached = report["models_cached"]
    not_cached = sorted(m for m in expected_models if cached.get(m) is not True)
    checks.append(Check(f"all env_{profile}.yaml models cached", not not_cached,
                        f"not cached: {not_cached}" if not_cached else ", ".join(sorted(expected_models))))

    py = str(report["python"])
    checks.append(Check("Python 3.13 (requirements were pinned on 3.13)", py.startswith("3.13."),
                        f"python={py}", fatal=False))
    return checks


# ---------------------------------------------------------------- A1 (fields written by scripts/prepare_data.py)

A1_FIELDS: tuple[str, ...] = ("corpus_mode", "n_passages_requested", "counts", "checks", "ok", "files", "sources")
FULL_CORPUS_RANGE: tuple[int, int] = (2_600_000, 2_750_000)  # BEIR NQ has ~2.68M passages
SUBSET_CORPUS_RANGE: tuple[int, int] = (400_000, 600_000)    # the ~500k fallback


def _rules_a1(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A1_FIELDS if k not in report]
    checks.append(Check("prepare_data fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    c, mode = report["counts"], report["corpus_mode"]
    checks.append(Check("prepare_data RESULT OK", report["ok"] is True, f"ok={report['ok']!r}"))

    n = c.get("corpus_passages")
    if mode == "full":
        lo, hi = FULL_CORPUS_RANGE
        expected = c.get("raw_corpus_passages", -1) - c.get("raw_duplicate_doc_ids", 0)
        size_ok = n == expected and lo <= (n or 0) <= hi
        detail = f"{n:,} passages; expected all {expected:,} unique BEIR passages, within {lo:,}-{hi:,}"
    elif mode == "subset":
        lo, hi = SUBSET_CORPUS_RANGE
        size_ok, detail = lo <= (n or 0) <= hi, f"{n:,} passages; expected {lo:,}-{hi:,}"
    else:
        size_ok = n == report["n_passages_requested"]
        detail = f"{n:,} passages; requested {report['n_passages_requested']}"
    checks.append(Check(f"corpus size matches mode={mode}", size_ok, detail))
    rows = report["files"].get("corpus", {}).get("rows")
    checks.append(Check("corpus.jsonl row count = corpus passages", rows == n, f"rows={rows}"))

    exp = load_experiment()
    n_t, n_c, folds = exp["questions"]["n_target"], exp["questions"]["n_clean"], exp["questions"]["folds"]
    want = {f"{r}/{f}": n // len(folds) for r, n in (("target", n_t), ("clean", n_c)) for f in folds}
    got = c.get("role_fold", {})
    checks.append(Check(f"{n_t} target + {n_c} clean, balanced over folds", got == want, str(got)))

    checks.append(Check("every question gold ID exists in the corpus", c.get("question_gold_missing_in_corpus") == 0,
                        f"missing: {c.get('question_gold_missing_in_corpus')}"))
    if mode == "full" or report.get("dev_subset"):
        checks.append(Check("every questions_all gold ID exists in the corpus",
                            c.get("questions_all_gold_missing_in_corpus") == 0,
                            f"missing: {c.get('questions_all_gold_missing_in_corpus')}"))
    checks.append(Check("no duplicate doc IDs", c.get("corpus_duplicate_doc_ids") == 0,
                        f"duplicates: {c.get('corpus_duplicate_doc_ids')}"))
    checks.append(Check("no duplicate question IDs", report["checks"].get("no_duplicate_qids") is True))
    return checks


# ---------------------------------------------------------------- A2-A9 (stubs; filled in with each step)

STUBS: dict[str, tuple[int, str]] = {
    # job: (step that fills in the rules, acceptance check from IMPLEMENTATION_PLAN.md section 5)
    "A2": (3, "BM25 index builds; Recall@100 on 1,000 questions reported; unique-token doc found at rank 1"),
    "A3": (4, "dense index builds; Recall@100 on 1,000 questions reported; unique-token doc found at rank 1"),
    "A4": (5, "SPLADE index builds; Recall@100 on 1,000 questions reported; unique-token doc found at rank 1"),
    "A5": (7, "row count = number of questions; every row has 150 sorted entries; 10 random rows match a "
              "fresh search"),
    "A6": (8, "a poison passage for question X is in X's top-10 for >= 1 retriever; index sizes reported"),
    "A7": (9, "throughput (requests/min) and projected hours for 60,000 requests; cache hit on repeat; "
              "greedy determinism on 20 prompts"),
    "A8": (11, "flag rate on 1,000 held-out clean passages close to alpha; cache size; fraction too_short"),
    "A9": (13, "all four retrievers and both filters run end to end on 20 questions"),
}

RULES: dict[str, Callable[[dict[str, Any]], list[Check]]] = {"A0": _rules_a0, "A1": _rules_a1}


def check_report(job: str, report: dict[str, Any], allow_local: bool = False) -> ReportResult:
    """Run the acceptance checks for `job` on a parsed report dict."""
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; expected one of {JOBS}")
    if job not in RULES:
        step, text = STUBS[job]
        return ReportResult(job, "NOT_IMPLEMENTED",
                            note=f"rules for {job} are added in step {step}. Acceptance check: {text}")

    env = report.get("env")
    checks = _common_checks(job, report, allow_local)
    checks.append(Check("report env is ada" + (" (local accepted as dry run)" if allow_local else ""),
                        env == "ada" or (allow_local and env == "local"),
                        f"env={env!r}"))
    checks.extend(RULES[job](report))

    result = ReportResult(job, "PASS", checks)
    if result.failed:
        result.status = "FAIL"
    elif env != "ada":
        result.status = "LOCAL_DRY_RUN"
        result.note = "env=local: the rules pass, but only an env=ada report can mark a step DONE"
    return result
