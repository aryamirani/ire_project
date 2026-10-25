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


# ---------------------------------------------------------------- A2 (fields written by scripts/build_index.py)

A2_FIELDS: tuple[str, ...] = ("retriever", "ok", "checks", "corpus", "n_docs", "n_poison", "build_time_s",
                              "index_bytes", "eval", "latency_ms", "determinism", "edge_queries", "unique_token",
                              "libraries")
MIN_RECALL_100 = 0.2   # "clearly above random" (random Recall@100 on ~2.7M passages is ~0)
EXPECTED_RECALL_100 = 0.6  # BEIR reports BM25 Recall@100 = 0.76 on NQ; lower is a warning worth a look


def _rules_a2(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A2_FIELDS if k not in report]
    checks.append(Check("build_index fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    checks.append(Check("retriever is bm25", report["retriever"] == "bm25", f"retriever={report['retriever']!r}"))
    checks.append(Check("build_index RESULT OK", report["ok"] is True,
                        str({k: v for k, v in report["checks"].items() if not v}) if not report["ok"] else ""))
    rows, n_docs, n_poison = report["corpus"].get("rows"), report["n_docs"], report["n_poison"]
    checks.append(Check("index holds corpus + poison", n_docs == (rows or 0) + n_poison,
                        f"n_docs={n_docs:,}, corpus rows={rows}, poison={n_poison}"))
    lo, _ = FULL_CORPUS_RANGE
    checks.append(Check("full-size corpus indexed", (rows or 0) >= lo, f"corpus rows={rows}", fatal=False))
    checks.append(Check("build time and index size reported",
                        isinstance(report["build_time_s"], (int, float)) and report["index_bytes"] > 0,
                        f"build {report['build_time_s']} s, {report['index_bytes'] / 2**20:,.1f} MiB"))

    checks.extend(_recall_checks(report, "BM25 (BEIR reports 0.76 on NQ)"))
    ut = report["unique_token"]
    checks.append(Check("unique-token passage found at rank 1", ut.get("rank") == 1,
                        f"rank={ut.get('rank')} in a {ut.get('n_docs_in_test_index')}-doc temp index"))
    checks.append(Check("same query twice gives identical results",
                        report["determinism"].get("same_query_twice") is True))
    edge_ok = all(v.get("error") is None for v in report["edge_queries"].values())
    checks.append(Check("empty / stopword / unknown-word queries do not crash", edge_ok, str(report["edge_queries"])))
    return checks


# ---------------------------------------------------------------- A3 (fields written by build_index.py --retriever dense)

A3_FIELDS: tuple[str, ...] = ("retriever", "ok", "checks", "corpus", "n_docs", "n_poison", "build_time_s",
                              "index_bytes", "model", "model_revision", "dim", "device", "eval", "latency_ms",
                              "determinism", "edge_queries", "prefix_test", "reencode_check", "norms",
                              "self_retrieval", "vram_peak_gib", "settings", "libraries")


def _recall_checks(report: dict[str, Any], name: str) -> list[Check]:
    """Recall@100 on the configured number of questions, clearly above random; all placeholder questions present."""
    exp = load_experiment()
    n_eval = exp["retrieval_eval"]["n_questions"]
    pool = report["eval"].get("questions_all_sample", {})
    want = min(n_eval, pool.get("n_available", 0))
    r10, r100 = pool.get("recall@10"), pool.get("recall@100")
    numeric = isinstance(r10, (int, float)) and isinstance(r100, (int, float))
    n_placeholder = exp["questions"]["n_target"] + exp["questions"]["n_clean"]
    ph = report["eval"].get("placeholder_questions", {}).get("n_questions")
    return [
        Check(f"Recall@100 on {n_eval:,} questions (or all available)", pool.get("n_questions") == want and want > 0,
              f"n_questions={pool.get('n_questions')}, available={pool.get('n_available')}"),
        Check(f"Recall@100 clearly above random (> {MIN_RECALL_100})",
              numeric and MIN_RECALL_100 < r100 <= 1 and r10 <= r100, f"recall@10={r10}, recall@100={r100}"),
        Check(f"Recall@100 at a plausible {name} level (>= {EXPECTED_RECALL_100})",
              numeric and r100 >= EXPECTED_RECALL_100, f"recall@100={r100}", fatal=False),
        Check(f"all {n_placeholder} questions.jsonl questions have gold in the index", ph == n_placeholder,
              f"n_questions={ph}"),
    ]


def _rules_a3(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A3_FIELDS if k not in report]
    checks.append(Check("build_index fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    checks.append(Check("retriever is dense", report["retriever"] == "dense", f"retriever={report['retriever']!r}"))
    checks.append(Check("build_index RESULT OK", report["ok"] is True,
                        str({k: v for k, v in report["checks"].items() if not v}) if not report["ok"] else ""))
    profile = "local" if report.get("env") == "local" else "ada"  # local only reaches here as a dry run
    want_model = load_env_profile(profile)["models"]["dense"]
    checks.append(Check(f"model is env_{profile}.yaml models.dense", report["model"] == want_model,
                        f"{report['model']} (expected {want_model})"))
    checks.append(Check("model revision recorded", report["model_revision"] not in ("", "unknown", None),
                        str(report["model_revision"])))
    checks.append(Check("E5 prefixes used", report["settings"].get("query_prefix") == "query: "
                        and report["settings"].get("passage_prefix") == "passage: ", str(report["settings"])))
    rows, n_docs, n_poison = report["corpus"].get("rows"), report["n_docs"], report["n_poison"]
    checks.append(Check("index holds corpus + poison", n_docs == (rows or 0) + n_poison,
                        f"n_docs={n_docs:,}, corpus rows={rows}, poison={n_poison}"))
    lo, _ = FULL_CORPUS_RANGE
    checks.append(Check("full-size corpus indexed", (rows or 0) >= lo, f"corpus rows={rows}", fatal=False))
    checks.append(Check("encoded on a GPU", report["device"] == "cuda", f"device={report['device']}", fatal=False))
    checks.append(Check("build time and index size reported",
                        isinstance(report["build_time_s"], (int, float)) and report["index_bytes"] > 0,
                        f"build {report['build_time_s']} s, {report['index_bytes'] / 2**20:,.1f} MiB, "
                        f"peak VRAM {report['vram_peak_gib']}"))
    checks.extend(_recall_checks(report, "dense"))
    rc = report["reencode_check"]
    checks.append(Check("stored vectors match re-encoded passages (cosine > 0.999)",
                        rc.get("passed") is True and (rc.get("min_cosine") or 0) > 0.999,
                        f"min cosine {rc.get('min_cosine')} over {rc.get('n')} passages"))
    checks.append(Check("vector norms 1 +/- 1e-3", report["norms"].get("passed") is True, str(report["norms"])))
    checks.append(Check("prefix changes the vectors", report["prefix_test"].get("differ") is True))
    sr = report["self_retrieval"]
    checks.append(Check("a passage's own text retrieves it at rank 1 (>= 90%)", sr.get("passed") is True,
                        f"top-1 rate {sr.get('top1_rate')} over {sr.get('n')} passages"))
    checks.append(Check("same query twice gives identical results",
                        report["determinism"].get("same_query_twice") is True))
    edge_ok = all(v.get("error") is None for v in report["edge_queries"].values())
    checks.append(Check("empty / stopword / unknown-word queries do not crash", edge_ok, str(report["edge_queries"])))
    return checks


# ---------------------------------------------------------------- A4 (fields written by build_index.py --retriever splade)

A4_FIELDS: tuple[str, ...] = ("retriever", "ok", "checks", "corpus", "n_docs", "n_poison", "build_time_s",
                              "index_bytes", "model", "model_revision", "device", "settings", "eval", "latency_ms",
                              "determinism", "edge_queries", "sparsity", "non_negative", "exactness",
                              "self_retrieval", "vram_peak_gib", "libraries")


def _rules_a4(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A4_FIELDS if k not in report]
    checks.append(Check("build_index fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    checks.append(Check("retriever is splade", report["retriever"] == "splade", f"retriever={report['retriever']!r}"))
    checks.append(Check("build_index RESULT OK", report["ok"] is True,
                        str({k: v for k, v in report["checks"].items() if not v}) if not report["ok"] else ""))
    profile = "local" if report.get("env") == "local" else "ada"  # local only reaches here as a dry run
    want_model = load_env_profile(profile)["models"]["splade"]
    checks.append(Check(f"model is env_{profile}.yaml models.splade", report["model"] == want_model,
                        f"{report['model']} (expected {want_model})"))
    checks.append(Check("model revision recorded", report["model_revision"] not in ("", "unknown", None),
                        str(report["model_revision"])))
    exp = load_experiment()["splade"]
    s = report["settings"]
    same = (s.get("doc_max_length"), s.get("query_max_length"), s.get("pooling")) == (
        exp["doc_max_length"], exp["query_max_length"], exp["pooling"])
    checks.append(Check("doc/query max length and pooling as in experiment.yaml", same, str(s)))
    rows, n_docs, n_poison = report["corpus"].get("rows"), report["n_docs"], report["n_poison"]
    checks.append(Check("index holds corpus + poison", n_docs == (rows or 0) + n_poison,
                        f"n_docs={n_docs:,}, corpus rows={rows}, poison={n_poison}"))
    lo, _ = FULL_CORPUS_RANGE
    checks.append(Check("full-size corpus indexed", (rows or 0) >= lo, f"corpus rows={rows}", fatal=False))
    checks.append(Check("encoded on a GPU", report["device"] == "cuda", f"device={report['device']}", fatal=False))
    checks.append(Check("build time, index size and peak VRAM reported",
                        isinstance(report["build_time_s"], (int, float)) and report["index_bytes"] > 0,
                        f"build {report['build_time_s']} s, {report['index_bytes'] / 2**30:,.2f} GiB, "
                        f"peak VRAM {report['vram_peak_gib']}"))
    sparsity = report["sparsity"]
    doc_mean = (sparsity.get("doc") or {}).get("mean")
    q_mean = (sparsity.get("query") or {}).get("mean")
    checks.append(Check("non-zeros per document and per query reported",
                        isinstance(doc_mean, (int, float)) and doc_mean > 0 and isinstance(q_mean, (int, float)),
                        f"doc mean {doc_mean}, query mean {q_mean}"))
    checks.append(Check("no empty document vectors", (sparsity.get("doc") or {}).get("empty_rows") == 0,
                        str(sparsity.get("doc")), fatal=False))
    checks.extend(_recall_checks(report, "SPLADE"))
    checks.append(Check("all weights non-negative", report["non_negative"].get("passed") is True,
                        str(report["non_negative"])))
    ex = report["exactness"]
    checks.append(Check("search() equals the dot product of stored vectors; score_docs agrees",
                        ex.get("passed") is True, f"max rel diff {ex.get('max_rel_diff_search_vs_manual')}, "
                        f"score_docs max abs diff {ex.get('max_abs_diff_score_docs_vs_search')}"))
    sr = report["self_retrieval"]
    checks.append(Check("a passage's own text retrieves it at rank 1 (>= 90%)", sr.get("passed") is True,
                        f"top-1 rate {sr.get('top1_rate')} over {sr.get('n')} passages"))
    checks.append(Check("same query twice gives identical results",
                        report["determinism"].get("same_query_twice") is True))
    edge = report["edge_queries"]
    edge_ok = all(v.get("error") is None for v in edge.values()) and (edge.get("''") or {}).get("hits") == 0
    checks.append(Check("edge queries do not crash; empty query returns nothing", edge_ok, str(edge)))
    return checks


# ---------------------------------------------------------------- A5 (fields written by scripts/build_cache.py)

A5_RETRIEVERS: tuple[str, ...] = ("bm25", "dense", "splade", "rrf")
A5_FIELDS: tuple[str, ...] = ("ok", "checks", "questions", "depth", "retrievers")


def _rules_a5(report: dict[str, Any]) -> list[Check]:
    checks = []
    missing = [k for k in A5_FIELDS if k not in report]
    checks.append(Check("build_cache fields present", not missing, f"missing: {missing}" if missing else ""))
    if missing:
        return checks
    checks.append(Check("build_cache RESULT OK", report["ok"] is True,
                        str({k: v for k, v in report["checks"].items() if not v}) if not report["ok"] else ""))
    exp = load_experiment()
    depth = exp["top_k_cache"]
    n_final = exp["questions"]["n_target"] + exp["questions"]["n_clean"]
    n_q = report["questions"].get("n")
    checks.append(Check(f"cache depth is {depth}", report["depth"] == depth, f"depth={report['depth']}"))
    checks.append(Check(f"question set has {n_final} questions", n_q == n_final, f"n={n_q}"))
    rs = report["retrievers"]
    for name in A5_RETRIEVERS:
        r = rs.get(name)
        if r is None or "error" in r:
            checks.append(Check(f"{name}: cache built", False, (r or {}).get("error", "not in the report")))
            continue
        checks.append(Check(f"{name}: row count = number of questions", r.get("rows") == n_q == r.get("n_questions"),
                            f"rows={r.get('rows')}, questions={n_q}"))
        rl = r.get("row_length") or {}
        if name == "dense":
            checks.append(Check(f"dense: every row exactly {depth} entries",
                                rl.get("min") == rl.get("max") == depth, str(rl)))
        else:
            checks.append(Check(f"{name}: rows at most {depth} entries", (rl.get("max") or 0) <= depth, str(rl)))
        checks.append(Check(f"{name}: all rows sorted (score desc, doc_id)", r.get("all_rows_sorted") is True))
        sc = r.get("spot_check") or {}
        checks.append(Check(f"{name}: random-row spot check vs fresh search passed",
                            sc.get("passed") is True and (sc.get("n") or 0) >= 10,
                            f"{sc.get('ids_and_order_equal')}/{sc.get('n')} rows equal, "
                            f"{sc.get('score_docs_agree')}/{sc.get('n')} scores agree"))
        checks.append(Check(f"{name}: sidecar fingerprint matches the index manifest",
                            r.get("fingerprint_matches_index") is True))
        n_poison = r.get("index_n_poison")
        values = n_poison.values() if isinstance(n_poison, dict) else [n_poison]
        checks.append(Check(f"{name}: index contains poison documents (A6 done)",
                            all(isinstance(v, int) and v > 0 for v in values),
                            f"index_n_poison={n_poison}: a cache without poison is only a timing test",
                            fatal=False))
    if "rrf" in rs and "error" not in rs["rrf"]:
        checks.append(Check("rrf: marked as unmasked reference", rs["rrf"].get("unmasked_reference") is True))
    return checks


# ---------------------------------------------------------------- A6-A9 (stubs; filled in with each step)

STUBS: dict[str, tuple[int, str]] = {
    # job: (step that fills in the rules, acceptance check from IMPLEMENTATION_PLAN.md section 5)
    "A6": (8, "a poison passage for question X is in X's top-10 for >= 1 retriever; index sizes reported"),
    "A7": (9, "throughput (requests/min) and projected hours for 60,000 requests; cache hit on repeat; "
              "greedy determinism on 20 prompts"),
    "A8": (11, "flag rate on 1,000 held-out clean passages close to alpha; cache size; fraction too_short"),
    "A9": (13, "all four retrievers and both filters run end to end on 20 questions"),
}

RULES: dict[str, Callable[[dict[str, Any]], list[Check]]] = {"A0": _rules_a0, "A1": _rules_a1, "A2": _rules_a2,
                                                          "A3": _rules_a3, "A4": _rules_a4, "A5": _rules_a5}


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
