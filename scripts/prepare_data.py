"""Step 2 / job A1: download BEIR NQ + nq_open, build corpus.jsonl, questions_all.jsonl, questions.jsonl.

Usage (repo root; PowerShell or bash):
    python scripts/prepare_data.py                          # mode from config (local: random 5,000; ada: full)
    python scripts/prepare_data.py --corpus-mode full       # all ~2.68M passages (Ada only)
    python scripts/prepare_data.py --dev-subset             # 1,000 questions + 50,000 passages into data/dev/
    python scripts/prepare_data.py --corpus-mode subset     # ~500k gold + hard negatives: TODO after step 5
    python scripts/prepare_data.py --source ukp             # original BEIR nq.zip instead of the HF copy

Outputs (paths from configs/paths.yaml, under $IRE_DATA_DIR):
    raw/beir_hf/... (or raw/beir/nq.zip), raw/nq_open/...   downloads (pinned revisions; resumable)
    data/corpus.jsonl, data/questions_all.jsonl, data/questions.jsonl   (or data/dev/ with --dev-subset)
    reports/<report>.json                        counts, file sizes, checks (default name A1 on Ada, A1_local here)
    <repo>/reports/data_card_<env>[_dev].json    counts, file sizes and sources (no absolute paths)

Resumable: downloads resume; if the outputs already exist for the same settings the build is skipped
(use --force to rebuild) and only the checks and the report run again. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402
from ire.data.download import (  # noqa: E402
    BEIR_HF_FILES,
    BEIR_NQ_URL,
    BEIR_NQ_ZIP,
    NQ_OPEN_REPO,
    NQ_OPEN_REVISION,
    beir_hf_paths,
    beir_hf_sizes,
    download_beir_hf,
    download_file,
    download_nq_open,
    nq_open_paths,
    nq_open_sizes,
    remote_size,
    sha256_file,
)
from ire.data.nq import BeirParquet, BeirSource, BeirZip  # noqa: E402
from ire.data.prepare import CORPUS_MODES, OutputPaths, prepare, sample_records, verify_outputs  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402

GIB = 2**30
STAMP = ".prepare_data.json"  # written next to the outputs; lets a re-run skip an identical build


def file_info(path: Path, data_dir: Path, rows: bool = True) -> dict[str, Any]:
    """Size, row count and sha256 of a file; the path is given relative to IRE_DATA_DIR."""
    info: dict[str, Any] = {"path": path.relative_to(data_dir).as_posix(), "bytes": path.stat().st_size,
                            "mib": round(path.stat().st_size / 2**20, 2), "sha256": sha256_file(path)}
    if rows:
        with path.open("rb") as fh:
            info["rows"] = sum(1 for line in fh if line.strip())
    return info


def raw_layout(source: str, raw_dir: Path) -> tuple[BeirSource, dict[str, Path]]:
    """The BEIR reader for `source` and every raw file path (role -> path) under raw_dir."""
    nq = {f"nq_open_{split}": path for split, path in nq_open_paths(raw_dir).items()}
    if source == "ukp":
        zip_file = raw_dir / BEIR_NQ_ZIP
        return BeirZip(zip_file), {"beir_nq_zip": zip_file, **nq}
    hf = beir_hf_paths(raw_dir)
    src = BeirParquet(hf["corpus"], hf["queries"], hf["qrels_test"].parent)
    return src, {f"beir_{role}": path for role, path in hf.items()} | nq


def remote_sizes(source: str) -> dict[str, int | None]:
    """Remote size of every raw file (role -> bytes), same keys as raw_layout."""
    beir = ({"beir_nq_zip": remote_size(BEIR_NQ_URL)} if source == "ukp"
            else {f"beir_{role}": size for role, size in beir_hf_sizes().items()})
    return beir | {f"nq_open_{split}": size for split, size in nq_open_sizes().items()}


def ensure_downloads(source: str, raw_dir: Path, max_download_gb: float, min_free_gb: float,
                     est_output_bytes: int, log) -> dict[str, Any]:
    """Check sizes and free disk, then download what is missing. Exits (code 1) if a limit would be crossed."""
    _, paths = raw_layout(source, raw_dir)
    sizes = remote_sizes(source)
    todo = sum(size or 0 for role, size in sizes.items()
               if not (paths[role].is_file() and paths[role].stat().st_size == size))
    raw_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw_dir).free
    free_after = free - todo - est_output_bytes
    log.info("source=%s download sizes (MiB): %s; still to download %.1f MiB", source,
             {k: round((v or 0) / 2**20, 1) for k, v in sizes.items()}, todo / 2**20)
    log.info("free disk %.1f GiB; estimated free after download + outputs %.1f GiB", free / GIB, free_after / GIB)
    if todo > max_download_gb * GIB:
        log.error("download of %.1f GiB exceeds --max-download-gb %.1f: stopping (ask before downloading)",
                  todo / GIB, max_download_gb)
        sys.exit(1)
    if free_after < min_free_gb * GIB:
        log.error("free disk would drop to %.1f GiB (< --min-free-gb %.1f): stopping", free_after / GIB, min_free_gb)
        sys.exit(1)
    if source == "ukp":
        download_file(BEIR_NQ_URL, paths["beir_nq_zip"], sizes["beir_nq_zip"])
    else:
        download_beir_hf(raw_dir)
    download_nq_open(raw_dir)
    return {"free_gib_before": round(free / GIB, 2), "downloaded_mib": round(todo / 2**20, 1),
            "remote_bytes": sizes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-mode", choices=CORPUS_MODES, help="default: corpus.mode from the env profile")
    parser.add_argument("--n-passages", type=int, help="random mode: total passages (default: corpus.n_passages)")
    parser.add_argument("--dev-subset", action="store_true",
                        help="1,000 questions + 50,000 passages (dev_subset in experiment.yaml) into data/dev/")
    parser.add_argument("--source", choices=["hf", "ukp"], default="hf",
                        help="where BEIR NQ comes from: hf = pinned Hugging Face copy (default, fast), "
                             "ukp = the original nq.zip (slow server)")
    parser.add_argument("--report", help="report name (default: A1 on Ada, A1_local otherwise; _dev added for --dev-subset)")
    parser.add_argument("--skip-download", action="store_true", help="use the raw files already in raw_dir")
    parser.add_argument("--force", action="store_true", help="rebuild even if the outputs exist for these settings")
    parser.add_argument("--max-download-gb", type=float, default=6.0)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--delete-raw", action="store_true",
                        help="after every check passes, delete the raw downloads (they can be downloaded again)")
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config()
    log = get_logger("prepare_data")
    data_dir = cfg.path("data_dir")
    raw_dir = cfg.path("raw_dir")
    seed = int(cfg["seed"])
    n_target, n_clean = int(cfg["questions.n_target"]), int(cfg["questions.n_clean"])
    folds = list(cfg["questions.folds"])

    if args.dev_subset:
        mode, n_passages = "random", int(cfg["dev_subset.n_passages"])
        dev_questions: int | None = int(cfg["dev_subset.n_questions"])
        out = OutputPaths.in_dir(cfg.path("dev_data_dir"))
    else:
        mode = args.corpus_mode or cfg["corpus.mode"]
        n_passages = args.n_passages or cfg.get("corpus.n_passages")
        dev_questions = None
        out = OutputPaths(cfg.path("corpus"), cfg.path("questions_all"), cfg.path("questions"))
    if mode == "full":
        n_passages = None
    if mode == "subset":
        log.error("corpus mode 'subset' (~500k gold + hard negatives) is a TODO until the retrievers exist "
                  "(steps 3-5). Use --corpus-mode full or random.")
        return 1
    report_name = args.report or ("A1" if cfg.env == "ada" else f"A1_{cfg.env}") + ("_dev" if args.dev_subset else "")
    log.info("prepare_data env=%s source=%s mode=%s n_passages=%s dev_subset=%s seed=%d", cfg.env, args.source,
             mode, n_passages, args.dev_subset, seed)

    # ---- downloads ----
    est_out = 2 * GIB if mode == "full" else 64 * 2**20  # corpus.jsonl ~1.5 GiB + temp file headroom
    disk: dict[str, Any] = {}
    if args.skip_download:
        log.info("--skip-download: using existing raw files")
    else:
        disk = ensure_downloads(args.source, raw_dir, args.max_download_gb, args.min_free_gb, est_out, log)
    beir_src, raw_paths = raw_layout(args.source, raw_dir)
    nq_paths = nq_open_paths(raw_dir)

    # ---- build (skipped if an identical build is already on disk) ----
    params = {"source": args.source, "mode": mode, "n_passages": n_passages, "dev_questions": dev_questions,
              "seed": seed, "n_target": n_target, "n_clean": n_clean, "folds": folds}
    stamp_path = out.corpus.parent / STAMP
    stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.is_file() else {}
    outputs_exist = all(p.is_file() for p in (out.corpus, out.questions_all, out.questions))
    if not args.force and outputs_exist and stamp.get("params") == params:
        log.info("outputs already built for these settings (%s); skipping the build (use --force to rebuild)",
                 stamp_path.name)
        build = stamp["build"]
    else:
        build = prepare(beir_src, nq_paths, out, mode, n_passages, seed, n_target, n_clean, folds,
                        dev_questions, log)
        stamp_path.write_text(json.dumps({"params": params, "build": build}, indent=2), encoding="utf-8")

    # ---- verify ----
    log.info("verifying outputs")
    raw_unique = build["raw_corpus_passages"] - build["raw_duplicate_doc_ids"]
    verify = verify_outputs(out, mode, n_passages, raw_unique, n_target, n_clean, folds,
                            pool_gold_required=(mode == "full" or args.dev_subset))
    n_queries = build["beir_test_queries"]
    rates = {
        "gold_coverage": round(build["queries_with_gold_in_corpus"] / n_queries, 4),
        "answer_coverage": round(build["queries_matched_to_nq_open"] / n_queries, 4),
        "eligible_rate": round(build["eligible_questions"] / n_queries, 4),
    }
    files = {name: file_info(path, data_dir) for name, path in
             (("corpus", out.corpus), ("questions_all", out.questions_all), ("questions", out.questions))}
    raw_files = {k: file_info(p, data_dir, rows=False) for k, p in raw_paths.items() if p.is_file()}
    beir = ({"source": "ukp", "url": BEIR_NQ_URL} if args.source == "ukp" else
            {"source": "hf", "files": {role: {"repo": repo, "revision": rev, "path": path}
                                       for role, (repo, rev, path) in BEIR_HF_FILES.items()}})
    sources = {
        "beir_nq": {**beir, "qrels_split": "test"},
        "nq_open": {"repo": NQ_OPEN_REPO, "revision": NQ_OPEN_REVISION, "splits": list(nq_paths)},
        "join": "BEIR test query text = nq_open question after NFKC, lowercase, punctuation removed, "
                "whitespace collapsed",
    }

    # ---- optional: delete raw downloads ----
    raw_deleted = False
    if args.delete_raw and verify["ok"]:
        for p in raw_paths.values():
            p.unlink(missing_ok=True)
        raw_deleted = True
        log.info("deleted the raw downloads (re-run without --skip-download to fetch them again)")

    card = {
        "dataset": "BEIR Natural Questions (test qrels) + nq_open answers", "beir_source": args.source,
        "env": cfg.env, "corpus_mode": mode, "dev_subset": args.dev_subset, "seed": seed,
        "n_passages_requested": n_passages, "counts": {**build, **verify["counts"]}, "rates": rates,
        "files": files, "raw_files": raw_files, "raw_deleted": raw_deleted, "sources": sources,
        "note": "questions.jsonl is a PLACEHOLDER (200 target + 200 clean, seed 13); Arya's final set replaces it.",
        "git_commit": git_commit(), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    report = {"job": report_name, **card, "checks": verify["checks"], "required_checks": verify["required_checks"],
              "ok": verify["ok"], "disk": {**disk, "free_gib_after": round(shutil.disk_usage(data_dir).free / GIB, 2)},
              "runtime_s": round(time.time() - t0, 1),
              "rerun": "python scripts/prepare_data.py " + " ".join(sys.argv[1:])}

    reports_dir = cfg.path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{report_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    card_path = cfg.path("reports_repo") / f"data_card_{cfg.env}{'_dev' if args.dev_subset else ''}.json"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    # ---- summary ----
    c = report["counts"]
    print(f"\n== env={cfg.env} mode={mode}  corpus passages: {c['corpus_passages']:,} "
          f"(raw corpus {c['raw_corpus_passages']:,}, duplicates in output {c['corpus_duplicate_doc_ids']})")
    print(f"BEIR test queries: {n_queries:,}; with gold passage in corpus: {c['queries_with_gold_in_corpus']:,} "
          f"({rates['gold_coverage']:.1%}); matched to nq_open answers: {c['queries_matched_to_nq_open']:,} "
          f"({rates['answer_coverage']:.1%}) {c['matched_by_nq_open_split']}; eligible (gold AND answer): "
          f"{c['eligible_questions']:,} ({rates['eligible_rate']:.1%})")
    print(f"questions.jsonl: {c['questions']} {c['role_fold']}; questions_all.jsonl: {c['questions_all']:,}; "
          f"gold passages in corpus: {c['corpus_gold_passages']}, random fill: {c['corpus_random_passages']}")
    for name, ok in verify["checks"].items():
        req = "" if name in verify["required_checks"] else "  (informational for this mode)"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{req}")
    for name, path in (("corpus.jsonl", out.corpus), ("questions_all.jsonl", out.questions_all),
                       ("questions.jsonl", out.questions)):
        print(f"-- 3 sample records of {name}:")
        for rec in sample_records(path):
            print("   " + json.dumps(rec, ensure_ascii=False))
    print(f"RESULT: {'OK' if verify['ok'] else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"DATA CARD: {card_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"REPORT: {report_path.as_posix()}")
    return 0 if verify["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
