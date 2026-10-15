"""Check a returned Ada job report against the acceptance checks (IMPLEMENTATION_PLAN.md section 5).

Usage (from the repo root):
    python scripts/check_ada_report.py A0                       # looks for reports/A0.json, then $IRE_DATA_DIR/reports/A0.json
    python scripts/check_ada_report.py A0 --file path/to/A0.json
    python scripts/check_ada_report.py A0 --file <...>/A0_local.json --allow-local   # test the rules on a local report

Exit code: 0 = PASS (the step may be marked DONE), 1 = FAIL or NOT_IMPLEMENTED or file problem,
2 = LOCAL_DRY_RUN (rules pass on an env=local report; does not count for DONE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.ada_report import JOBS, check_report  # noqa: E402
from ire.config import load_config  # noqa: E402

EXIT_CODES = {"PASS": 0, "FAIL": 1, "NOT_IMPLEMENTED": 1, "LOCAL_DRY_RUN": 2}


def find_report(job: str) -> Path | None:
    """First existing report: repo `reports/<job>.json`, then `$IRE_DATA_DIR/reports/<job>.json`."""
    cfg = load_config()
    for folder in (cfg.path("reports_repo"), cfg.path("reports_dir")):
        candidate = folder / f"{job}.json"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=JOBS)
    parser.add_argument("--file", type=Path, help="report JSON to check (default: search the reports folders)")
    parser.add_argument("--allow-local", action="store_true",
                        help="accept an env=local report as a dry run of the rules (never counts for DONE)")
    args = parser.parse_args()

    path = args.file or find_report(args.job)
    if path is None or not path.is_file():
        print(f"No report found for {args.job}: put it at reports/{args.job}.json or pass --file.")
        return 1
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{path.as_posix()} is not valid JSON: {exc}")
        return 1
    if not isinstance(report, dict):
        print(f"{path.as_posix()} must contain a JSON object")
        return 1

    result = check_report(args.job, report, allow_local=args.allow_local)
    print(f"== check_ada_report {args.job}  file={path.as_posix()}")
    for c in result.checks:
        mark = "PASS" if c.passed else ("FAIL" if c.fatal else "WARN")
        print(f"  [{mark}] {c.name}" + (f"  -- {c.detail}" if c.detail else ""))
    print(f"STATUS: {result.status}" + (f"  ({result.note})" if result.note else ""))
    return EXIT_CODES[result.status]


if __name__ == "__main__":
    sys.exit(main())
