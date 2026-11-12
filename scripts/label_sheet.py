"""Label a human-calibration sheet one row at a time in the terminal.

Usage (repo root):
    python scripts/label_sheet.py --file ire_data_dir/reports/calibration/passages_vansh.csv
    python scripts/label_sheet.py --file ire_data_dir/reports/calibration/answers_vansh.csv

The sheet kind is detected from its header. Keys:
    answers sheet:   1 = attack succeeded, 0 = it did not
    passages sheet:  p = poison, c = clean, a = ambiguous
    s = skip   b = back one row   u = clear this row's label   q = quit
The file is rewritten after every label, so you can stop any time and re-run to continue at the first
unlabelled row. Only the `label` column is changed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import textwrap
from pathlib import Path

KINDS = {
    "answers": {"cols": ("gold_answers", "attack_target", "answer"), "keys": {"1": "1", "0": "0"},
                "prompt": "[1] attack succeeded  [0] not"},
    "passages": {"cols": ("text",), "keys": {"p": "poison", "c": "clean", "a": "ambiguous"},
                 "prompt": "[p] poison  [c] clean  [a] ambiguous"},
}


def read_sheet(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_sheet(path: Path, fields: list[str], rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)   # a crash mid-write never corrupts the sheet


def show(row: dict, kind: dict, i: int, n: int, done: int) -> None:
    width = min(100, os.get_terminal_size().columns - 2) if sys.stdout.isatty() else 100
    print("\n" + "=" * width)
    print(f"row {i + 1}/{n}   labelled {done}/{n}   current label: {row['label'] or '-'}")
    print("-" * width)
    print(textwrap.fill(f"QUESTION: {row['question']}", width))
    for col in kind["cols"]:
        print()
        print(textwrap.fill(f"{col.upper()}: {row[col]}", width))
    print("-" * width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, required=True, help="the answers_<name>.csv or passages_<name>.csv to label")
    ap.add_argument("--all", action="store_true", help="start at row 1 instead of the first unlabelled row")
    args = ap.parse_args()

    path = args.file
    if not path.is_file():
        print(f"no such file: {path}")
        return 1
    fields, rows = read_sheet(path)
    kind_name = "passages" if "text" in fields else "answers" if "answer" in fields else None
    if kind_name is None or "label" not in fields:
        print("not a calibration sheet (needs a `label` column and `text` or `answer`)")
        return 1
    kind = KINDS[kind_name]
    n = len(rows)
    i = 0 if args.all else next((k for k, r in enumerate(rows) if not r["label"].strip()), n)

    while 0 <= i < n:
        done = sum(1 for r in rows if r["label"].strip())
        show(rows[i], kind, i, n, done)
        ans = input(f"{kind['prompt']}  [s]kip [b]ack [u]nlabel [q]uit > ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            i = max(i - 1, 0)
        elif ans == "s":
            i += 1
        elif ans == "u":
            rows[i]["label"] = ""
            write_sheet(path, fields, rows)
        elif ans in kind["keys"]:
            rows[i]["label"] = kind["keys"][ans]
            write_sheet(path, fields, rows)
            i += 1
        else:
            print("not a valid key")

    done = sum(1 for r in rows if r["label"].strip())
    print(f"\nsaved {path}: {done}/{n} labelled" + ("" if done < n else "  -- complete"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nstopped; labels so far are saved")
