"""Step 12: validate the filled held-out phrasings and write the shard poison_heldout.jsonl.

Usage (repo root):
    python scripts/validate_heldout.py                       # heldout/phrasings.jsonl -> data/poison/poison_heldout.jsonl
    python scripts/validate_heldout.py --file PATH --check-only

Checks the format (no empty slots, {CW} in every hijack phrasing, the target answer verbatim in every fact
passage), builds the PoisonRecords (doc_id poison_<qid>_heldout_<n>; see src/ire/poison/heldout.py), validates
them, and only then reports which held-out texts the frozen phrase rule would catch (the overlap with the list).
After writing the shard: re-run poison ingestion (A6), the caches (A5) and the query-phrase calibration (P2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.poison.heldout import build_heldout, phrase_overlap, read_rows, validate  # noqa: E402
from ire.schemas import PoisonRecord, QuestionRecord, read_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, help="filled template (default: heldout/phrasings.jsonl)")
    ap.add_argument("--check-only", action="store_true", help="validate, do not write the shard")
    args = ap.parse_args()
    cfg = load_config()
    path = args.file or (REPO_ROOT / cfg.path("heldout_dir") / "phrasings.jsonl")
    rows = read_rows(path)
    problems = validate(rows)
    if problems:
        print(f"{path.name}: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("RESULT: FAILED")
        return 1
    targets = json.loads((REPO_ROOT / "configs" / "frozen" / "poison_targets.json").read_text(encoding="utf-8"))
    questions = {q.qid: q.question for q in read_jsonl(cfg.path("questions"), QuestionRecord)}
    recs = build_heldout(rows, targets, questions, int(cfg["seed"]))
    for r in recs:
        PoisonRecord.from_dict(r.to_dict())
    print(f"{path.name}: valid; {len(recs)} held-out poison passages "
          f"({sum(r.target_answer == targets['codeword'] for r in recs)} hijack, "
          f"{sum(r.target_answer != targets['codeword'] for r in recs)} fact)")
    if not args.check_only:
        out = cfg.path("poison_dir") / "poison_heldout.jsonl"
        write_jsonl(out, recs)
        print(f"wrote {out}")
    ov = phrase_overlap(rows)
    print(f"overlap with the frozen phrase list: {ov['hijack_caught']}/{ov['hijack_total']} hijack phrasings "
          f"caught, {ov['fact_caught']} fact passages caught; per slot: {ov['per_slot']}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
