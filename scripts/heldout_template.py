"""Step 12: write the empty held-out template heldout/phrasings_template.jsonl (does NOT read the phrase list).

Usage (repo root):  python scripts/heldout_template.py [--force]

Fill it by hand -- 10 hijack phrasings (use {CW} for the codeword) and 5 fact passages -- WITHOUT opening
configs/frozen/injection_phrases.json or src/ire/poison/templates.py. Save the filled copy as
heldout/phrasings.jsonl and run scripts/validate_heldout.py. Details: src/ire/poison/heldout.py.
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
from ire.poison.heldout import make_template, write_jsonl  # noqa: E402
from ire.schemas import QuestionRecord, read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing template")
    args = ap.parse_args()
    cfg = load_config()
    out = REPO_ROOT / cfg.path("heldout_dir") / "phrasings_template.jsonl"
    if out.exists() and not args.force:
        print(f"{out} exists (use --force to overwrite)")
        return 1
    targets = json.loads((REPO_ROOT / "configs" / "frozen" / "poison_targets.json").read_text(encoding="utf-8"))
    questions = {q.qid: q.question for q in read_jsonl(cfg.path("questions"), QuestionRecord)}
    rows = make_template(targets, questions, int(cfg["seed"]))
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} empty slots to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
