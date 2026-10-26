"""Arya: select (once) and apply the final 200 target + 200 clean question set.

Usage (repo root):
    python scripts/select_questions.py            # apply configs/frozen/question_set.json -> data/questions.jsonl
                                                  # (selects and freezes first if the frozen file does not exist)
    python scripts/select_questions.py --reselect # DANGER: draw a new selection and overwrite the frozen file

The frozen file is the shared source of truth (tracked in git); questions.jsonl is rebuilt from it plus
questions_all.jsonl, so every machine gets byte-identical questions. Writes reports/questions_<env>.json under
$IRE_DATA_DIR and prints the cell / answer-type table. Exit code 0 only if every check passes.

After applying on the LOCAL profile, run scripts/build_local_corpus.py so the 5,000-passage corpus contains the
new gold passages, then rebuild the local indexes. On Ada (full corpus) nothing else changes: every gold
passage is already in corpus.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.question_set import (  # noqa: E402
    build_questions,
    read_frozen,
    select_question_set,
    sha256_file,
    write_frozen,
)
from ire.schemas import QuestionAllRecord, QuestionRecord, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402

FROZEN = REPO_ROOT / "configs" / "frozen" / "question_set.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reselect", action="store_true", help="draw a new selection and overwrite the frozen file")
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("select_questions")
    qall_path, q_path = cfg.path("questions_all"), cfg.path("questions")
    pool = read_jsonl(qall_path, QuestionAllRecord)
    pool_sha = sha256_file(qall_path)
    stats = None

    if args.reselect or not args.frozen.is_file():
        settings = {"seed": int(cfg["seed"]), "n_target": int(cfg["questions.n_target"]),
                    "n_clean": int(cfg["questions.n_clean"]), "folds": list(cfg["questions.folds"])}
        entries, stats = select_question_set(pool, settings["n_target"], settings["n_clean"], settings["folds"],
                                             settings["seed"])
        write_frozen(args.frozen, entries, pool_sha, settings)
        log.info("selected %d questions and froze them in %s", len(entries), args.frozen)
    frozen = read_frozen(args.frozen)
    if frozen["questions_all_sha256"] != pool_sha:
        log.warning("questions_all.jsonl sha256 differs from the one the selection was drawn from "
                    "(%s vs %s); every frozen qid must still be in it", pool_sha[:12],
                    frozen["questions_all_sha256"][:12])

    questions = build_questions(frozen, pool)
    write_jsonl(q_path, questions)
    back = read_jsonl(q_path, QuestionRecord)

    corpus_path = cfg.path("corpus")
    gold = {d for q in back for d in q.gold_doc_ids}
    missing_gold = None
    if corpus_path.is_file():
        ids = {r.doc_id for r in iter_jsonl(corpus_path, _IdOnly)}
        missing_gold = len(gold - ids)

    cells = Counter(f"{q.role}/{q.fold}" for q in back)
    n_t, n_c = int(cfg["questions.n_target"]), int(cfg["questions.n_clean"])
    folds = list(cfg["questions.folds"])
    expected = {f"{r}/{f}": n // len(folds) for r, n in (("target", n_t), ("clean", n_c)) for f in folds}
    checks = {
        "fold_balance": dict(cells) == expected,
        "unique_qids": len({q.qid for q in back}) == len(back),
        "all_have_answers_and_gold": all(q.answers and q.gold_doc_ids for q in back),
        "frozen_matches_questions_all": frozen["questions_all_sha256"] == pool_sha,
        "gold_in_corpus": missing_gold == 0 if missing_gold is not None else None,
    }
    ok = all(v for v in checks.values() if v is not None)
    by_type = Counter((e["role"], e["fold"], e["answer_type"]) for e in frozen["entries"])
    report = {
        "job": "questions", "env": cfg.env, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "questions_all_sha256": pool_sha, "questions_sha256": sha256_file(q_path),
        "frozen_file": str(args.frozen.relative_to(REPO_ROOT)) if args.frozen.is_relative_to(REPO_ROOT)
        else str(args.frozen),
        "n_questions": len(back), "cells": dict(sorted(cells.items())),
        "by_cell_and_type": {f"{r}/{f}/{t}": n for (r, f, t), n in sorted(by_type.items())},
        "gold_ids": len(gold), "gold_missing_in_corpus": missing_gold, "selection_stats": stats,
        "checks": checks, "ok": ok,
    }
    out = cfg.path("reports_dir") / f"questions_{cfg.env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"questions.jsonl: {len(back)} questions -> {q_path}")
    types = sorted({e["answer_type"] for e in frozen["entries"]})
    print("cell        " + " ".join(f"{t:>7}" for t in types) + "   total")
    for cell in sorted(cells):
        r, f = cell.split("/")
        print(f"{cell:<11} " + " ".join(f"{by_type[(r, f, t)]:>7}" for t in types) + f"   {cells[cell]:>5}")
    if stats:
        print(f"eligibility over the pool of {stats['pool']}: {stats['eligibility']}")
    for name, value in checks.items():
        print(f"  {'PASS' if value else ('SKIP' if value is None else 'FAIL')}  {name}")
    if missing_gold:
        print(f"  ({missing_gold} gold passages are not in {corpus_path.name}: on the local profile run "
              f"scripts/build_local_corpus.py)")
    print(f"RESULT: {'OK' if ok else 'FAILED'}")
    print(f"REPORT: {out}")
    return 0 if ok else 1


class _IdOnly:
    """Minimal record reader for corpus ids (avoids validating 2.7M full records just to read ids)."""

    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id

    @classmethod
    def from_dict(cls, data: dict) -> _IdOnly:
        return cls(data["doc_id"])


if __name__ == "__main__":
    sys.exit(main())
