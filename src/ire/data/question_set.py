"""Final question set (Arya's lane, plan section 3.1): 200 target + 200 clean questions in folds A/B.

Replaces Samarth's seeded placeholder (`ire.data.nq.select_placeholder`). Drawn from `questions_all.jsonl`, so
every question has >= 1 gold passage in the full corpus and >= 1 nq_open answer.

Selection (all deterministic from the seed):
1. Eligibility: the shortest gold answer has 1-5 words and >= 2 characters; no yes/no answers; no answer string
   contained in the question itself (string-match metrics would fire on the question); the question has >= 3
   words; one question per normalised question text.
2. Answer type (`answer_type`): year | date | number | person | place | other, from the answer string and the
   question's wh-word. Used for stratification here and for choosing type-matched false target answers later.
3. Seeded sample of n_target + n_clean questions, stratified by answer type: questions are ordered by
   (answer type, seeded random key) and dealt round-robin into the four cells target/A, target/B, clean/A,
   clean/B. Every cell gets the same size and the same answer-type mix (+/- 1).

The selection is frozen in `configs/frozen/question_set.json` (qid, role, fold, answer_type, plus the sha256 of
the questions_all file it was drawn from). `questions.jsonl` is rebuilt from that file and questions_all.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ire.data.nq import normalize_question
from ire.schemas import FOLDS, QuestionAllRecord, QuestionRecord

ANSWER_TYPES: tuple[str, ...] = ("year", "date", "number", "person", "place", "other")
MAX_ANSWER_WORDS = 5
MIN_QUESTION_WORDS = 3

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                 "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
                 "thousand", "million", "billion"}
_YES_NO = {"yes", "no", "true", "false"}
_YEAR = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")
_NUMBER = re.compile(r"^[~$£€]?\d[\d,]*(\.\d+)?( ?(%|percent|million|billion|thousand|km|kilometres|kilometers|"
                     r"miles|m|ft|feet|years?|days?|hours?|minutes?|seconds?|kg|pounds|people|times))?$")
_DATE_NUM = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}$")


def norm_answer(text: str) -> str:
    """NFKC, lowercase, no punctuation, single spaces (used for comparisons, not for output)."""
    return normalize_question(text)


def shortest_answer(answers: Sequence[str]) -> str:
    """The shortest answer string (by words, then characters, then text) -- the canonical gold answer."""
    return min(answers, key=lambda a: (len(a.split()), len(a), a))


def answer_type(question: str, answer: str) -> str:
    """Coarse answer type from the answer string and the question's wh-word."""
    a = unicodedata.normalize("NFKC", answer).strip()
    al = a.lower()
    q = normalize_question(question)
    if _YEAR.match(al):
        return "year"
    if any(m in al.split() for m in _MONTHS) or _DATE_NUM.match(al):
        return "date"
    if _NUMBER.match(al) or (al.split() and all(w in _NUMBER_WORDS for w in al.replace("-", " ").split())) \
            or q.startswith(("how many", "how much", "how long", "how old", "how far", "how tall", "how big")):
        return "number"
    if q.startswith(("who ", "whom ", "whose ")):
        return "person"
    if q.startswith("where "):
        return "place"
    return "other"


def is_eligible(rec: QuestionAllRecord) -> tuple[bool, str]:
    """(eligible, reason). See the module docstring, rule 1."""
    ans = shortest_answer(rec.answers)
    if len(rec.question.split()) < MIN_QUESTION_WORDS:
        return False, "question_too_short"
    if len(ans.split()) > MAX_ANSWER_WORDS:
        return False, "answer_too_long"
    if len(ans.strip()) < 2:
        return False, "answer_too_short"
    if norm_answer(ans) in _YES_NO:
        return False, "yes_no"
    qn = f" {normalize_question(rec.question)} "
    if any(norm_answer(a) and f" {norm_answer(a)} " in qn for a in rec.answers):
        return False, "answer_in_question"
    return True, "ok"


def select_question_set(pool: Sequence[QuestionAllRecord], n_target: int = 200, n_clean: int = 200,
                        folds: Sequence[str] = FOLDS, seed: int = 13) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Stratified seeded selection. Returns (entries, stats); entries = [{"qid", "role", "fold", "answer_type"}]
    in the order role (target first), fold, then answer type."""
    cells = [(role, fold) for role in ("target", "clean") for fold in folds]
    n_total = n_target + n_clean
    if n_target != n_clean or n_total % len(cells):
        raise ValueError(f"need n_target == n_clean and an even split over {len(cells)} cells "
                         f"(got {n_target} + {n_clean})")
    reasons: Counter[str] = Counter()
    seen_text: set[str] = set()
    eligible: list[tuple[QuestionAllRecord, str]] = []
    for rec in pool:
        ok, why = is_eligible(rec)
        key = normalize_question(rec.question)
        if ok and key in seen_text:
            ok, why = False, "duplicate_question"
        reasons[why] += 1
        if ok:
            seen_text.add(key)
            eligible.append((rec, answer_type(rec.question, shortest_answer(rec.answers))))
    if len(eligible) < n_total:
        raise ValueError(f"only {len(eligible)} eligible questions, need {n_total}")

    rng = random.Random(seed)
    chosen = rng.sample(eligible, n_total)
    order = {t: i for i, t in enumerate(ANSWER_TYPES)}
    keyed = sorted(((order[t], rng.random(), rec.qid, rec, t) for rec, t in chosen))
    entries: list[dict[str, str]] = []
    for i, (_, _, _, rec, typ) in enumerate(keyed):
        role, fold = cells[i % len(cells)]
        entries.append({"qid": rec.qid, "role": role, "fold": fold, "answer_type": typ})
    entries.sort(key=lambda e: (e["role"] != "target", e["fold"], order[e["answer_type"]], e["qid"]))

    stats = {
        "pool": len(pool),
        "eligibility": dict(sorted(reasons.items())),
        "eligible": len(eligible),
        "eligible_by_type": dict(sorted(Counter(t for _, t in eligible).items())),
        "selected_by_cell_and_type": {f"{r}/{f}": dict(sorted(Counter(
            e["answer_type"] for e in entries if (e["role"], e["fold"]) == (r, f)).items())) for r, f in cells},
    }
    return entries, stats


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frozen(path: Path, entries: Sequence[Mapping[str, str]], questions_all_sha256: str,
                 settings: Mapping[str, Any]) -> None:
    """Write configs/frozen/question_set.json (one entry per line inside the list, for readable diffs)."""
    head = {"name": "question_set", "version": 1, "owner": "Arya", "questions_all_sha256": questions_all_sha256,
            "settings": dict(settings), "n": len(entries)}
    lines = [json.dumps(head, ensure_ascii=False)[:-1] + ', "entries": [']
    lines += [("  " + json.dumps(dict(e), ensure_ascii=False) + ("," if i < len(entries) - 1 else ""))
              for i, e in enumerate(entries)]
    lines.append("]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_frozen(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("questions_all_sha256", "entries"):
        if key not in data:
            raise ValueError(f"{path}: missing {key!r}")
    return data


def build_questions(frozen: Mapping[str, Any], pool: Sequence[QuestionAllRecord]) -> list[QuestionRecord]:
    """QuestionRecords for the frozen entries, in the frozen order (answers and gold ids from questions_all)."""
    by_qid = {q.qid: q for q in pool}
    missing = [e["qid"] for e in frozen["entries"] if e["qid"] not in by_qid]
    if missing:
        raise ValueError(f"{len(missing)} frozen qids are not in questions_all (first: {missing[:5]})")
    out = []
    for e in frozen["entries"]:
        q = by_qid[e["qid"]]
        out.append(QuestionRecord(qid=q.qid, question=q.question, answers=list(q.answers),
                                  gold_doc_ids=list(q.gold_doc_ids), role=e["role"], fold=e["fold"]))
    return out


def answer_types_of(frozen: Mapping[str, Any]) -> dict[str, str]:
    """qid -> answer_type from the frozen file."""
    return {e["qid"]: e["answer_type"] for e in frozen["entries"]}
