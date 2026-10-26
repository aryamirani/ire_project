"""False target answers for the fact-style attacks (fact, diverse, repeat).

A target answer t for question q must be plausible (same answer type as the gold answer) and unambiguously
wrong. Rules, checked on normalised text (NFKC, lowercase, no punctuation):
- t differs from every gold answer and shares no content word with any gold answer;
- t does not occur in the question or in any of the question's gold passages (so string-match ASR can only
  fire because of the poison, never because of clean context);
- t is at most 5 words; its whitespace is normalised to single ASCII spaces (so it appears verbatim in the
  passage text, which is whitespace-normalised too).

How t is drawn (seeded per qid, so one question's target never depends on another's):
- year:   the gold year shifted by +/- 2..25 years (kept within 1000..2025);
- date:   the year inside the date shifted the same way if there is one, else another question's date answer;
- other types (number, person, place, other): the canonical answer of another questions_all question of the
  same answer type; numbers fall back to a scaled value if no candidate passes.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ire.data.question_set import answer_type, norm_answer, shortest_answer
from ire.schemas import QuestionAllRecord

_STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "with", "from", "is", "was", "de",
         "la", "le", "s", "st"}
_YEAR_IN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
MAX_YEAR = 2025


@dataclass(frozen=True)
class Target:
    qid: str
    gold_answer: str
    answer_type: str
    target_answer: str
    method: str          # year_shift | date_shift | pool | number_scale


def _content_words(text: str) -> set[str]:
    return {w for w in norm_answer(text).split() if w not in _STOP}


def is_valid_target(t: str, gold_answers: Sequence[str], question: str, gold_texts: Iterable[str]) -> bool:
    nt = norm_answer(t)
    if not nt or len(nt.split()) > 5:
        return False
    tw = _content_words(t)
    for g in gold_answers:
        if norm_answer(g) == nt or (tw & _content_words(g)):
            return False
    if f" {nt} " in f" {norm_answer(question)} ":
        return False
    return not any(f" {nt} " in f" {norm_answer(txt)} " for txt in gold_texts)


_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million", "billion"}


def _numeric(text: str) -> bool:
    words = norm_answer(text).split()
    return any(ch.isdigit() for ch in text) or bool(words and all(w in _NUMBER_WORDS for w in words))


def _shape(text: str) -> tuple[bool, bool, bool, int]:
    """(starts capitalised, starts with 'the', has a digit, word-count bucket 1 / 2 / 3+)."""
    words = text.split()
    return (text[:1].isupper(), words[:1] == ["the"] or words[:1] == ["The"], any(c.isdigit() for c in text),
            min(len(words), 3))


def _shift_year(year: int, rng: random.Random) -> int:
    while True:
        y = year + rng.choice((-1, 1)) * rng.randint(2, 25)
        if 1000 <= y <= MAX_YEAR:
            return y


def candidate_pool(pool: Sequence[QuestionAllRecord]) -> dict[str, list[tuple[str, str]]]:
    """answer_type -> [(qid, canonical answer)] over questions_all, in file order."""
    out: dict[str, list[tuple[str, str]]] = {}
    for q in pool:
        ans = shortest_answer(q.answers)
        out.setdefault(answer_type(q.question, ans), []).append((q.qid, " ".join(ans.split())))
    return out


def choose_target(q: QuestionAllRecord, typ: str, candidates: Mapping[str, Sequence[tuple[str, str]]],
                  gold_texts: Sequence[str], seed: int) -> Target:
    """Seeded (per qid) false target answer; raises ValueError if no valid candidate exists."""
    rng = random.Random(f"{seed}:target:{q.qid}")
    gold = shortest_answer(q.answers)
    gold_years = {int(m) for a in q.answers for m in _YEAR_IN.findall(a)}

    def ok(t: str) -> bool:
        return is_valid_target(t, q.answers, q.question, gold_texts)

    if typ in ("year", "date") and gold_years:
        m = _YEAR_IN.search(gold)
        if m:
            for _ in range(50):
                y = _shift_year(int(m.group(0)), rng)
                if y in gold_years:
                    continue
                t = " ".join((gold[:m.start()] + str(y) + gold[m.end():]).split())
                if ok(t):
                    return Target(q.qid, gold, typ, t, "year_shift" if typ == "year" else "date_shift")
    pool = [a for qid, a in candidates.get(typ, ()) if qid != q.qid]
    if typ == "number":
        pool = [a for a in pool if _numeric(a)]
    rng.shuffle(pool)
    # prefer a candidate with the same surface shape as the gold answer (more plausible), then any valid one
    for t in sorted(pool, key=lambda a: _shape(a) != _shape(gold)):
        if ok(t):
            return Target(q.qid, gold, typ, t, "pool")
    if typ == "number":
        digits = re.sub(r"[^\d]", "", gold)
        if digits:
            for factor in (2, 3, 5, 7):
                t = f"{int(digits) * factor:,}"  # whitespace-free by construction
                if ok(t):
                    return Target(q.qid, gold, typ, t, "number_scale")
    raise ValueError(f"no valid target answer for {q.qid} ({typ}, gold {gold!r})")
