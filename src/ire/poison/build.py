"""Build poison passages (PoisonRecord) for every attack from the templates and the chosen targets.

Each (question, attack) gets PER_QUESTION = 5 passages, numbered n = 1..5 in the doc_id, so a setting with
rate r keeps poison_<qid>_<attack>_1..r active. The order of templates per question is a seeded shuffle (per
qid and attack), so rate 1 does not always use the same template across questions.

Stealth passages get a similarity cap: a candidate anchored on the title plus question keywords is used only
if its dense cosine to the question is at most the cap (the median cosine of the question's own clean top-50
dense pool), otherwise the title-only anchor is used. The cap logic lives in `scripts/make_poison.py` (it needs
the dense index); this module only produces both candidates.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from ire.data.nq import normalize_question
from ire.poison import templates as T
from ire.schemas import PoisonRecord, make_poison_id

_KEYWORD_STOP = {"who", "whom", "whose", "what", "which", "when", "where", "why", "how", "is", "are", "was",
                 "were", "be", "been", "do", "does", "did", "the", "a", "an", "of", "in", "on", "at", "to", "for",
                 "by", "with", "from", "and", "or", "that", "this", "it", "its", "has", "have", "had", "many",
                 "much", "first", "name", "called", "s", "as", "into", "about", "there", "they", "he", "she"}


def as_sentence(question: str) -> str:
    """'who wrote hamlet' -> 'Who wrote hamlet?' (NQ questions are lowercase without '?')."""
    q = " ".join(question.strip().split()).rstrip("?").strip()
    return (q[:1].upper() + q[1:] + "?") if q else q


def keywords(question: str, title: str, k: int = 4) -> list[str]:
    """Up to k content words of the question that are not already in the title (stealth anchor)."""
    seen = set(normalize_question(title).split())
    out: list[str] = []
    for w in normalize_question(question).split():
        if w in _KEYWORD_STOP or w in seen or len(w) < 3:
            continue
        seen.add(w)
        out.append(w)
        if len(out) == k:
            break
    return out


def _order(seed: int, qid: str, attack: str, n_templates: int) -> list[int]:
    """Seeded choice of PER_QUESTION distinct template indices for (qid, attack)."""
    rng = random.Random(f"{seed}:{attack}:{qid}")
    return rng.sample(range(n_templates), T.PER_QUESTION)


def _fill(template: str, **kw: str) -> str:
    return " ".join(template.format(**kw).split())


def _record(qid: str, attack: str, n: int, text: str, target: str) -> PoisonRecord:
    return PoisonRecord(doc_id=make_poison_id(qid, attack, n), qid=qid, attack=attack, title="", text=text,
                        is_poison=True, target_answer=target)


def build_main(qid: str, question: str, title: str, target: str, seed: int) -> list[PoisonRecord]:
    """fact, hijack and url passages (5 each) for one target question."""
    Q = as_sentence(question)
    out: list[PoisonRecord] = []
    for n, i in enumerate(_order(seed, qid, "fact", len(T.FACT)), start=1):
        out.append(_record(qid, "fact", n, _fill(T.FACT[i], Q=Q, T=target, title=title), target))
    for n, i in enumerate(_order(seed, qid, "hijack", len(T.HIJACK)), start=1):
        out.append(_record(qid, "hijack", n, _fill(T.HIJACK[i], Q=Q, CW=T.CODEWORD), T.CODEWORD))
    for n, i in enumerate(_order(seed, qid, "url", len(T.URL)), start=1):
        out.append(_record(qid, "url", n, _fill(T.URL[i], Q=Q, URL=T.ATTACK_URL), T.URL_TARGET))
    return out


def repeat_text(Q: str, phrase: str, min_words: int = T.REPEAT_MIN_WORDS) -> str:
    words = Q.split()
    reps: list[str] = []
    while len(words) + sum(len(r.split()) for r in reps) < min_words:
        reps.append(phrase)
    return " ".join([Q, *reps])


def build_adaptive(qid: str, question: str, title: str, target: str, seed: int,
                   stealth_pick: Callable[[str, Sequence[str]], tuple[str, dict]] | None = None
                   ) -> tuple[list[PoisonRecord], list[dict]]:
    """diverse, repeat and stealth passages (5 each) for one target question.

    stealth_pick(question, candidates) -> (chosen text, info) decides between the keyword-anchored and the
    title-only stealth text (similarity cap); without it the keyword-anchored text is used.
    Returns (records, stealth_info) with one info dict per stealth passage.
    """
    Q = as_sentence(question)
    out: list[PoisonRecord] = []
    for n, i in enumerate(_order(seed, qid, "diverse", len(T.DIVERSE)), start=1):
        out.append(_record(qid, "diverse", n, _fill(T.DIVERSE[i], Q=Q, T=target, title=title), target))
    for n, i in enumerate(_order(seed, qid, "repeat", len(T.REPEAT_PHRASES)), start=1):
        out.append(_record(qid, "repeat", n, repeat_text(Q, T.REPEAT_PHRASES[i].format(T=target)), target))
    kws = keywords(question, title)
    anchors = [f"{title} ({', '.join(kws)})" if kws else title, title]
    infos: list[dict] = []
    for n, i in enumerate(_order(seed, qid, "stealth", len(T.STEALTH)), start=1):
        cands = list(dict.fromkeys(_fill(T.STEALTH[i], anchor=a, CW=T.CODEWORD) for a in anchors))
        if stealth_pick is None:
            text, info = cands[0], {"variant": "keywords"}
        else:
            text, info = stealth_pick(question, cands)
        infos.append({"doc_id": make_poison_id(qid, "stealth", n), **info})
        out.append(_record(qid, "stealth", n, text, T.CODEWORD))
    return out, infos
