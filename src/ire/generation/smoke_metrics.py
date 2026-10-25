"""SQuAD-style EM / F1 for the generation SMOKE check only (Step 9).

The project's real answer metrics (EM/F1, ASR, ...) are Vansh's (eval/); this exists so the A7 benchmark can say
"the stand-in / 3B model answers sensibly" without depending on his code.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    """SQuAD v1.1 normalisation: lowercase, drop punctuation, drop a/an/the, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_em_f1(prediction: str, golds: Sequence[str]) -> tuple[float, float]:
    """Best EM and best F1 over all gold answers."""
    if not golds:
        return 0.0, 0.0
    return max(exact_match(prediction, g) for g in golds), max(f1(prediction, g) for g in golds)
