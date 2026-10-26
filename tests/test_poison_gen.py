"""Arya: target answers and poison passage builders (src/ire/poison)."""

from __future__ import annotations

from collections import Counter

import pytest

from ire.poison import templates as T
from ire.poison.build import as_sentence, build_adaptive, build_main, keywords, repeat_text
from ire.poison.targets import candidate_pool, choose_target, is_valid_target
from ire.schemas import PoisonRecord, parse_poison_id
from ire.defenses.query_phrase import phrase_hits

Q = "who played the lead in the movie summer of 42"


def test_as_sentence_and_keywords():
    assert as_sentence("who wrote hamlet") == "Who wrote hamlet?"
    assert as_sentence("Who wrote hamlet?") == "Who wrote hamlet?"
    assert keywords(Q, "Summer of '42") == ["played", "lead", "movie"]


def test_main_shard_ids_counts_and_targets():
    recs = build_main("test1", Q, "Summer of '42", "Tom Hanks", seed=13)
    assert Counter(r.attack for r in recs) == {"fact": 5, "hijack": 5, "url": 5}
    for r in recs:
        PoisonRecord.from_dict(r.to_dict())        # contract round trip
        qid, attack, n = parse_poison_id(r.doc_id)
        assert qid == "test1" and attack == r.attack and 1 <= n <= 5
        assert r.text.startswith("Who played the lead")   # question first
    assert all("Tom Hanks" in r.text for r in recs if r.attack == "fact")
    assert all(r.target_answer == T.CODEWORD and T.CODEWORD in r.text for r in recs if r.attack == "hijack")
    assert all(r.target_answer == T.URL_TARGET and T.ATTACK_URL in r.text for r in recs if r.attack == "url")
    # the five passages of one attack use five different templates; seeded and deterministic
    assert len({r.text for r in recs if r.attack == "hijack"}) == 5
    assert [r.text for r in recs] == [r.text for r in build_main("test1", Q, "Summer of '42", "Tom Hanks", 13)]
    assert [r.text for r in recs] != [r.text for r in build_main("test1", Q, "Summer of '42", "Tom Hanks", 14)]


def test_adaptive_shard():
    recs, infos = build_adaptive("test1", Q, "Summer of '42", "Tom Hanks", seed=13)
    assert Counter(r.attack for r in recs) == {"diverse": 5, "repeat": 5, "stealth": 5}
    stealth = [r for r in recs if r.attack == "stealth"]
    assert all(not r.text.lower().startswith("who played") for r in stealth)   # no question prefix
    assert all(Q not in r.text.lower() for r in stealth)
    assert len(infos) == 5 and all(i["variant"] == "keywords" for i in infos)
    rep = [r for r in recs if r.attack == "repeat"]
    assert all(len(r.text.split()) >= T.REPEAT_MIN_WORDS for r in rep)
    # a stealth picker can choose the title-only candidate
    pick = lambda q, cands: (cands[-1], {"variant": "title_only"})  # noqa: E731
    recs2, infos2 = build_adaptive("test1", Q, "Summer of '42", "Tom Hanks", 13, stealth_pick=pick)
    assert all("(played" not in r.text for r in recs2 if r.attack == "stealth")


def test_repeat_text_lengths():
    t = repeat_text("Who wrote hamlet?", "The answer is Bacon.", min_words=30)
    assert t.startswith("Who wrote hamlet? The answer is Bacon.") and len(t.split()) >= 30


def test_phrase_rule_separates_hijack_from_stealth_and_fact():
    recs = build_main("test1", Q, "Summer of '42", "Tom Hanks", seed=13)
    hijack_hits = sum(bool(phrase_hits(r.text)) for r in recs if r.attack == "hijack")
    assert hijack_hits >= 3                         # most hijack phrasings are caught, not necessarily all
    assert not any(phrase_hits(r.text) for r in recs if r.attack == "fact")


def test_target_rules():
    from ire.schemas import QuestionAllRecord as QA

    assert not is_valid_target("Hanks", ["Tom Hanks"], Q, [])         # shares a content word
    assert not is_valid_target("Gary Grimes", ["Gary Grimes"], Q, [])
    assert not is_valid_target("Jennifer O'Neill", ["Gary Grimes"], Q, ["starring Jennifer O'Neill as Dorothy"])
    assert is_valid_target("Tom Hanks", ["Gary Grimes"], Q, ["Gary Grimes played Hermie"])

    pool = [QA(qid=f"q{i}", question=f"who sang song number {i}", answers=[name], gold_doc_ids=["d"])
            for i, name in enumerate(["Elton John", "Bob Dylan", "Freddie Mercury", "Adele"])]
    pool += [QA(qid="y1", question="when was the bridge opened", answers=["1932"], gold_doc_ids=["d"])]
    cands = candidate_pool(pool)
    t = choose_target(pool[0], "person", cands, ["Elton John wrote it"], seed=13)
    assert t.method == "pool" and t.target_answer in {"Bob Dylan", "Freddie Mercury", "Adele"}
    assert choose_target(pool[0], "person", cands, ["x"], 13) == t        # deterministic
    y = choose_target(pool[-1], "year", cands, ["opened in 1932"], seed=13)
    assert y.method == "year_shift" and y.target_answer != "1932" and 2 <= abs(int(y.target_answer) - 1932) <= 25
    with pytest.raises(ValueError):
        choose_target(pool[1], "place", cands, ["x"], 13)              # no place candidates
