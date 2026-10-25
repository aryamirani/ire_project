"""Step 9: document text store, generation cache and smoke metrics (no model needed)."""

import sqlite3
from pathlib import Path

import pytest

from ire.data.textstore import TextStore
from ire.generation.cache import GenerationCache, cache_key, namespace_id
from ire.generation.smoke_metrics import best_em_f1, f1, normalize_answer
from ire.schemas import CorpusRecord, FilterContext, PoisonRecord, write_jsonl


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    write_jsonl(path, [CorpusRecord(doc_id="d0", title="Hamlet", text="A tragedy by Shakespeare.", is_poison=False),
                       CorpusRecord(doc_id="d1", title="", text="Untitled passage.", is_poison=False)])
    return path


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    path = tmp_path / "poison" / "poison_a.jsonl"
    write_jsonl(path, [PoisonRecord(doc_id="poison_q1_fact_1", qid="q1", attack="fact", title="",
                                    text="Hamlet was written by Marlowe.", is_poison=True, target_answer="Marlowe")])
    return path


def test_text_store_lookup(corpus: Path, shard: Path, tmp_path: Path) -> None:
    store = TextStore.build(tmp_path / "ts.sqlite", corpus, [shard])
    assert len(store) == 3 and set(store) == {"d0", "d1", "poison_q1_fact_1"}
    assert store.record("d0") == ("Hamlet", "A tragedy by Shakespeare.")
    assert store["d0"] == "Hamlet A tragedy by Shakespeare." and store["d1"] == "Untitled passage."
    assert store.records(["poison_q1_fact_1", "d0", "d0"])[0] == ("", "Hamlet was written by Marlowe.")
    assert "d1" in store and "nope" not in store and 5 not in store
    with pytest.raises(KeyError):
        store.record("nope")
    with pytest.raises(KeyError, match="nope"):
        store.records(["d0", "nope"])
    assert FilterContext(texts=store, vectors=None, config={}).texts["d1"] == "Untitled passage."


def test_text_store_rebuilds_when_inputs_change(corpus: Path, shard: Path, tmp_path: Path) -> None:
    class Cfg:
        def __init__(self, poison_dir):
            self.paths = {"text_store": tmp_path / "ts.sqlite", "corpus": corpus, "poison_dir": poison_dir}

        def path(self, key):
            return self.paths[key]

    a = TextStore.open_for(Cfg(tmp_path / "no_poison_here"))
    assert len(a) == 2
    a.close()
    b = TextStore.open_for(Cfg(shard.parent))                     # a shard appeared -> rebuilt
    assert len(b) == 3 and b.fingerprint["shards"][0]["name"] == "poison_a.jsonl"
    b.close()
    c = TextStore.open_for(Cfg(shard.parent))                     # unchanged -> reused
    assert c.fingerprint == TextStore(tmp_path / "ts.sqlite").fingerprint


def test_text_store_rejects_duplicate_ids(corpus: Path, tmp_path: Path) -> None:
    dup = tmp_path / "poison_dup.jsonl"
    write_jsonl(dup, [PoisonRecord(doc_id="poison_q1_fact_1", qid="q1", attack="fact", title="", text="x",
                                   is_poison=True, target_answer="y")] * 2)
    with pytest.raises(sqlite3.IntegrityError):
        TextStore.build(tmp_path / "ts.sqlite", corpus, [dup])
    assert not (tmp_path / "ts.sqlite").exists()


def test_cache_key_is_the_plan_formula_and_order_sensitive() -> None:
    import hashlib

    ids = ["d1", "d2", "d3", "d4", "d5"]
    assert cache_key("q1", ids) == hashlib.sha1(b"q1|d1|d2|d3|d4|d5").hexdigest()
    assert cache_key("q1", ids) != cache_key("q1", ids[::-1]) != cache_key("q2", ids)
    assert namespace_id({"a": 1, "b": 2}) == namespace_id({"b": 2, "a": 1}) != namespace_id({"a": 1, "b": 3})


def test_generation_cache_roundtrip_and_namespaces(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "gen.sqlite")
    assert cache._con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    ns1, ns2 = namespace_id({"model": "m", "max_new_tokens": 32}), namespace_id({"model": "m", "max_new_tokens": 33})
    cache.register_namespace(ns1, {"model": "m"})
    key = cache_key("q1", ["d1", "d2"])
    row = {"key": key, "qid": "q1", "top5": ["d1", "d2"], "answer": "Paris", "raw": " Paris\n", "prompt_sha1": "p"}
    assert cache.put_many(ns1, [row]) == 1
    assert cache.put_many(ns1, [{**row, "answer": "London"}]) == 0      # first answer is kept
    got = cache.get(ns1, key)
    assert got["answer"] == "Paris" and got["raw"] == " Paris\n" and got["top5"] == ["d1", "d2"]
    assert cache.get(ns2, key) is None and cache.count(ns1) == 1 and cache.count() == 1
    cache.close()
    assert GenerationCache(tmp_path / "gen.sqlite").get(ns1, key)["answer"] == "Paris"   # persisted


def test_smoke_metrics() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert best_em_f1("the eiffel tower", ["Eiffel Tower", "Paris"]) == (1.0, 1.0)
    em, f = best_em_f1("It was William Shakespeare", ["Shakespeare", "William Shakespeare"])
    assert em == 0.0 and f == pytest.approx(2 * (2 / 4) * 1 / (2 / 4 + 1))
    assert f1("", "") == 1.0 and best_em_f1("x", []) == (0.0, 0.0)
