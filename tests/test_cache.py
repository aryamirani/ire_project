"""Step 7: top-150 cache writer / loader on temp folders with fake searchers (no models, no real indexes)."""

import json
from pathlib import Path

import pytest

from ire.config.loader import Config
from ire.retrieval.cache import (
    CacheError,
    StaleCacheError,
    UnmaskedReferenceError,
    cache_dir_for,
    cache_path,
    current_key,
    fuse_rows,
    load_cache,
    rrf_component_rows,
    sidecar_path,
    write_cache,
)
from ire.retrieval.rrf import fuse
from ire.schemas import QuestionRecord, read_jsonl, write_jsonl

N_Q = 10
DEPTH = 5


def make_cfg(tmp_path: Path) -> Config:
    """A real Config with every path in tmp_path; index manifests are written by hand."""
    paths = {"questions": tmp_path / "questions.jsonl", "cache_dir": tmp_path / "caches",
             "indexes_dir": tmp_path / "indexes"}
    values = {"top_k_cache": DEPTH, "rrf": {"k": 60, "depth": DEPTH, "components": ["bm25", "dense"]}}
    cfg = Config(env="test", values=values, paths=paths)
    write_jsonl(paths["questions"], [QuestionRecord(qid=f"q{i}", question=f"question {i}", answers=["a"],
                                                    gold_doc_ids=["d0"], role="target" if i % 2 else "clean",
                                                    fold="A" if i < N_Q // 2 else "B") for i in range(N_Q)])
    for name in ("bm25", "dense", "splade"):
        set_fingerprint(cfg, name, {"corpus_sha256": "abc", "shards": [], "settings": {"name": name}})
    return cfg


def set_fingerprint(cfg: Config, name: str, fp: dict) -> None:
    d = cfg.path("indexes_dir") / cfg.env / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"fingerprint": fp, "n_poison": 0}), encoding="utf-8")


class FakeSearcher:
    """Deterministic rows sorted by (-score, doc_id); can crash after some calls to simulate a killed job."""

    def __init__(self, name: str, fail_after_calls: int | None = None) -> None:
        self.name, self.calls, self.fail_after_calls = name, 0, fail_after_calls

    def rows(self, batch):
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("simulated kill")
        out = []
        for q in batch:
            i = int(q.qid[1:])
            row = [(f"d{(i * 7 + j * (3 if self.name == 'dense' else 1)) % 40:02d}", float(10 - j // 2))
                   for j in range(DEPTH)]
            out.append(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))
        return out


def build(cfg: Config, name: str, searcher: FakeSearcher | None = None, force: bool = False, batch_size: int = 3):
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    key = current_key(cfg, name, cfg.path("questions"))
    if name == "rrf":
        comps = rrf_component_rows(cfg, cfg.path("questions"), [q.qid for q in questions])
        compute = lambda b: fuse_rows(comps["bm25"], comps["dense"], b, 60, DEPTH)  # noqa: E731
    else:
        compute = (searcher or FakeSearcher(name)).rows
    return write_cache(cache_path(cache_dir_for(cfg), name), key, questions, compute, DEPTH,
                       exact_len=DEPTH if name == "dense" else None, batch_size=batch_size, force=force)


def test_build_and_load(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    side = build(cfg, "bm25")
    assert not side["skipped"] and side["n_questions"] == N_Q and side["unmasked_reference"] is False
    assert side["row_length"] == {"min": DEPTH, "median": DEPTH, "max": DEPTH, "short_rows": 0}
    rows = load_cache(["bm25"], cfg=cfg)["bm25"]
    assert list(rows) == [f"q{i}" for i in range(N_Q)] and rows["q0"] == FakeSearcher("bm25").rows(
        [QuestionRecord(qid="q0", question="x", answers=["a"], gold_doc_ids=["d"], role="target", fold="A")])[0]
    sidecar = json.loads(sidecar_path(cache_path(cache_dir_for(cfg), "bm25")).read_text(encoding="utf-8"))
    assert sidecar["key"]["index_fingerprint"]["settings"] == {"name": "bm25"} and sidecar["key"]["depth"] == DEPTH


def test_rerun_is_noop_and_force_is_byte_identical(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    build(cfg, "dense")
    path = cache_path(cache_dir_for(cfg), "dense")
    first = path.read_bytes()
    never = FakeSearcher("dense", fail_after_calls=0)       # would crash if it were called
    assert build(cfg, "dense", never)["skipped"] and never.calls == 0
    assert not build(cfg, "dense", force=True)["skipped"] and path.read_bytes() == first


def test_resume_after_kill_is_byte_identical(tmp_path: Path) -> None:
    clean_cfg = make_cfg(tmp_path / "clean")
    build(clean_cfg, "splade")
    clean = cache_path(cache_dir_for(clean_cfg), "splade").read_bytes()

    cfg = make_cfg(tmp_path / "killed")
    out = cache_path(cache_dir_for(cfg), "splade")
    with pytest.raises(RuntimeError, match="simulated kill"):
        build(cfg, "splade", FakeSearcher("splade", fail_after_calls=2))   # dies after 2 batches of 3
    partial = out.with_name(out.name + ".partial")
    assert not out.exists() and len(partial.read_text(encoding="utf-8").splitlines()) == 6
    with partial.open("ab") as fh:
        fh.write(b'{"qid": "q6", "retr')                                    # half-written line from the kill
    resumed = FakeSearcher("splade")
    side = build(cfg, "splade", resumed)
    assert side["resumed_from_question"] == 6 and resumed.calls == 2       # only q6..q9 searched again
    assert out.read_bytes() == clean and not partial.exists()


def test_partial_from_other_key_starts_over(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(RuntimeError):
        build(cfg, "bm25", FakeSearcher("bm25", fail_after_calls=1))
    set_fingerprint(cfg, "bm25", {"corpus_sha256": "new", "shards": [], "settings": {}})
    assert build(cfg, "bm25")["resumed_from_question"] == 0


def test_stale_questions_or_index_refused(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    build(cfg, "bm25")
    qfile = cfg.path("questions")
    original = qfile.read_bytes()  # bytes: on Windows write_text would also change the line endings
    qfile.write_bytes(original.replace(b"question 3", b"question three"))
    with pytest.raises(StaleCacheError, match="questions_sha256 differs"):
        load_cache(["bm25"], cfg=cfg)
    qfile.write_bytes(original)
    load_cache(["bm25"], cfg=cfg)                                          # back to current
    set_fingerprint(cfg, "bm25", {"corpus_sha256": "abc", "settings": {"name": "bm25"},
                                  "shards": [{"name": "poison_fact.jsonl", "sha256": "f00"}]})
    with pytest.raises(StaleCacheError, match="index_fingerprint differs.*poison ingestion"):
        load_cache(["bm25"], cfg=cfg)


def test_missing_cache_and_missing_index(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(CacheError, match="no cache"):
        load_cache(["dense"], cfg=cfg)
    (cfg.path("indexes_dir") / "test" / "splade" / "manifest.json").unlink()
    with pytest.raises(CacheError, match="no finished index"):
        current_key(cfg, "splade", cfg.path("questions"))


def test_rrf_built_from_cached_components_and_guarded(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    with pytest.raises(CacheError, match="refusing to build rrf"):
        build(cfg, "rrf")                                                  # components not cached yet
    build(cfg, "bm25")
    build(cfg, "dense")
    side = build(cfg, "rrf")
    assert side["unmasked_reference"] is True
    with pytest.raises(UnmaskedReferenceError, match="mask the bm25 and dense lists and call fuse\\(\\) instead"):
        load_cache(["rrf"], cfg=cfg)
    with pytest.raises(UnmaskedReferenceError):
        load_cache(["bm25", "rrf"], cfg=cfg)
    c = load_cache(["bm25", "dense", "rrf"], allow_unmasked_reference=True, cfg=cfg)
    for qid, row in c["rrf"].items():
        assert row == fuse(c["bm25"][qid], c["dense"][qid], k=60, depth=DEPTH)


def test_rrf_goes_stale_with_a_component_index(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    build(cfg, "bm25")
    build(cfg, "dense")
    build(cfg, "rrf")
    set_fingerprint(cfg, "dense", {"corpus_sha256": "abc", "shards": [{"name": "poison_x.jsonl", "sha256": "1"}],
                                   "settings": {"name": "dense"}})
    with pytest.raises(StaleCacheError, match="components differs"):
        load_cache(["rrf"], allow_unmasked_reference=True, cfg=cfg)
    with pytest.raises(CacheError, match="refusing to build rrf"):
        build(cfg, "rrf")                                                  # the dense cache itself is stale now


def test_row_rules_enforced(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    key = current_key(cfg, "dense", cfg.path("questions"))
    out = cache_path(cache_dir_for(cfg), "dense")
    unsorted = lambda b: [[("d1", 1.0), ("d2", 2.0)] for _ in b]          # noqa: E731
    tie_wrong = lambda b: [[("d2", 1.0), ("d1", 1.0)] for _ in b]         # noqa: E731  (tie must be d1, d2)
    short = lambda b: [[("d1", 2.0), ("d2", 1.0)] for _ in b]             # noqa: E731
    for compute in (unsorted, tie_wrong):
        with pytest.raises(CacheError, match="not sorted"):
            write_cache(out, key, questions, compute, DEPTH, force=True)
    with pytest.raises(CacheError, match="exactly 5"):
        write_cache(out, key, questions, short, DEPTH, exact_len=DEPTH, force=True)
    long = lambda b: [[(f"d{j}", 10.0 - j) for j in range(DEPTH + 1)] for _ in b]  # noqa: E731
    with pytest.raises(CacheError, match="at most 5"):
        write_cache(out, key, questions, long, DEPTH, force=True)
