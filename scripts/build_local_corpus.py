"""Arya (LOCAL profile only): rebuild the small random corpus around the FINAL question set.

Samarth's prepare_data.py builds the local corpus around the placeholder questions. After
scripts/select_questions.py has written the final questions.jsonl, this script rewrites corpus.jsonl as
all gold passages of questions.jsonl + seeded random others (corpus.n_passages in env_local.yaml, seed from
experiment.yaml), using the raw BEIR files prepare_data.py already downloaded and Samarth's own helpers.
questions_all.jsonl and questions.jsonl are not touched.

Usage (repo root, after prepare_data.py and select_questions.py):
    python scripts/build_local_corpus.py
Then rebuild the local indexes:  python scripts/build_index.py --retriever bm25 --force  (and dense, splade).

Refuses to run on the Ada profile (the full corpus already contains every gold passage).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.download import beir_hf_paths  # noqa: E402
from ire.data.nq import BeirParquet, choose_passages, scan_corpus, write_corpus  # noqa: E402
from ire.schemas import CorpusRecord, QuestionRecord, iter_jsonl, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402


def main() -> int:
    cfg = load_config()
    log = get_logger("build_local_corpus")
    if cfg["corpus.mode"] != "random":
        print(f"corpus.mode is {cfg['corpus.mode']!r} (env={cfg.env}); this script is only for the local random "
              "corpus. Nothing to do.")
        return 1
    hf = beir_hf_paths(cfg.path("raw_dir"))
    if not hf["corpus"].is_file():
        print(f"raw BEIR files not found under {cfg.path('raw_dir')}: run scripts/prepare_data.py first")
        return 1
    src = BeirParquet(hf["corpus"], hf["queries"], hf["qrels_test"].parent)
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    gold = {d for q in questions for d in q.gold_doc_ids}
    log.info("scanning the raw corpus for %d gold ids of %d questions", len(gold), len(questions))
    scan = scan_corpus(src, gold)
    missing = gold - set(scan.gold_positions)
    if missing:
        print(f"{len(missing)} gold ids are not in the raw corpus (first: {sorted(missing)[:5]})")
        return 1
    n_passages, seed = int(cfg["corpus.n_passages"]), int(cfg["seed"])
    keep = choose_passages(scan.n_passages, set(scan.gold_positions.values()), n_passages, seed)
    n = write_corpus(src, cfg.path("corpus"), keep)
    ids = {r.doc_id for r in iter_jsonl(cfg.path("corpus"), CorpusRecord)}
    ok = n == n_passages and gold <= ids
    print(f"corpus.jsonl: {n} passages ({len(gold)} gold of the final questions, {n - len(gold)} random, "
          f"seed {seed}); all gold present: {gold <= ids}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
