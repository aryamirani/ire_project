# Measuring RAG Safety Against Bad Data

IRE course project (Arya, Samarth, Vansh). One RAG pipeline over BEIR Natural Questions (2.68M passages) with
four retrievers (BM25, e5-base-v2 dense, SPLADE, BM25+dense RRF) and Qwen2.5-3B-Instruct. We inject poisoned
passages (fact, hijack, url, plus three adaptive attacks and a held-out set written blind) and measure how much
harm reaches the answer, with and without three training-free detectors (query-and-phrase, cluster, perplexity)
compared against random and oracle removal. Design: `docs/IRE_Final_Project_Plan.pdf`. Progress so far:
`docs/mid_report/mid_report.pdf`.

## Layout
| Path | What |
|---|---|
| `src/ire/` | the package: `data/`, `retrieval/` (indexes, top-150 cache, poison masking), `poison/`, `defenses/`, `generation/` |
| `scripts/` | one command per stage (prepare data, build index, make / ingest poison, build cache, calibrate filters, smoke test) |
| `ada/` | cluster job wrappers; each is one command, resumable, and writes a JSON report |
| `configs/` | paths, environment profiles, experiment settings; `frozen/` holds the question set, phrase list and attack targets (never edited) |
| `reports/` | JSON reports from the Ada runs (checked with `scripts/check_ada_report.py <job>`) |
| `heldout/` | the blind held-out phrasings and the instructions given to their writer |
| `tests/` | pytest suite |
| `docs/` | plan, mid report, step log (`HANDOFF.md`, `HANDOFF_ARYA.md`), original step plan, Ada runbook |

Large files (corpus, indexes, caches, models) live under `$IRE_DATA_DIR`, never in git.

## Setup and a local run
```bash
python -m venv .venv && .venv/bin/pip install torch==2.6.0 -r requirements.txt && .venv/bin/pip install -e . --no-deps
cp .env.example .env    # IRE_ENV=local (small corpus, small models) or ada; set IRE_DATA_DIR
python scripts/download_models.py
python scripts/prepare_data.py
python scripts/build_index.py --retriever bm25   # then dense, splade
python scripts/make_poison.py && python scripts/index_poison.py
python scripts/build_cache.py --retriever all
pytest -q
```
On macOS prefix commands with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` (faiss and torch ship different
OpenMP runtimes). Full-corpus jobs run on the Ada cluster: see `docs/ADA_RUNBOOK.md`.

## Status
Who has done what and what is left: `TEAM_STATUS.md`. Retrieval, attacks, detectors and the smoke test are done;
the evaluation harness, sweeps and statistics are next (`VANSH_START.md`).
