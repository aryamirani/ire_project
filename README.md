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
| `src/ire/` | the package: `data/`, `retrieval/` (indexes, top-150 cache, `pool.py` = candidate pool with poison masking), `poison/`, `defenses/` (three filters + `controls.py` random / oracle), `generation/`, `eval/` (metrics, statistics, sweep runner, analysis) |
| `scripts/` | one command per stage (prepare data, build index, make / ingest poison, build cache, calibrate filters, smoke test, `run_sweep.py`, `analyze.py`, `make_plots.py`, human-calibration sheets and agreement) |
| `ada/` | cluster job wrappers; each is one command, resumable, and writes a JSON report |
| `configs/` | paths, environment profiles, experiment settings; `frozen/` holds the question set, phrase list and attack targets (never edited) |
| `reports/` | JSON reports from the Ada runs (checked with `scripts/check_ada_report.py <job>`) |
| `heldout/` | the blind held-out phrasings and the instructions given to their writer |
| `tests/` | pytest suite |
| `docs/` | plan, mid report, step log (`HANDOFF.md`, `HANDOFF_ARYA.md`), original step plan, Ada runbook |

Large files (corpus, indexes, caches, models) live under `$IRE_DATA_DIR`, never in git.

## Setup and a local run
```bash
uv venv --python 3.13 .venv && source .venv/bin/activate     # Python 3.11+ needed (macOS python3 is 3.9)
uv pip install torch==2.6.0 && uv pip install -r requirements.txt matplotlib && uv pip install -e . --no-deps
cp .env.example .env    # IRE_ENV=local (small corpus, small models) or ada; set IRE_DATA_DIR
                        # (outside the repo, or exactly <repo>/ire_data_dir, which is git-ignored)
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
Who has done what and what is left: `TEAM_STATUS.md`. Retrieval, attacks, detectors and the smoke test are done and
passed on Ada. The evaluation harness (masking, metrics, controls, sweep runner, statistics, plots, calibration
tools) is written and tested, and has run end to end on the local tiny corpus (pilot, analysis, plots, calibration
sheets); A7 passed on Ada (batch size 32). The Ada pilot and all 384 sweep settings have run (145,600 records, 0 failed); the analysis tables and figures are in `reports/analysis_main/`; the human calibration is done (judge accuracy 0.993 against the human consensus). Only the report is left (`TEAM_STATUS.md`, decision rule `docs/DECISION_RULE.md`). Deadline 20 Nov.
