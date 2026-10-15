# HANDOFF -- Samarth's lane (indexes, retrievers, cache, generation, cluster + perplexity filters)

Last updated: 2026-10-15 (after step 0)

## Status board
| Step | Title | Status | Date | Commit |
|---|---|---|---|---|
| 0 | Repo skeleton and environment | DONE-LOCAL | 2026-10-15 | `step 0: repo skeleton and environment` |
| 1 | Config, paths, logging, data-contract schemas | TODO | | |
| 2 | Load NQ corpus and questions | TODO | | |
| 3 | BM25 index and retriever | TODO | | |
| 4 | Dense index and retriever (+ get_doc_vector) | TODO | | |
| 5 | SPLADE index and retriever | TODO | | |
| 6 | RRF hybrid retriever | TODO | | |
| 7 | Top-150 cache builder | TODO | | |
| 8 | Poison ingestion hook (index poison shards) | TODO | | |
| 9 | Generation harness and cache | TODO | | |
| 10 | Cluster filter | TODO | | |
| 11 | Perplexity filter and score cache | TODO | | |
| 12 | Held-out phrasing tool | TODO | | |
| 13 | End-to-end smoke test | TODO | | |

Status values: TODO, IN PROGRESS, DONE-LOCAL (passes on tiny local profile), DONE (also passed on Ada), REOPENED, BLOCKED.

## Ada requests (for Arya and Vansh)
| Job | After step | Owner | Command | Status (REQUESTED / RUNNING / PASSED / FAILED) | Report |
|---|---|---|---|---|---|
| A0 setup | 0 | Arya or Vansh (whoever starts first) | `bash ada/setup_env.sh` (login node), then `bash ada/setup_env.sh --check-only` (GPU session) | REQUESTED | `$IRE_DATA_DIR/reports/A0.json` |
| A1 data | 2 | Arya | | not yet requested | |
| A2 bm25 | 3 | Arya | | not yet requested | |
| A3 dense | 4 | Vansh | | not yet requested | |
| A4 splade | 5 | Vansh | | not yet requested | |
| A5 cache | 7 | Vansh | | not yet requested | |
| A6 poison | 8 | Arya | | not yet requested | |
| A7 generate | 9 | Vansh | | not yet requested | |
| A8 perplexity | 11 | Vansh | | not yet requested | |
| A9 smoke | 13 | Arya/Vansh | | not yet requested | |

Teammates: the full instructions for each job are in `ADA_RUNBOOK.md`.

```
ADA REQUEST A0
Owner: whoever starts first (Arya or Vansh)          Needs: nothing
Run:   git pull && bash ada/setup_env.sh                    (on the LOGIN node: it needs internet)
       then, inside a GPU session:  bash ada/setup_env.sh --check-only
Time:  10-30 min (estimate)      Disk: ~15 GB under $IRE_DATA_DIR (estimate: environment ~6 GB + models ~8 GB)
Success looks like: the check_env output ends with a RESULT line saying OK and lists a GPU, and the script prints "SUMMARY A0: OK".
Send back: $IRE_DATA_DIR/reports/A0.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A0_setup.log and the report file if one exists.
```

## What teammates can use right now
- `python scripts/check_env.py [--report JOB]` -- prints Python/torch/CUDA/GPU, free disk under `$IRE_DATA_DIR`, library
  versions, Java, and whether the profile's models are cached; `--report JOB` also writes `$IRE_DATA_DIR/reports/JOB.json`.
  Exit code 0 = OK.
- `python scripts/download_models.py [--env local|ada]` -- downloads the profile's models into `HF_HOME` (resumable).
- `bash ada/setup_env.sh [--check-only]` -- job A0 (Linux/Ada only).
- `ire.config`: `get_env_name()`, `get_data_dir()`, `load_env_profile(name=None)` -- reads `.env` and
  `configs/env_<IRE_ENV>.yaml`.

## Interface changes
(Anything that differs from the contracts in IMPLEMENTATION_PLAN.md section 2. Empty = none.)

None to the data contracts. Layout notes (not contract changes):
- `src/ire/config/` and `src/ire/vectors/` are packages (the plan's layout lists them as folders). Step 1's
  "`src/ire/config.py`" will live in `src/ire/config/`, and step 4's `vectors.py` in `src/ire/vectors/`.
- Models are downloaded to the Hugging Face cache at `HF_HOME` (set in `.env` to `$IRE_DATA_DIR/hf_cache`), not to a
  separate `$IRE_DATA_DIR/models` folder, so `from_pretrained(<model id>)` works the same everywhere.
- There is no `data/` folder in the repo: all data, indexes, caches and models live under `$IRE_DATA_DIR`.

## Step log
(Newest first. Per step: what was built, files, how to run, verify results with actual numbers, open issues.)

### Step 0 -- Repo skeleton and environment -- DONE-LOCAL (2026-10-15, env=local)
**Built**
- Profiles `configs/env_local.yaml` (5,000 passages, 100 dev questions, `intfloat/e5-small-v2`,
  `Qwen/Qwen2.5-1.5B-Instruct`, SPLADE cocondenser-ensembledistil, GPT-2 small, bm25s, FAISS flat) and
  `configs/env_ada.yaml` (corpus `full`, dev subset 1,000 q + 50,000 passages, `intfloat/e5-base-v2`,
  `Qwen/Qwen2.5-3B-Instruct`, same SPLADE / GPT-2, HNSW). `IRE_ENV` selects one; seed 13 in both.
- `src/ire/config/env.py`: loads `.env` (never overrides already-set variables, never prints values), validates
  `IRE_ENV`, requires `IRE_DATA_DIR` to be set and **outside** the repo, loads the profile YAML.
- Package layout `src/ire/{config,data,retrieval,vectors,generation,defenses,utils}/` plus placeholder
  `defenses/cluster.py` and `defenses/perplexity.py` (docstrings only; implemented in steps 10-11).
- `scripts/check_env.py`, `scripts/download_models.py`.
- `ada/common.sh` (loads `.env`, activates env, GPU info, SLURM settings in one place), `ada/setup_env.sh`
  (job A0), `ada/README.md`; A0 block added to `ADA_RUNBOOK.md`.
- `requirements.txt` (pinned direct deps, Python 3.13; torch 2.6.0 installed from the cu124 index),
  `requirements-lock.txt` (full `pip freeze`), `pyproject.toml` (src layout, pytest config).
- `tests/test_skeleton.py` (7 tests).
- `.gitignore`: appended `models/`, `envs/`, `*.memmap`, `*.mmap`, `*.pkl`, `*.json.tmp`,
  `.claude/settings.local.json`, and negations `!src/ire/data/`, `!src/ire/vectors/` (the existing `data/` and
  `vectors/` patterns were also ignoring the source packages). `.gitattributes`: appended `*.sh text eol=lf`.
- The existing venv was missing everything except torch; installed the pinned packages into it (torch untouched)
  plus `pip install -e . --no-deps`.

**How to run (PowerShell, repo root)**
```
.\.venv\Scripts\python.exe scripts/check_env.py --report A0_local
.\.venv\Scripts\python.exe -m pytest -q
```

**Verify results (env=local)**
- `check_env.py`: Python 3.13.7, torch 2.6.0+cu124 (CUDA 12.4), GPU 0 = NVIDIA GeForce RTX 3050 6GB Laptop GPU
  (6.0 GiB total, 5.04 GiB free), disk 11.7 GiB free of 349.7 GiB under `IRE_DATA_DIR` (writable); all required
  libraries importable (numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, transformers 5.19.0,
  sentence-transformers 6.1.0, faiss 1.15.1, bm25s 0.3.13, PyStemmer 3.1.0, datasets 5.1.0,
  huggingface_hub 1.33.0, pytest 9.1.1); pyserini not installed and Java not found (neither is needed with bm25s);
  all 4 local models cached; `RESULT: OK`, exit 0. Local report: `$IRE_DATA_DIR/reports/A0_local.json`.
- Extra sanity: transformers sees torch; `intfloat/e5-small-v2` loads from cache on CUDA and encodes (dim 384).
- `pytest -q`: 7 passed.
- Checked that the HF token value appears in no repo file and not in the report. `.env` stays git-ignored.
- `git status` clean after the commit; no data files tracked (see the commit's file list: code, configs and docs only).

**Open issues**: see below.

## Open issues / blockers
- **Low local disk**: only 11.7 GiB free on C: (97% used). Fine for the local profile, but step 2 downloads BEIR NQ
  (the full corpus zip must be downloaded even to build a 5,000-passage subset). Samarth: free some space before step 2.
- **Model choices to confirm** (config values, easy to change before the Ada jobs run): Ada dense model
  `intfloat/e5-base-v2` (job A3) and Ada LLM `Qwen/Qwen2.5-3B-Instruct` (job A7; same family as the local stand-in).
- `ada/common.sh` has placeholder SLURM values (`TODO_partition`); the first teammate on Ada should fill them in.
- `.gitattributes` sends `*.npy` to LFS while `.gitignore` ignores `*.npy`. Harmless (ignored files are never
  tracked), but if a `.npy` ever needs to be shared, remove it from `.gitignore` deliberately.
