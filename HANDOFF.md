# HANDOFF -- Samarth's lane (indexes, retrievers, cache, generation, cluster + perplexity filters)

Last updated: 2026-10-15 (after step 1)

## Status board
| Step | Title | Status | Date | Suggested message |
|---|---|---|---|---|
| 0 | Repo skeleton and environment | DONE-LOCAL | 2026-10-15 | `step 0: repo skeleton and environment` (committed) |
| 1 | Config, paths, logging, data-contract schemas | DONE-LOCAL | 2026-10-15 | `step 1: config, paths, logging, contract schemas` |
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
- `ire.config.load_config(env=None)` -- merged `configs/experiment.yaml` + `configs/env_<IRE_ENV>.yaml`.
  `cfg["pool_size"]`, `cfg.get("cluster.tau_c")`, `cfg.path("corpus")` (a `Path` from `configs/paths.yaml`),
  `cfg.env`. Importing `ire.config` loads `.env`, so nothing has to be exported by hand (PowerShell included).
  Also `get_env_name()`, `get_data_dir()`, `load_env_profile(name=None)`.
- `ire.schemas` -- the section-2 contracts: `CorpusRecord`, `PoisonRecord`, `QuestionRecord`,
  `RetrievalCacheRecord` (`.from_dict(d)`, `.to_dict()`), `read_jsonl(path, RecordType)`, `iter_jsonl(...)`,
  `write_jsonl(path, records, append=False)`; poison IDs: `is_poison_id`, `make_poison_id`, `parse_poison_id`;
  filter API types: `FilterContext`, `FilterResult`, `FilterFn`, `VectorAPI`; `GenerationOutput`.
  Arya/Vansh: validate your files with e.g. `read_jsonl("poison_fact.jsonl", PoisonRecord)`; errors name the line.
- `ire.utils.log.get_logger(name)` -- console + `$IRE_DATA_DIR/logs/<name>.log`, every line tagged `env=<IRE_ENV>`.
- `python scripts/check_ada_report.py <A0..A9> [--file PATH] [--allow-local]` -- checks a returned Ada report
  (looks in `reports/<job>.json`, then `$IRE_DATA_DIR/reports/<job>.json`). Exit 0 = PASS. Rules exist for A0;
  A1-A9 print NOT_IMPLEMENTED until their step adds the rules.

## Interface changes
(Anything that differs from the contracts in IMPLEMENTATION_PLAN.md section 2. Empty = none.)

None to the data contracts. Layout notes (not contract changes):
- `src/ire/config/` and `src/ire/vectors/` are packages (the plan's layout lists them as folders). Step 1's
  "`src/ire/config.py`" will live in `src/ire/config/`, and step 4's `vectors.py` in `src/ire/vectors/`.
- Models are downloaded to the Hugging Face cache at `HF_HOME` (set in `.env` to `$IRE_DATA_DIR/hf_cache`), not to a
  separate `$IRE_DATA_DIR/models` folder, so `from_pretrained(<model id>)` works the same everywhere.
- There is no `data/` folder in the repo: all data, indexes, caches and models live under `$IRE_DATA_DIR`.
- (Step 1) Validation is **strict**: every section-2 field is required and unknown extra fields are rejected
  (e.g. a `trust_tier` field fails). The poison record needs `title` (use `""`), as in section 2; the PDF's
  example omits it. If you need an extra field, tell Samarth so it is added here first.
- (Step 1) `PoisonRecord.doc_id` must equal `poison_<qid>_<attack>_<n>` built from the record's own `qid` and
  `attack`; `target_answer` must be non-empty. `RetrievalCacheRecord.top150` must have at most 150 entries, unique
  doc_ids, sorted by score descending (ties allowed), and finite numeric scores.
- (Step 1) `FilterContext` fields: `texts` (doc_id -> text), `vectors` (a `VectorAPI` or None), `config`
  (an `ire.config.Config`), `thresholds` (fold-specific, default `{}`).

## Step log
(Newest first. Per step: what was built, files, how to run, verify results with actual numbers, open issues.)

### Step 1 -- Config, paths, logging, contract schemas -- DONE-LOCAL (2026-10-15, env=local)
**Built**
- `configs/paths.yaml`: every file location; `${IRE_DATA_DIR}` for big data (corpus, questions, poison shards,
  indexes, vectors, caches, logs, reports), repo-relative for small tracked files (`reports/`,
  `configs/perplexity_thresholds.json`, `heldout/`).
- `configs/experiment.yaml`: shared settings from the plan: seed 13, 200 target + 200 clean, folds A/B, retrievers,
  `top_k_cache` 150, `pool_size` 50, `final_k` 5, `rrf_k` 60, attacks, poison rates 1/3/5, generation
  (temperature 0, 32 new tokens), cluster (top-20, k=2, tau_c 0.85 + sweep 0.70-0.95, ROUGE-L 0.25), perplexity
  (alpha 2.5% + sweep 1/2.5/5/10%, 1,000 calibration passages, min 20 tokens, rule `either`).
  Per-machine values (`corpus.mode`, models, batch sizes) stay in `env_*.yaml`; `seed` moved out of them so it has
  a single source.
- `src/ire/config/loader.py`: `load_config()` deep-merges experiment + env profile, checks required keys, resolves
  paths (`${VAR}` from the environment; an unset variable is an error; `IRE_DATA_DIR` must be outside the repo).
  `ire/config/env.py` now loads `.env` on import (python-dotenv, existing variables win).
- `src/ire/schemas.py`: section-2 records with strict validation + JSONL helpers (atomic writes, line-numbered
  errors) + poison-ID helpers + filter/vector/generation interface types.
- `src/ire/utils/log.py`: `get_logger`, `close_logger`, `log_file_path`.
- `src/ire/ada_report.py` + `scripts/check_ada_report.py`: A0 rules check every field `check_env.py` writes
  (ok, torch + >= 1 GPU, required libraries, writable disk + free space, every `env_ada.yaml` model cached, git
  commit; warnings for a dirty tree or Python other than 3.13). A1-A9 are stubs that state their acceptance check
  and the step that will implement it. Only `env=ada` reports can PASS; `--allow-local` gives `LOCAL_DRY_RUN`.
- Tests: `tests/test_config.py`, `tests/test_schemas.py`, `tests/test_ada_report.py`, `tests/test_log.py`.
- Process: CLAUDE.md and IMPLEMENTATION_PLAN.md section 0 now say "do not commit"; `.claude/settings.json` turns
  off commit/PR attribution.

**How to run (PowerShell, repo root)**
```
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_ada_report.py A0                      # once a teammate returns A0.json
.\.venv\Scripts\python.exe scripts/check_ada_report.py A0 --file <IRE_DATA_DIR>/reports/A0_local.json --allow-local
```

**Verify results (env=local)**
- `pytest -q`: 119 passed in 0.90s (7 skeleton, 13 config, 75 schemas, 23 Ada report, 1 log).
  Schemas: round trip for all 4 record types; for every field of every record, a missing field and a wrong-type
  value both raise; unknown fields, bad enum values, mismatched poison IDs, unsorted / duplicate / >150 /
  bool / NaN cache entries raise; read errors give the line number.
- `check_ada_report.py A0` with no returned report: "No report found for A0", exit 1 (correct: A0 not back yet).
- `check_ada_report.py A0 --file .../A0_local.json --allow-local` on the real local report (refreshed today):
  all A0 rules pass (GPU RTX 3050 6 GB, 13 required libraries, 10.9 GiB free, 4 local models cached), 1 warning
  (dirty tree), `STATUS: LOCAL_DRY_RUN`, exit 2. Without `--allow-local` it fails on `env='local'`.
  `check_ada_report.py A5`: `STATUS: NOT_IMPLEMENTED (rules for A5 are added in step 7 ...)`, exit 1.
- Missing path key: `load_config().path('corpus_typo')` -> `ConfigError: Path key 'corpus_typo' is not defined in
  configs/paths.yaml. Known keys: cache_dir, corpus, ...`. A `paths.yaml` missing a required key fails with
  "paths.yaml is missing required path key(s): corpus" (tested).
- Logger: wrote `2026-10-15 16:53:34 INFO    env=local step1_verify: step 1 verify log line` to console and
  `$IRE_DATA_DIR/logs/step1_verify.log`.
- Config (env=local): seed 13, corpus.mode subset, dense `intfloat/e5-small-v2`, pool 50 / cache 150 / final 5,
  `corpus` -> `<IRE_DATA_DIR>/data/corpus.jsonl`.
- No data downloaded. No Ada job for this step.

### Step 0 -- Repo skeleton and environment -- DONE-LOCAL (2026-10-15, env=local)
**Built**
- Profiles `configs/env_local.yaml` (5,000 passages, 100 dev questions, `intfloat/e5-small-v2`,
  `Qwen/Qwen2.5-1.5B-Instruct`, SPLADE cocondenser-ensembledistil, GPT-2 small, bm25s, FAISS flat) and
  `configs/env_ada.yaml` (corpus `full`, dev subset 1,000 q + 50,000 passages, `intfloat/e5-base-v2`,
  `Qwen/Qwen2.5-3B-Instruct`, same SPLADE / GPT-2, HNSW). `IRE_ENV` selects one; seed 13 in both
  (moved to `configs/experiment.yaml` in step 1).
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
- **Low local disk**: 10.9-12 GiB free on C: (97% used); one check_env run at 10:32 UTC even saw 4.2 GiB (a
  temporary dip from something outside the project). Fine for steps 0-1, but step 2 downloads BEIR NQ (the full
  corpus zip must be downloaded even to build a 5,000-passage subset). Samarth: free some space before step 2.
- `questions_all.jsonl` (step 2) has no `role`/`fold` yet, so it cannot use `QuestionRecord` as is; step 2 will
  define its record and record it under "Interface changes".
- `src/ire/stubs.py` (`candidate_pool` stub, section 2) is not created yet; it will be added when first needed
  (step 13 at the latest).
- **Model choices to confirm** (config values, easy to change before the Ada jobs run): Ada dense model
  `intfloat/e5-base-v2` (job A3) and Ada LLM `Qwen/Qwen2.5-3B-Instruct` (job A7; same family as the local stand-in).
- `ada/common.sh` has placeholder SLURM values (`TODO_partition`); the first teammate on Ada should fill them in.
- `.gitattributes` sends `*.npy` to LFS while `.gitignore` ignores `*.npy`. Harmless (ignored files are never
  tracked), but if a `.npy` ever needs to be shared, remove it from `.gitignore` deliberately.
