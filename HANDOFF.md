# HANDOFF -- Samarth's lane (indexes, retrievers, cache, generation, cluster + perplexity filters)

Last updated: 2026-10-16 (after step 2)

## Status board
| Step | Title | Status | Date | Suggested message |
|---|---|---|---|---|
| 0 | Repo skeleton and environment | DONE-LOCAL | 2026-10-15 | `step 0: repo skeleton and environment` (committed) |
| 1 | Config, paths, logging, data-contract schemas | DONE-LOCAL | 2026-10-15 | `step 1: config, paths, logging, contract schemas` (committed) |
| 2 | Load NQ corpus and questions | DONE-LOCAL | 2026-10-16 | `step 2: load NQ corpus and questions` |
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
| A1 data | 2 | Arya | `bash ada/run_prepare_data.sh` (login node) | REQUESTED | `$IRE_DATA_DIR/reports/A1.json` + `reports/data_card_ada.json` |
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

```
ADA REQUEST A1
Owner: Arya          Needs: A0 done (environment exists)
Run:   git pull && bash ada/run_prepare_data.sh               (on the LOGIN node: it downloads ~735 MB)
Time:  10-30 min (estimate: download at a few MB/s + two passes over 2.68M passages)
Disk:  ~3 GB under $IRE_DATA_DIR (estimate: raw parquet 0.75 GB + corpus.jsonl ~1.5-2 GB)
Success looks like: the output ends with "RESULT: OK" and "SUMMARY A1: OK", with corpus passages ~2.68M,
       200 target + 200 clean (100 per role and fold) and every check PASS.
Send back: $IRE_DATA_DIR/reports/A1.json (paste it in the team chat or commit it to reports/)
           and commit reports/data_card_ada.json (counts, sizes, sources; no absolute paths in it).
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A1_prepare_data.log and the report file if one exists.
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
  (looks in `reports/<job>.json`, then `$IRE_DATA_DIR/reports/<job>.json`). Exit 0 = PASS. Rules exist for A0 and
  A1; A2-A9 print NOT_IMPLEMENTED until their step adds the rules.
- `python scripts/prepare_data.py [--corpus-mode full|random] [--dev-subset] [--source hf|ukp] [--force]` -- builds
  `$IRE_DATA_DIR/data/{corpus,questions_all,questions}.jsonl` (step 2; details in the step log). Local files exist now:
  5,000 passages, 2,255 pool questions, 400 placeholder questions.
- `ire.data.nq`: `BeirParquet` / `BeirZip` readers (`iter_beir_corpus` streams the corpus), `load_beir_queries`,
  `load_beir_qrels`, `load_nq_open`, `normalize_question` (the join key), `select_placeholder(pool, 200, 200, seed=13)`.
- `ire.schemas.QuestionAllRecord` for `questions_all.jsonl`.
- **Arya**: `questions.jsonl` is a PLACEHOLDER (200 target + 200 clean, seed 13, folds A/B 100 each). Your final
  set replaces it; same `QuestionRecord` format. Pick from `questions_all.jsonl` (every question there has a gold
  passage and an nq_open answer). After replacing it, re-run step 2 in random mode locally so the local corpus
  contains the new gold passages (tell Samarth).

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
- (Step 2) **New record `QuestionAllRecord`** for `questions_all.jsonl`: `{"qid": str, "question": str,
  "answers": [str], "gold_doc_ids": [str]}` (no role/fold). `answers` and `gold_doc_ids` must be non-empty: the file
  only holds questions with >= 1 gold passage in the full corpus and >= 1 nq_open answer.
- (Step 2) In the **local** profile `questions_all.jsonl` lists all 2,255 eligible questions, but the 5,000-passage
  local corpus only contains the gold passages of the 400 `questions.jsonl` questions (2,304 pool gold ids are not
  in it). In `full` mode and with `--dev-subset` every `questions_all` gold id is in the corpus (checked).
- (Step 2) Config: corpus modes are now `full` | `random` (gold + seeded random fill) | `subset` (~500k hard
  negatives, TODO after step 5). `env_local.yaml` changed `corpus.mode: subset` -> `random` and
  `n_dev_questions: 100` -> `400` (the local profile now uses the 400 placeholder questions). The `dev_subset`
  sizes moved to `experiment.yaml`; new path key `dev_data_dir`.
- (Step 2) BEIR NQ is downloaded from the pinned Hugging Face copy (`BeIR/nq`, `BeIR/nq-qrels`) by default, because the
  original UKP server ran at 0.06-0.4 MB/s from Samarth's machine; `--source ukp` uses the original `nq.zip`. Both give
  identical outputs (tested on a fixture).

## Step log
(Newest first. Per step: what was built, files, how to run, verify results with actual numbers, open issues.)

### Step 2 -- Load NQ corpus and questions -- DONE-LOCAL (2026-10-16, env=local)
**Built**
- `src/ire/data/download.py`: pinned, resumable downloads with verified TLS (certifi): BEIR NQ from Hugging Face
  (`BeIR/nq` @ `b7253e6c`, `BeIR/nq-qrels` @ `519acd4e`) or the original UKP `nq.zip` (`--source ukp`), and
  `google-research-datasets/nq_open` @ `5dd9790a` (validation + train parquet).
- `src/ire/data/nq.py`: `BeirParquet` / `BeirZip` readers (corpus streamed in batches / line by line, never fully
  in RAM), qrels (score > 0), nq_open answers (union over rows and splits), the join key `normalize_question` (NFKC,
  lowercase, punctuation removed, whitespace collapsed), eligible pool, seeded placeholder selection, seeded passage
  selection, corpus writer.
- `src/ire/data/prepare.py`: `prepare()` (two streaming passes: scan, then write) and `verify_outputs()` (re-reads
  all three files through the schemas). Mode `subset` raises `NotImplementedError` (clearly marked TODO, steps 3-5).
- `scripts/prepare_data.py`: CLI with free-disk and download-size guards (`--max-download-gb 6`, `--min-free-gb 5`),
  resume stamp (`data/.prepare_data.json`), `--force`, `--dev-subset`, `--delete-raw`; writes
  `$IRE_DATA_DIR/reports/A1_local.json` and the data card `reports/data_card_local.json` (relative paths only).
- `ada/run_prepare_data.sh` (job A1); A1 rules in `scripts/check_ada_report.py` (corpus size matches the mode,
  200 + 200 balanced over folds, every gold ID exists, no duplicate doc/question IDs, corpus rows = count).
- `ire.schemas.QuestionAllRecord`; `requirements.txt` adds `pyarrow==26.0.0`, `certifi==2026.7.22`.
- Tests: `tests/test_data_nq.py` (13, on a synthetic BEIR zip + parquet built in a temp dir, no network),
  `QuestionAllRecord` cases in `tests/test_schemas.py`, A1 rule cases in `tests/test_ada_report.py`.

**How to run (PowerShell, repo root)**
```
.\.venv\Scripts\python.exe scripts/prepare_data.py                 # local: random mode, 5,000 passages
.\.venv\Scripts\python.exe scripts/check_ada_report.py A1 --file <IRE_DATA_DIR>/reports/A1_local.json --allow-local
.\.venv\Scripts\python.exe -m pytest -q
```

**Verify results (env=local, local profile; numbers are for correctness only)**
- Downloads: 733.3 MiB (BEIR corpus parquet 728.8, queries 0.1, qrels 0.1, nq_open 4.5). Free disk under
  `IRE_DATA_DIR`: 35.63 GiB before, 34.48 GiB after. Run time 132 s (download ~96 s, both corpus passes ~26 s).
- BEIR NQ corpus: 2,681,468 passages, 0 duplicate ids, no `poison_` ids. Test qrels: 4,201 gold ids, all present.
- BEIR test queries: 3,452; with >= 1 gold passage in the corpus: 3,452 (100.0%).
  Matched to nq_open answers: 2,255 (answer coverage 65.3%), all from nq_open validation (0 extra from train).
  The 1,197 unmatched are not spelling/spacing near-misses (0 more match with all whitespace removed); nq_open keeps
  only NQ questions with short answers, so they have no answer there. Eligible (gold AND answer): 2,255.
- `questions.jsonl`: 400 = target/A 100, target/B 100, clean/A 100, clean/B 100 (seed 13). Their 509 gold passages
  are all in the corpus (0 missing).
- `corpus.jsonl`: 5,000 = 509 gold + 4,491 seeded random passages, 0 duplicate doc_ids (2.78 MiB).
  `questions_all.jsonl`: 2,255 (0.35 MiB). `questions.jsonl`: 0.07 MiB.
- Checks: corpus_size_matches_mode, no_duplicate_doc_ids, no_duplicate_qids, all_question_gold_in_corpus,
  fold_balance, questions_subset_of_questions_all all PASS. questions_all_gold_in_corpus is false by design in
  local random mode (2,304 pool gold ids not in the 5,000-passage corpus) and is not required in this mode.
- Resume: a re-run downloads 0 MiB and skips the build (6.2 s). Determinism: `--force` rebuild gives byte-identical
  files (same sha256 for all three).
- `check_ada_report.py A1 --allow-local` on the real local report: every A1 rule passes, `STATUS: LOCAL_DRY_RUN`.
- `pytest -q`: 156 passed (7 skeleton, 13 config, 13 data, 88 schemas, 34 Ada report, 1 log).
- Samples (shortened):
  - corpus: `{"doc_id": "doc91", "title": "Market economy", "text": "A market economy is an economic system where
    decisions regarding investment, production, and distribution ...", "is_poison": false}`
  - questions_all: `{"qid": "test3", "question": "who is the leader of the ontario pc party", "answers": ["Patrick
    Brown", "Patrick Walter Brown MPP", "Patrick Walter Brown"], "gold_doc_ids": ["doc17", "doc18"]}`
  - questions: `{"qid": "test1587", "question": "who starred in the movie summer of 42", "answers": ["Oliver
    Conant", ...], "gold_doc_ids": ["doc56384"], "role": "target", "fold": "A"}`
- Raw download kept: 737.5 MiB in `raw/` (under the 1 GB delete threshold); re-downloadable with the same command.

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
- Disk: resolved for now (34.5 GiB free after step 2).
- `questions.jsonl` is a placeholder until Arya's final 200 + 200 set replaces it (see "What teammates can use").
- Corpus mode `subset` (~500k gold + hard negatives) is a TODO that needs the retrievers (after step 5).
- Answer coverage is 65.3% (2,255 of 3,452 BEIR test questions have an nq_open answer); the pool of 2,255 is far
  more than the 400 needed.
- A 4 MiB partial `raw/beir/nq.zip.part` from the abandoned slow UKP download is still in `IRE_DATA_DIR`
  (kept so `--source ukp` can resume; Samarth can delete it).
- `src/ire/stubs.py` (`candidate_pool` stub, section 2) is not created yet; it will be added when first needed
  (step 13 at the latest).
- **Model choices to confirm** (config values, easy to change before the Ada jobs run): Ada dense model
  `intfloat/e5-base-v2` (job A3) and Ada LLM `Qwen/Qwen2.5-3B-Instruct` (job A7; same family as the local stand-in).
- `ada/common.sh` has placeholder SLURM values (`TODO_partition`); the first teammate on Ada should fill them in.
- `.gitattributes` sends `*.npy` to LFS while `.gitignore` ignores `*.npy`. Harmless (ignored files are never
  tracked), but if a `.npy` ever needs to be shared, remove it from `.gitignore` deliberately.
