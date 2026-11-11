# 11 -- Infrastructure, configuration, Ada jobs and reproducibility  (Samarth's lane: steps 0-1, report checker)

## 1. Two environments, one code base
`IRE_ENV=local | ada` (in `.env`) selects a profile. **local** = Samarth's laptop (Windows 11, RTX 3050 6 GB, 16 GB RAM,
Python 3.13, torch 2.6.0+cu124), tiny corpus and small stand-in models -- correctness only. **ada** = the Ada
cluster (Linux, RTX 2080 Ti 11 GB), full corpus and real models -- every reported number. Samarth had no Ada access;
he wrote the job scripts and requests, Arya and Vansh ran them and returned JSON reports, which Samarth checked.
Status values: **DONE-LOCAL** (tests and local checks pass) vs **DONE** (an Ada report passes
`scripts/check_ada_report.py`). Never estimate or invent an Ada result.

## 2. Configuration (`src/ire/config/`)
- `configs/experiment.yaml` -- shared settings (seed 13, pool sizes, retriever settings, attacks, rates, filter
  anchors and sweeps, generation settings). Values from the plan.
- `configs/env_local.yaml`, `configs/env_ada.yaml` -- what differs per machine (corpus mode/size, models, device,
  batch sizes, VRAM cap, dense index type). **Deep-merged on top of** experiment.yaml by `load_config()`.
- `configs/paths.yaml` -- every file location. `${IRE_DATA_DIR}` is replaced from the environment (an unset variable
  is an error); paths without a variable are relative to the repo root (small tracked files only).
- `configs/frozen/` -- question set, phrase list, poison targets: written once, never edited.
- `configs/sweep.yaml` -- the sweep grid.
```python
cfg = load_config()            # active IRE_ENV
cfg["pool_size"]               # 50          (dotted lookup: cfg["rrf.k"] -> 60; missing key -> ConfigError naming the file)
cfg.get("cluster.tau_c")       # 0.85
cfg.path("corpus")             # Path(<IRE_DATA_DIR>/data/corpus.jsonl)
cfg.thresholds_path("query_phrase_thresholds")   # Ada: tracked file; local: local calibration if present
```
- Importing `ire.config` loads `.env` (python-dotenv), so nothing is exported by hand (PowerShell included). `HF_TOKEN`
  is never printed or logged (`check_env` prints only "set / not set"); `.env` is git-ignored.
- `get_data_dir()` refuses an `IRE_DATA_DIR` inside the repo (except exactly `<repo>/ire_data_dir`, git-ignored), so
  2.68M-passage files and 19 GB of indexes can never be committed by accident.
- Required keys are checked at load (`REQUIRED_CONFIG_KEYS`, `REQUIRED_PATH_KEYS`): a missing key fails at startup,
  not hours into a job.
- **No absolute paths in code**; everything goes through `cfg.path(...)`.

## 3. Logging and run info
`get_logger(name)` -> console + `$IRE_DATA_DIR/logs/<name>.log`, **every line tagged `env=<IRE_ENV>`**, so local and Ada
logs can never be confused. `run_info.git_commit()` records the commit (and dirty state) in every report.

## 4. Ada jobs: one command, resumable, a JSON report
| Job | Script | Produces | Status |
|---|---|---|---|
| A0 | `ada/setup_env.sh` | environment + models + `check_env` report | PASSED |
| A1 | `ada/run_prepare_data.sh` | corpus, questions_all, questions + data card | PASSED |
| A2/A3/A4 | `ada/run_index.sh bm25|dense|splade` | the three indexes | PASSED |
| P1 | `ada/run_make_poison.sh` | poison shards (Arya) | done |
| A6 | `ada/run_index_poison.sh` | poison ingested + masking check | PASSED (re-run with held-out 2026-10-30) |
| A5 | `ada/run_cache.sh all` | top-150 caches (after A6!) | PASSED (re-run 2026-10-30) |
| P2 | `ada/run_query_phrase.sh` | query-phrase cutoffs | done (re-run 2026-10-30) |
| A8 | `ada/run_ppl.sh` | perplexity thresholds + score cache | PASSED |
| A9 | `ada/run_smoke.sh` | end-to-end smoke test | PASSED |
| A7 | `ada/run_generate_benchmark.sh` | 3B throughput, cache and determinism checks | **PASSED** (run by Arya, 2026-11-10) |
| sweep | `ada/run_sweep.sh [--pilot] [--tier ...]` | run records | **DONE**: pilot 2026-11-10; 384/384 settings 2026-11-11 on gnode082 (run by Arya) |
Every job prints a one-line summary and writes `$IRE_DATA_DIR/reports/<job>.json` (env, git commit, timestamps,
checks, numbers, the exact rerun command). Order constraint: **A2-A4 -> A6 -> A5** (A6 makes every cache stale).

### The report checker (`src/ire/ada_report.py`, `scripts/check_ada_report.py`)
`check_report(job, report)` runs per-job acceptance rules (A0-A9) and returns PASS / FAIL / LOCAL_DRY_RUN (all checks
pass on an env=local report, only with `--allow-local`; never counts as DONE) / NOT_IMPLEMENTED. Examples of rules:
A2-A4 recall reported and the unique-token / self-retrieval test passed; A5 400 rows, sorted, spot checks 20/20, and a
warning if the indexes hold 0 poison; A6 poison in its target's top-10 for at least one retriever; A8 held-out flag
rate within tolerance of alpha. Rules A0-A7 by Samarth, A8-A9 by Arya. 29 tests cover the rules.

### Ada facts (found by Arya; `docs/ADA_RUNBOOK.md` "How jobs run on Ada")
- QOS: at most 10 CPUs, 1 GPU, 32,000 MB RAM per user, 5 queued + running jobs; memory is tied to CPUs (~3 GB each):
  `--mem=32G` exceeds the cap, so big jobs use **`-c 10 --mem=29G`**. Partition `u22`.
- `/share1` is not visible on compute nodes; home is 50 GB. So all indexes (~19 GB) live on **one node's local
  /scratch (gnode056)**, and every index/cache job is submitted to that node through **`ada/pinned_node.sh`**, which
  syncs the code, stages data in, runs the job and copies reports/logs/caches back to home.
- When gnode056 was full (2026-11-11, queue estimate 1.5 days), the sweeps ran on **gnode082** (same RTX 2080 Ti):
  a 1-CPU job on gnode056 copied what the sweep reads (the 12 GB dense index for document vectors, the bm25 / splade
  `manifest.json` files whose fingerprints the cache loader checks, and the text store) to home, and
  `IRE_STAGE_INDEXES=1 bash ada/pinned_node.sh ...` copied them onto gnode082's /scratch. The sweep reads the top-150
  caches, not the bm25 / splade indexes themselves. The home copy was deleted afterwards (home back to 69%).
- With torch 2.6 + transformers 5.19, `import transformers` fails on a node without a GPU, so every job that loads a
  model must be a GPU job (BM25 and data preparation do not import it).

## 5. Reproducibility -- how we guarantee the same result twice
| Mechanism | Example |
|---|---|
| one global seed (13), **string-keyed RNGs** | `random.Random(f"{seed}:target:{qid}")` -- a question's draw never depends on other questions or on processing order |
| deterministic tie-breaking | every ranking: score desc, then doc_id |
| pinned versions | `requirements.txt` (torch 2.6.0, transformers 5.19.0, faiss-cpu 1.15.1, bm25s 0.3.13, numpy 2.5.4, scikit-learn 1.9.1, ...) and `requirements-lock.txt` |
| pinned model revisions | resolved from the local HF snapshot; recorded in index manifests; loading with another revision fails |
| content fingerprints | indexes (corpus + shard sha256 + settings + revision), caches (+ questions sha256), text store, generation namespace, perplexity namespace |
| frozen inputs | `configs/frozen/*` never edited; LF line endings forced by `.gitattributes` so hashes match on Windows and Linux |
| exact search | Flat dense index, unpruned SPLADE, BM25 full scoring |
| atomic writes | `.building` folders swapped in; `.tmp` files renamed; SQLite transactions per batch |
| batch independence | queries encoded one at a time; one query per FAISS call; clean/poison never share an encoding batch |
| tests | 435 pytest tests (433 on 2026-11-08 + 2 added 2026-11-10; all pass) |
Local and Ada `questions.jsonl` are byte-identical (sha256 0365180a...) -- a direct check that the pipeline reproduces.

## 6. Repository hygiene
- Large files (corpus, indexes, caches, vectors, model weights) live under `$IRE_DATA_DIR`, never in git.
- Git: the team commits under their own names; no AI attribution in commits or files (one early commit, a49ead8,
  still carries such a trailer; rewriting pushed history needs a force push and was left alone).
- `.gitattributes`: `*.sh` LF (they run on Linux), `configs/frozen/**` and `heldout/**` LF, LFS rules for data.

## 7. Likely viva questions
- *How do you know a local result is not confused with an Ada result?* Every report and log line carries `env`; the
  report checker refuses env=local for DONE.
- *What happens if someone changes a model?* The revision is in the index manifest and every namespace: the index
  refuses to load, caches become stale, the generation cache uses a new namespace.
- *Why string-keyed seeds rather than one RNG?* With one sequential RNG, adding or reordering a question changes every
  later draw; keyed RNGs make each draw local and stable.
