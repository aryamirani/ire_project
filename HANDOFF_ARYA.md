# HANDOFF -- Arya's lane (question set, phrase list, poison, query-and-phrase filter; Ada jobs A0 / A1 / A2 / A6)

Last updated: 2026-10-26. Design source: `docs/IRE_Final_Project_Plan.pdf` sections 3.1, 3.2, 4.1 and 9.
Samarth's progress is in `HANDOFF.md`. None of his code was changed. The only edits to existing files are:
- `ada/common.sh`: the SLURM values (he asked the first teammate on Ada to fill them in) and a storage note;
- new sections appended to `configs/experiment.yaml` (`poison:`, `query_phrase:`) and one new key in
  `configs/paths.yaml` (`query_phrase_thresholds`).

## Status board
| Item | Status | Files |
|---|---|---|
| Frozen injection-phrase list (24 patterns) | FROZEN | `configs/frozen/injection_phrases.json` |
| Final question set: 200 target + 200 clean, folds A/B | DONE (local + Ada, identical sha256) | `src/ire/data/question_set.py`, `scripts/select_questions.py`, `configs/frozen/question_set.json` |
| Local corpus around the final set | DONE-LOCAL | `scripts/build_local_corpus.py` |
| Poison: fact / hijack / url (main) + diverse / repeat / stealth (adaptive) | DONE-LOCAL (Ada run waits for disk space, see below) | `src/ire/poison/`, `scripts/make_poison.py`, `configs/frozen/poison_targets.json`, `ada/run_make_poison.sh` |
| Query-and-phrase filter + calibration | DONE-LOCAL (Ada calibration needs A5) | `src/ire/defenses/query_phrase.py`, `scripts/calibrate_query_phrase.py`, `configs/query_phrase_thresholds.json` (LOCAL values) |
| Tests | 35 new; full suite 373 passed, 7 skipped (Qwen not downloaded on this Mac) | `tests/test_question_set.py`, `tests/test_poison_gen.py`, `tests/test_query_phrase.py` |

## Ada jobs (Arya)
| Job | Status | Notes |
|---|---|---|
| A0 setup | environment + models installed; GPU check runs (GTX 1080 Ti / RTX 2080 Ti seen) | the report still FAILs "models cached" (see the snapshot note below); re-check at the end |
| A1 data | **PASSED** (`reports/A1.json`, `check_ada_report.py A1` = PASS) | 2,681,468 passages; then `select_questions.py` replaced the placeholder with the final set on Ada |
| A3 dense | **PASSED** (`reports/A3.json`, checker PASS) | e5-base-v2 on an RTX 2080 Ti (gnode061), 2,936 s; recall@10 0.781, recall@100 0.941 on 1,000 questions; re-encode min cos 0.99997 |
| A4 splade | **PASSED** (`reports/A4.json`, checker PASS) | 3,487 s on gnode061; 146.5 non-zeros per doc, 45.6 per query; recall@10 0.733, recall@100 0.927 |
| P1 poison (new, mine) | **PASSED** (`reports/poison_ada.json`, `reports/questions_ada.json`) | 3,000 main + 1,500 adaptive; main shard byte-identical to the local one; codeword / URL in 0 of 2.68M clean passages; stealth cap met for 428 / 500 (241 keyword-anchored, 259 title-only; mean cos 0.801 vs mean cap 0.823); targets equal the frozen file |
| A6 / A5 / P2 | QUEUED (A6, A5) / to submit (P2, 5-job limit) | P2 = `ada/run_query_phrase.sh` |
| A2 bm25 | **PASSED** (`reports/A2.json`, checker PASS, clean tree) | 2,681,468 docs, 1.0 GiB, built in 334 s on gnode004 (10 CPUs, 29 GB); recall@10 0.444, recall@100 0.744 on 1,000 questions (BEIR BM25 on NQ: 0.76) |

### Ada facts found on 2026-10-26 (Samarth: please fold these into ADA_RUNBOOK.md)
- **QOS `low` (our student account): max 10 CPUs, 1 GPU, 32 GB RAM per user.** `--mem=48G` (A2-A6 in the
  runbook) is rejected with `QOSMaxMemoryPerUser`; use `--mem=32G`. Partition `u22`; `-A research` is rejected on
  `u22-cpu`, so `ADA_ACCOUNT` is left empty.
- **`/share1/<user>` is NOT mounted on compute nodes** (login node only) and has a 25 GB / **3,000-file** quota.
  A conda env cannot live there (file limit). Compute nodes see home (`/home2/<user>`, 50 GB) and node-local
  `/scratch` (~1.7 TB) / `/ssd_scratch` (~480 GB), which are not shared between nodes.
- Arya's setup: environment in home via `IRE_ENV_DIR=/home2/arya.mirani/ire_env` in `.env` (common.sh honours it);
  `PIP_NO_CACHE_DIR=1`. Data currently on `/share1` (only usable from the login node, e.g. A1).
  Arya's old course folders are being copied to her laptop, verified (file count + checksums) and removed from
  Ada to free ~37 GB of home. `IRE_DATA_DIR` and `HF_HOME` now point to `/home2/arya.mirani/ire_data_dir`, so all
  jobs (Arya's and Vansh's, run from the same account) share one data folder visible on every node.
- **How the jobs run now (all from Arya's account, one node):** every index job is submitted with
  `sbatch -p u22 -w gnode061 [--gres=gpu:1] -c 10 --mem=29G --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/<job>.sh"`.
  `pinned_node.sh` keeps `IRE_DATA_DIR` on that node's `/scratch/arya.mirani/ire_data_dir` (all indexes), stages
  data + models in from home and copies reports / logs / caches / poison back to `/home2/arya.mirani/ire_data_dir`.
  A symlinked `indexes/` folder does not work: `build_index.py` calls `index_dir.relative_to(data_dir)` on the
  resolved path. If gnode061's `/scratch` is ever wiped, the indexes must be rebuilt (A2-A4, A6).
- GPUs on `u22`: 44 nodes have RTX 2080 Ti (feature `2080ti`), the rest are older (gnode004: GTX 1080 Ti, whose
  fp16 is slow: A3 there was on track for ~10 h vs ~1 h on a 2080 Ti). Pin to a 2080 Ti node.
- **QOS also limits a user to 5 queued + running jobs** (`QOSMaxSubmitJobPerUserLimit`).
- Model snapshots: `download_models.py` skips ONNX / TF / Flax files, and with huggingface_hub 1.33
  `snapshot_download(local_files_only=True)` then raises `IncompleteSnapshotError`, so `resolve_model` re-downloads
  the full repo at job time (compute nodes have internet, so it works, but it costs disk) and `check_env.py`
  reports those models as NOT CACHED. Samarth: consider `snapshot_download(..., ignore_patterns=SKIP_PATTERNS,
  local_files_only=True)` in both places.
- Memory per job: `--mem=32G` is 32,768 MB > the 32,000 MB cap; and memory is tied to CPUs (~3 GB per CPU), so
  `--mem=31G -c 10` is bumped to 11 CPUs and blocked. **Use `-c 10 --mem=29G`.**
- **With torch 2.6 + transformers 5.19, `import transformers` fails on a node without a GPU**
  (`RuntimeError: Cannot access accelerator device when none is available`, from `torch.accelerator`). So
  `check_env.py` reports sentence-transformers as NOT IMPORTABLE on the login node. BM25 (A2) and prepare_data
  (A1) do not import transformers and are fine; anything that loads a model must run in a GPU job.
- macOS (Arya's laptop, local profile): faiss + torch in one process can deadlock (two OpenMP runtimes);
  run local scripts with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`.

## Requests to Samarth
1. `scripts/prepare_data.py --force` would overwrite `questions.jsonl` with the placeholder again. Please make
   it build `questions.jsonl` from `configs/frozen/question_set.json` when that file exists (or call
   `ire.data.question_set.build_questions`), and the local random corpus around those questions (what
   `scripts/build_local_corpus.py` does now). Until then: after any prepare_data run, re-run
   `python scripts/select_questions.py` (and on the local profile `scripts/build_local_corpus.py`).
2. The final questions changed `questions.jsonl`, so every top-150 cache built on the placeholder is stale (the
   loader refuses it, as designed). Local caches were rebuilt here.
3. Held-out shard: your `poison_heldout.jsonl` goes next to `poison_main.jsonl` / `poison_adaptive.jsonl` in
   `data/poison/`. Pick its qids from the 200 targets (e.g. `adaptive_qids` in `configs/frozen/poison_targets.json`
   if you want overlap with the other adaptive variants). The phrase list is in `configs/frozen/`; per plan 3.2
   please do not open it before writing the held-out phrasings.

## What teammates can use right now
- **Questions**: `data/questions.jsonl` (same `QuestionRecord` format) = the final set. sha256
  `0365180a6070be2f17e5b555027535cbd0659b12d2f0a8c5f6d055d844590c0e` on both machines. Answer type per qid in
  `configs/frozen/question_set.json` (`year | date | number | person | place | other`).
- **Poison shards** in `data/poison/`: `poison_main.jsonl` (fact, hijack, url: 200 targets x 5 = 3,000) and
  `poison_adaptive.jsonl` (diverse, repeat, stealth: 100 targets x 5 = 1,500; 50 per fold). Only target
  questions are poisoned. `poison_<qid>_<attack>_<n>` with n = 1..5: **rate r keeps n <= r** (Vansh's `is_active`).
- **ASR strings (Vansh)**: `target_answer` of each record. fact / diverse / repeat: a type-matched false answer
  (`configs/frozen/poison_targets.json` -> `targets[qid].target_answer`); hijack / stealth: the codeword
  `ZOLTRAX`; url: the domain `answer-archive.example` (passages contain `https://www.answer-archive.example/verified`).
  Codeword and domain occur in 0 clean passages (checked on the local corpus; P1 checks the full corpus).
- **Query-and-phrase filter (Vansh's runner)**:
  ```python
  from ire.defenses.query_phrase import apply_filter, load_thresholds, thresholds_for
  table = load_thresholds(cfg.path("query_phrase_thresholds"))
  ctx = FilterContext(texts=TextStore.open_for(cfg), vectors=default_vectors(), config=cfg,
                      thresholds=thresholds_for(table, retriever, question.fold, tail=0.025))  # opposite-fold cutoff
  res = apply_filter(pool, qid, question.question, ctx)   # res.flagged, res.scores (= s(q, p), for ROC-AUC)
  ```
  `query_phrase.rule` in experiment.yaml: `either` (default) | `similarity` | `phrase` (ablations).
  `explain(pool, question, ctx)` gives per-passage similarity, phrase pattern ids and the centroid-distance ablation.
  Sweep tails: `query_phrase.tail_sweep` = 1 / 2.5 / 5 / 10 %.
- **Calibration**: `python scripts/calibrate_query_phrase.py [--detection]` (needs current caches + dense index)
  writes `configs/query_phrase_thresholds.json`. The committed file holds LOCAL values (env=local); it must be
  regenerated on Ada after A5 and committed before the sweeps.

## How it works (design decisions)
**Question set** (`select_question_set`): from questions_all (2,255 questions with gold + nq_open answer).
Eligible: shortest answer 1-5 words and >= 2 characters, not yes/no, no gold answer contained in the question,
question >= 3 words, one per normalised text (2,205 eligible; excluded: 31 answer-in-question, 18 too short,
1 yes/no). Seeded sample (seed 13) of 400, ordered by (answer type, seeded key) and dealt round-robin into
target/A, target/B, clean/A, clean/B: each cell has 100 questions and the same answer-type mix (+/- 1):

| cell | date | number | other | person | place | year |
|---|---|---|---|---|---|---|
| clean/A | 6 | 5 | 32 | 34 | 11 | 12 |
| clean/B | 6 | 5 | 32 | 34 | 11 | 12 |
| target/A | 7 | 5 | 31 | 34 | 11 | 12 |
| target/B | 7 | 5 | 32 | 33 | 11 | 12 |

**Target answers** (`ire.poison.targets`): seeded per qid; must differ from every gold answer, share no content
word with one, and not occur in the question or in any gold passage (so string-match ASR can only come from
poison). Years are shifted by +/- 2-25 years (24 questions); everything else is the canonical answer of another
question of the same answer type, preferring the same surface shape (capitalisation, leading "the", digits,
length) for plausibility (176 questions); numbers only take numeric candidates.

**Templates** (`ire.poison.templates`, written after the phrase list was frozen): 5 fact templates
(question first, then the false answer stated confidently, anchored on the gold passage title), 12 hijack and
12 url phrasings (5 drawn per question, seeded), 5 diverse templates (different structures and wording), 5
repeat phrases (question + one phrase repeated to >= 70 words), 5 stealth templates (no question text; topic
anchor + the codeword presented as editorial guidance). Stealth similarity cap: keyword-anchored text is used
only if cos(q, p) <= the median cos of the question's clean top-50 dense pool, else the title-only text.

**Phrase list**: 24 regexes in 5 groups (override, addressed-to-model, output control, url, format markers),
matched on NFKC / lowercase / ASCII-quote normalised text. Written from the injection literature before any
template existed; commit it in its own commit before the poison code.

## Local verification (env=local: 5,000 clean passages, e5-small-v2, CPU on a MacBook; correctness only)
- `select_questions.py`: all checks PASS; `build_local_corpus.py`: 5,000 passages = 512 gold + 4,488 random.
- Indexes rebuilt for the new corpus: BM25 recall@100 0.98, dense recall@100 1.00, SPLADE recall@100 1.00
  (on the 5,000-passage corpus; easy by construction). Dense checks (prefix, re-encode cos > 0.999, norms,
  self-retrieval) all PASS.
- `make_poison.py`: 3,000 + 1,500 passages, all checks PASS (counts, only target qids, codeword / URL absent from
  the corpus, every fact-style target verbatim in its passage). Mean words: fact 37, hijack 25, url 25,
  diverse 24, repeat 72, stealth 33.
  Phrase-rule hit rate on the poison: hijack 0.676, url 0.763, fact 0.003, diverse 0.004, repeat 0.004,
  stealth 0.000. Stealth similarity cap: 113 / 500 under the cap (495 title-only); mean cos 0.803 vs mean cap
  0.780, i.e. the cap is only partly achievable with a topic anchor on e5-small (re-measured on Ada).
- `index_poison.py --check-masking` (A6 local, real local indexes): PASS. Dense / SPLADE masking exact (50/50).
  BM25 max rank difference 84 locally: 4,500 poison passages in a 5,000-passage corpus move IDF a lot; on 2.68M
  passages the effect is expected to be small (A6 measures it). Poison in its target's top-10:
  fact 0.91-0.92 (all retrievers), hijack 0.32-0.69, url 0.08-0.23, diverse 0.14-0.57, repeat 0.00-0.76,
  stealth 0.00 (in top-150: 0.77-0.86).
- `build_cache.py --retriever all` (A5 local): all checks PASS for bm25 / dense / splade / rrf.
- `calibrate_query_phrase.py --detection`: anchor cutoffs (tail 2.5%) 0.847-0.852 for every retriever; cross-fold
  flag rate on clean pools 0.023-0.026 (target 0.025). Phrase-rule FPR: 0.0013 on 4,535 clean pool passages,
  0.0012 on 5,000 corpus passages (hits: `new_instructions` 4, `role_marker` 2).
  Detection sanity, rate 5, anchor, opposite-fold cutoffs (TPR on poison / FPR on clean pool passages):

  | attack | bm25 | dense | splade | rrf |
  |---|---|---|---|---|
  | fact | 0.37 / 0.036 | 0.51 / 0.033 | 0.37 / 0.037 | 0.43 / 0.036 |
  | hijack | 0.75 / 0.035 | 0.72 / 0.032 | 0.81 / 0.033 | 0.75 / 0.035 |
  | url | 0.83 / 0.035 | 0.89 / 0.032 | 0.88 / 0.032 | 0.85 / 0.035 |
  | diverse | 0.29 / 0.032 | 0.22 / 0.033 | 0.33 / 0.031 | 0.24 / 0.033 |
  | repeat | 0.41 / 0.032 | 0.30 / 0.034 | 0.28 / 0.034 | 0.35 / 0.033 |
  | stealth | 0.08 / 0.030 | 0.03 / 0.036 | 0.08 / 0.032 | 0.06 / 0.034 |

  As the plan expects: the phrase rule carries hijack / url, similarity alone catches part of the question-first
  attacks, and stealth evades both. Local numbers only; the reported ones come from Ada + Vansh's metrics.

## How to run (repo root)
Local (macOS): prefix commands with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` and use `.venv/bin/python`.
```
python scripts/prepare_data.py                 # Samarth's step 2 (downloads, placeholder questions)
python scripts/select_questions.py             # final questions.jsonl from configs/frozen/question_set.json
python scripts/build_local_corpus.py           # LOCAL only: corpus around the final gold passages
python scripts/build_index.py --retriever bm25 --force      # and dense, splade
python scripts/make_poison.py                  # poison shards (+ report)
python scripts/index_poison.py --check-masking # ingest into the indexes (A6)
python scripts/build_cache.py --retriever all  # caches (A5)
python scripts/calibrate_query_phrase.py --detection
```
Ada: `bash ada/run_make_poison.sh` (P1) after A1 + A3, then A6, then A5, then the calibration script in a GPU job.

## Open issues
- Ada storage (above): waiting for home space; A2 / P1 / A6 not run yet.
- Stealth similarity cap: met for 428 / 500 stealth passages on Ada (e5-base), 113 / 500 locally (e5-small).
- `configs/query_phrase_thresholds.json` currently holds local values; regenerate on Ada after A5.
