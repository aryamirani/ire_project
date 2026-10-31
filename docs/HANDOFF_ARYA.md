# HANDOFF -- Arya's lane (question set, phrase list, poison, query-and-phrase filter; Ada jobs A0 / A1 / A2 / A6)

Last updated: 2026-10-31. Design source: `docs/IRE_Final_Project_Plan.pdf` sections 3.1, 3.2, 4.1 and 9.
Samarth's progress is in `docs/HANDOFF.md`. Edits to files that already existed before this lane:
- `ada/common.sh`: the SLURM values (he asked the first teammate on Ada to fill them in) and a storage note;
- new sections appended to `configs/experiment.yaml` (`poison:`, `query_phrase:`) and one new key in
  `configs/paths.yaml` (`query_phrase_thresholds`);
- small fixes in Samarth's code found while running Ada (commits 56d807f, 273e784, 78cbd66, 0214e48, 872c238),
  reviewed and kept by Samarth (see "Steps 10-13" below and `docs/HANDOFF.md`, "Interface changes");
- steps 10-13, taken over with Samarth's agreement (section "Steps 10-13").

## Status board
| Item | Status | Files |
|---|---|---|
| Frozen injection-phrase list (24 patterns) | FROZEN | `configs/frozen/injection_phrases.json` |
| Final question set: 200 target + 200 clean, folds A/B | DONE (local + Ada, identical sha256) | `src/ire/data/question_set.py`, `scripts/select_questions.py`, `configs/frozen/question_set.json` |
| Local corpus around the final set | DONE-LOCAL | `scripts/build_local_corpus.py` |
| Poison: fact / hijack / url (main) + diverse / repeat / stealth (adaptive) | DONE (P1 on Ada PASSED; held-out shard added 2026-10-30) | `src/ire/poison/`, `scripts/make_poison.py`, `configs/frozen/poison_targets.json`, `ada/run_make_poison.sh` |
| Query-and-phrase filter + calibration | DONE (P2 on Ada; re-run 2026-10-30 with the held-out shard, cutoffs unchanged) | `src/ire/defenses/query_phrase.py`, `scripts/calibrate_query_phrase.py`, `configs/query_phrase_thresholds.json` (Ada values) |
| Tests | 35 new; full suite 373 passed, 7 skipped (Qwen not downloaded on this Mac) | `tests/test_question_set.py`, `tests/test_poison_gen.py`, `tests/test_query_phrase.py` |

## Ada jobs (Arya)
| Job | Status | Notes |
|---|---|---|
| A0 setup | **PASSED** (`reports/A0.json`, checker PASS) | environment in home, all 4 models cached (after the model-cache lookup fix); this check ran on a GTX 1080 Ti node |
| A1 data | **PASSED** (`reports/A1.json`, `check_ada_report.py A1` = PASS) | 2,681,468 passages; then `select_questions.py` replaced the placeholder with the final set on Ada |
| A3 dense | **PASSED** (`reports/A3.json`, checker PASS) | e5-base-v2 on an RTX 2080 Ti (gnode061), 2,936 s; recall@10 0.781, recall@100 0.941 on 1,000 questions; re-encode min cos 0.99997 |
| A4 splade | **PASSED** (`reports/A4.json`, checker PASS) | 3,487 s on gnode061; 146.5 non-zeros per doc, 45.6 per query; recall@10 0.733, recall@100 0.927 |
| P1 poison (new, mine) | **PASSED** (`reports/poison_ada.json`, `reports/questions_ada.json`) | 3,000 main + 1,500 adaptive; main shard byte-identical to the local one; codeword / URL in 0 of 2.68M clean passages; stealth cap met for 428 / 500 (241 keyword-anchored, 259 title-only; mean cos 0.801 vs mean cap 0.823); targets equal the frozen file |
| A6 poison ingestion | **PASSED** (`reports/A6.json`, checker PASS) | 574 s; 4,500 poison passages in all three indexes (5,005 after the held-out re-run on 2026-10-30, see below). Masking: dense / SPLADE exact (50/50); BM25 max rank diff 32, mean 0.40, top-5 unchanged 96%, Jaccard@50 0.995. Clean questions whose top-5 changed: BM25 2%, dense 0%, SPLADE 1%. Poison in its target's top-10: fact 0.92 / 0.75 / 0.87 (BM25 / dense / SPLADE), hijack 0.61 / 0.24 / 0.28, url 0.08 / 0.31 / 0.16, diverse 0.56 / 0.43 / 0.12, repeat 0.00 / 0.28 / 0.74, stealth 0.00 / 0.00 / 0.01 |
| A5 caches | **PASSED** (`reports/A5.json`, checker PASS) | 400 final questions x bm25 / dense / splade / rrf over the poisoned indexes (rebuilt 2026-10-30 with 5,005 poison each), spot checks 20/20 everywhere. Built with `bash ada/run_cache.sh all --batch-size 1 --force`: the first run (default batch 32) failed the dense spot check (17/20 rows equal, scores differ by ~2e-7), because FAISS uses a BLAS path for batches of >= 20 queries and a direct path for single queries, so near-ties reorder. Samarth: consider making batch size 1 the default for dense, or comparing with a tolerance. |
| P2 query-phrase calibration | **DONE** (`reports/query_phrase_ada.json`, `configs/query_phrase_thresholds.json` = Ada values) | e5-base-v2, 232 s. Anchor cutoffs (tail 2.5%): 0.870-0.878 for every retriever and fold. Cross-fold flag rate on clean pools 1.8-3.7% (target 2.5%; 100 clean questions per fold, so some fold-to-fold variation). Phrase-rule FPR: 0.0011 on 21,037 clean pool passages, 0.0013 on 20,000 random corpus passages. Detection sanity at rate 5 (TPR, BM25 / dense / SPLADE / RRF): url 0.85 / 0.84 / 0.84 / 0.84, hijack 0.79 / 0.77 / 0.78 / 0.77, fact 0.62 / 0.57 / 0.58 / 0.56, diverse 0.61 / 0.51 / 0.54 / 0.51, repeat 0.39 / 0.35 / 0.34 / 0.33, stealth 0.01 / 0.01 / 0.00 / 0.00; clean-passage FPR 0.030-0.036. NOTE: pinned_node.sh does not copy repo files back, so the thresholds file was taken from `~/ire_node/configs/`. |
| A2 bm25 | **PASSED** (`reports/A2.json`, checker PASS, clean tree) | 2,681,468 docs, 1.0 GiB, built in 334 s on gnode004 (10 CPUs, 29 GB); recall@10 0.444, recall@100 0.744 on 1,000 questions (BEIR BM25 on NQ: 0.76) |

### Ada notes
General Ada facts (limits, storage, pinned node, what to check before submitting) are in `docs/ADA_RUNBOOK.md`,
"How jobs run on Ada". Three things that are only recorded here:
- **Pinned node moved gnode061 -> gnode056 (2026-10-26)**: gnode061's GPUs were taken by other users. The 19 GB of
  indexes were copied in two rounds through home with CPU-only jobs and verified file by file with sha256
  (`/home2/arya.mirani/ire_migrate/{out,in}.sh`). To move again: the same two scripts, or rebuild (A2-A4, A6, A5:
  about 2.5 h of jobs).
- Model snapshots: `download_models.py` skips ONNX / TF / Flax files, and with huggingface_hub 1.33
  `snapshot_download(local_files_only=True)` then raises `IncompleteSnapshotError`, so `resolve_model` re-downloads
  the full repo at job time and `check_env.py` reports those models as NOT CACHED. Possible fix:
  `snapshot_download(..., ignore_patterns=SKIP_PATTERNS, local_files_only=True)` in both places.
- macOS (local profile): faiss + torch in one process can deadlock (two OpenMP runtimes); run local scripts and
  pytest with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`.

## Steps 10-13 (Samarth's, taken over by Arya on 2026-10-26 with his agreement)
| Step | Files | Status |
|---|---|---|
| 10 cluster filter | `src/ire/defenses/cluster.py`, `tests/test_cluster.py`, `scripts/check_cluster.py` | DONE-LOCAL; runs inside A9 on Ada |
| 11 perplexity filter + score cache | `src/ire/defenses/perplexity.py`, `tests/test_perplexity.py`, `scripts/calibrate_perplexity.py`, `ada/run_ppl.sh`, `configs/perplexity_thresholds.json` (Ada values) | DONE (A8 PASSED) |
| 12 held-out tooling | `src/ire/poison/heldout.py`, `scripts/heldout_template.py`, `scripts/validate_heldout.py`, `heldout/phrasings_template.jsonl` | DONE: phrasings written blind by Samarth's side (`heldout/phrasings.jsonl`, 2026-10-29), ingested by A6 on 2026-10-30 |
| 13 smoke test | `scripts/smoke_test.py`, `src/ire/stubs.py` (candidate_pool stub), `ada/run_smoke.sh` | DONE (A9 PASSED) |

- **Cluster filter** (plan 4.2): k-means k=2 (seeded) on the dense vectors of the top-20 pool passages; a cluster
  of >= 2 is flagged only if mean pairwise cosine >= tau_c (anchor 0.85; `ctx.thresholds["tau_c"]` for the sweep)
  and mean pairwise ROUGE-L >= 0.25 (own LCS implementation). `scores` = the cluster's mean cosine; `backfill()`
  reports survivors from beyond rank 20. Local check (5,000 passages): clean questions flagged 0-0.5%; rate-5
  detection ~0.5 for fact / hijack / url / repeat, diverse 0.04-0.21; ~50 ms per question.
- **Perplexity filter** (plan 4.3): GPT-2 small, token-midpoint split, f = mean NLL of each half scored after
  `<|endoftext|>`, PD = f(pre) - f(post) (two-tailed band), PM = max (one-tailed); passages < 20 tokens never
  flagged; scores cached per doc_id (SQLite); batches capped at 8,192 padded tokens. **A8 on Ada**:
  thresholds at alpha 0.025 {"pd_low": -1.9279330670833588, "pd_high": 1.565777567029, "pm_high": 6.501932060718536}; held-out flag rate PD
  0.0396, PM 0.0236 (tolerance
  +/-0.018; includes the calibration sample's variance; PD at the anchor is noisy with 1,000
  calibration passages, ~12 per tail). Passage length 102.8 tokens on average,
  7.3% too short. 37,153 pool + poison passages scored.
  Detection at the anchor (rule either): {'diverse': 0.1442, 'fact': 0.125, 'hijack': 0.138, 'repeat': 0.054, 'stealth': 0.04, 'url': 0.064} vs clean pool
  0.0188 -- weak, as the plan's counter-evidence predicted.
- **Smoke test, A9 on Ada**: 20 questions (10 target + 10 clean) x 4 retrievers, stub pool (no masking), all three
  filters, Qwen2.5-3B: every check PASS, runtime 143.7 s. Flagged per filter (bm25): {'query_phrase': 154, 'cluster': 187, 'perplexity': 36}.
- Code changes made on the way (all small, in Samarth's files, listed for his review): FAISS index loaded only when
  a search runs (`DenseRetriever.index`), LLM loaded straight onto the GPU (`device_map`), plus the three fixes
  above (prepare_data / dense batching / model-cache lookups).

- **Held-out re-run, 2026-10-30** (A6 48840 -> A5 48841 -> P2 48842, all PASS): `validate_heldout.py` wrote
  505 passages (500 hijack + 5 fact; the frozen phrase rule catches 3/10 phrasings). Held-out reach: top-10 BM25
  0.0317, dense 0.1089, SPLADE 0.1743; top-150 1.0 / 0.9109 / 1.0. Query-and-phrase at rate 5 on held-out: TPR
  bm25 0.4706, dense 0.4886, splade 0.44, rrf 0.4351 (clean FPR 0.029-0.033); cutoffs unchanged.
  The first A6 attempt (48705) failed `dense_masking_exact` (48/50 identical, one near-tie swap in 2 questions,
  rel. score diff 5.8e-7): the saved clean reference `masking_reference_dense.json` dated from 2026-10-26, before
  dense search went one query per call (273e784), so it was built with batched search. It was moved aside to
  `masking_reference_dense.batched-20261026.json` (kept, not deleted) and A6 rebuilt it: now 50/50, diff 0.0.

## Requests to Samarth
Status (Samarth, 2026-10-26): 1 is done by commit 56d807f (prepare_data builds the final questions and the local
corpus around them; re-run locally, questions.jsonl is byte-identical to Ada's); 2 is done (local indexes and
caches rebuilt on the final set); 3 waits for the held-out phrasings. The Ada facts below are now in
docs/ADA_RUNBOOK.md ("How jobs run on Ada"), and the fixes in Samarth's files are reviewed and kept.
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
  writes `configs/query_phrase_thresholds.json`. The committed file holds the **Ada** values (env=ada, e5-base-v2,
  final questions, poisoned indexes); re-run P2 only if the caches are rebuilt.

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
- `configs/query_phrase_thresholds.json` now holds the Ada values (calibrated on the final caches).
