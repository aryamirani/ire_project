# TEAM_STATUS -- who has done what, what is left, and in what order (2026-10-26)

Owners follow the plan's work split (`docs/IRE_Final_Project_Plan.pdf`, section 9). Details: Samarth's lane in
`HANDOFF.md`, Arya's lane in `HANDOFF_ARYA.md`. Plan dates: report write-up 26 Oct - 1 Nov, buffer 2-5 Nov,
submission before 6 Nov, so the sweeps have to start within days.

## Done
| Item | Owner (plan) | Who did it | Evidence |
|---|---|---|---|
| Steps 0-9: repo, config, data loading, BM25 / dense / SPLADE / RRF retrievers, top-150 cache, poison ingestion, generation harness + cache | Samarth | Samarth | `HANDOFF.md` |
| Final question set (200 target + 200 clean, folds A/B) | Arya | Arya | `configs/frozen/question_set.json` |
| Frozen injection-phrase list | Arya | Arya | `configs/frozen/injection_phrases.json` |
| Poison: fact, hijack, url, diverse, repeat, stealth (4,500 passages) | Arya | Arya | `src/ire/poison/`, `reports/poison_ada.json` |
| Query-and-phrase filter + calibration (Ada thresholds) | Arya | Arya | `src/ire/defenses/query_phrase.py`, `configs/query_phrase_thresholds.json` |
| Ada A0 setup, A1 data, A2 BM25 | Arya | Arya | `reports/A1.json`, `reports/A2.json` |
| Ada A3 dense, A4 SPLADE, A5 caches | Vansh | Arya (on Arya's account, to unblock the chain) | `reports/A3.json`, `A4.json`, `A5.json` |
| Ada A6 poison ingestion | Arya | Arya | `reports/A6.json` |

## Left, in order
Each row can start once the rows it depends on are done. Rows with the same number can run in parallel.

| # | Task | Owner | Depends on | Notes |
|---|---|---|---|---|
| 1 | Step 12: held-out phrasings (10 hijack + 5 fact passages, written WITHOUT opening the phrase list) -> `poison_heldout.jsonl` | Samarth | -- | Do this first: adding a shard changes the indexes, so A6, A5 and P2 must be re-run afterwards (row 3). |
| 1 | `candidate_pool` masking + `is_active` (rate r keeps `poison_<qid>_<attack>_1..r`; RRF: mask both lists, then `fuse`) | **Vansh** | -- | Can start now on the current Ada caches (`load_cache`). |
| 1 | Metrics: ASR (target string / codeword / URL domain in the answer), EM / F1, nDCG@5/@10, Recall@k, n_poison_top5, collateral poisoning, detection (ROC-AUC, TPR, FPR, gold dropped), latency / filter overhead | **Vansh** | -- | ASR strings are each record's `target_answer`; per-question targets in `configs/frozen/poison_targets.json`. |
| 1 | Random filter + oracle filter controls | **Vansh** | -- | Random drops the same number of passages as the detector under test (plan 4.4). |
| 1 | Ada A7: LLM throughput benchmark (`ada/run_generate_benchmark.sh`) + decide generation batch size (HANDOFF.md open issue) | **Vansh** | -- | Run on Arya's account with `pinned_node.sh` on gnode061 (see `HANDOFF_ARYA.md`). |
| 2 | Step 10: cluster filter | Samarth | -- | |
| 2 | Step 11: perplexity filter + score cache; then Ada A8 (`run_ppl.sh`) | Samarth (code), **Vansh** (A8 run) | -- | |
| 3 | Re-run A6 (ingest held-out shard), A5 (caches), P2 (query-phrase calibration) | Arya | row 1 (held-out shard) | ~20 min on gnode061 with the existing indexes. |
| 4 | Step 13: end-to-end smoke test; Ada A9 | Samarth (code), Arya or Vansh (run) | rows 1-3, cluster + perplexity filters | |
| 4 | Sweep runner: retrievers x attacks x rates x filter settings -> run records (plan section 8) via the generation cache | **Vansh** | candidate_pool, metrics, all three filters | |
| 5 | 500-request pilot (throughput, cache hit rate, failure modes) | **Vansh** (+ Arya to run on Ada) | sweep runner, A7 | Plan 6.1; decides the cut order if compute is short. |
| 5 | Write down the decision rule before any sweep (headline anchor, what "helps" means, what we say if not) | All, led by **Vansh** | -- | Plan 5.5. |
| 6 | Full sweeps (about 60,000 LLM requests upper bound) | **Vansh** (runs), Arya (Ada support) | pilot | |
| 6 | Human calibration: 150 answers in pairs (Arya+Samarth, Samarth+Vansh, Vansh+Arya), 100 passage labels each; Cohen's kappa, judge accuracy | All; **Vansh** owns the agreement code | first generated answers | Plan section 7. |
| 7 | Cross-fitting (choose on fold A, report on B, swap), McNemar for ASR, paired bootstrap for nDCG@5, Wilson intervals, Holm correction | **Vansh** | sweeps | |
| 8 | Plots: ASR vs rate per retriever, detector ROC, ASR vs n_poison_top5 (mechanism check), complementarity table | **Vansh** | stats | |
| 9 | Report write-up | All | results | Each person writes their lane's method section. |

## Ada notes for whoever runs a job
- Everything runs from Arya's account; data in `/home2/arya.mirani/ire_data_dir`, indexes on gnode061's local
  `/scratch` (submit with `-w gnode061` through `ada/pinned_node.sh`).
- Limits: 10 CPUs, 1 GPU, `--mem=29G`, at most 5 queued + running jobs.
- Do not start two jobs that write the same index or cache at the same time.
