# TEAM_STATUS -- who has done what, what is left, and in what order (updated 2026-10-30)

**Reassignment (2026-10-26, agreed with Samarth by phone):** Arya took over Samarth's remaining steps 10
(cluster filter), 11 (perplexity filter + A8), the step 12 *tooling* (template + validator) and 13 (smoke test +
A9). The held-out phrasings themselves (step 12) stay open: they must be written by someone who has not seen the
frozen phrase list (not Arya). Steps 0-9 are Samarth's work.

**Arya's and Samarth's lanes are done.** The held-out phrasings were written by Samarth's side on 2026-10-29
(row 1, DONE) and Arya re-ran A6 -> A5 -> P2 with them on 2026-10-30 (row 2, DONE). Everything else that is left is Vansh's lane. Vansh: start from `VANSH_START.md`.
Details: Samarth's lane in `docs/HANDOFF.md`, Arya's lane (and steps 10-13) in `docs/HANDOFF_ARYA.md`. Plan dates: report
write-up 26 Oct - 1 Nov, buffer 2-5 Nov, submission before 6 Nov.

## Done
| Item | Owner (plan) | Who did it | Evidence |
|---|---|---|---|
| Steps 0-9: repo, config, data loading, BM25 / dense / SPLADE / RRF retrievers, top-150 cache, poison ingestion, generation harness + cache | Samarth | Samarth | `docs/HANDOFF.md` |
| Final question set (200 target + 200 clean, folds A/B) | Arya | Arya | `configs/frozen/question_set.json` |
| Frozen injection-phrase list | Arya | Arya | `configs/frozen/injection_phrases.json` |
| Poison: fact, hijack, url, diverse, repeat, stealth (4,500 passages) + held-out shard (505, phrasings by Samarth's side) | Arya | Arya | `src/ire/poison/`, `reports/poison_ada.json` |
| Query-and-phrase filter + calibration (Ada thresholds) | Arya | Arya | `src/ire/defenses/query_phrase.py`, `configs/query_phrase_thresholds.json` |
| Step 10: cluster filter | Samarth | Arya (reassigned) | `src/ire/defenses/cluster.py`, `scripts/check_cluster.py` |
| Step 11: perplexity filter, score cache, calibration | Samarth | Arya (reassigned) | `src/ire/defenses/perplexity.py`, `configs/perplexity_thresholds.json` (Ada values) |
| Step 12 tooling: held-out template + validator (phrasings written blind by Samarth's side, 2026-10-29) | Samarth | Arya (reassigned) | `scripts/heldout_template.py`, `scripts/validate_heldout.py`, `heldout/phrasings_template.jsonl` |
| Step 13: end-to-end smoke test + `candidate_pool` stub | Samarth | Arya (reassigned) | `scripts/smoke_test.py`, `src/ire/stubs.py` |
| Ada A0 setup, A1 data, A2 BM25, A6 poison ingestion | Arya | Arya | `reports/A0.json`, `A1.json`, `A2.json`, `A6.json` |
| Ada reports A0-A6, A8, A9 re-checked with `scripts/check_ada_report.py` (all PASS); steps 0, 2-5, 7, 8 marked DONE; docs/ADA_RUNBOOK.md updated with the Ada facts; local data rebuilt on the final questions; "Ready for integration" section | Samarth | Samarth | `docs/HANDOFF.md` (2026-10-26) |
| Ada A3 dense, A4 SPLADE, A5 caches | Vansh (runner) | Arya (on Arya's account, to unblock the chain) | `reports/A3.json`, `A4.json`, `A5.json` |
| Ada A8 perplexity (step 11's thresholds + score cache) | Vansh (runner) | Arya (as part of finishing step 11) | `reports/A8.json` |
| Ada A9 smoke test | Arya or Vansh | Arya | `reports/A9.json` |
| Small fixes in Samarth's code found while running Ada (prepare_data builds the frozen set; dense search one query per FAISS call; model-cache lookups; FAISS index loaded only for search; LLM loaded straight onto the GPU) | Samarth | Arya | commits 56d807f, 273e784, 78cbd66, 0214e48, 872c238: **reviewed and kept by Samarth (2026-10-26)**; noted in `docs/HANDOFF.md` "Interface changes" |

## Left, in order
Each row can start once the rows it depends on are done. Rows with the same number can run in parallel.

| # | Task | Owner | Depends on | Notes |
|---|---|---|---|---|
| 1 | Held-out phrasings: fill `heldout/phrasings_template.jsonl` (10 hijack phrasings with `{CW}`, 5 fact passages) WITHOUT opening the phrase list or the attack templates; save as `heldout/phrasings.jsonl`; `python scripts/validate_heldout.py` | Samarth (or Vansh, if he has not seen the list) | -- | **DONE 2026-10-29** (`heldout/phrasings.jsonl`, written blind; `validate_heldout.py --check-only`: valid, 505 passages = 500 hijack + 5 fact; the frozen phrase rule catches 3/10 hijack phrasings, 0/5 fact). |
| 2 | Re-run A6 (ingest the held-out shard), A5 (caches), P2 (query-phrase calibration) on gnode056 | Arya | row 1 (done) | **DONE 2026-10-30** (jobs 48840 / 48841 / 48842; `reports/A6.json`, `reports/A5.json` PASS in `check_ada_report.py`; `reports/query_phrase_ada.json`). 5,005 poison passages in every index; held-out reach top-10 BM25 0.032 / dense 0.109 / SPLADE 0.174; query-and-phrase TPR on held-out at rate 5: 0.47 / 0.49 / 0.44 / 0.44 (bm25 / dense / splade / rrf) vs 0.77-0.79 on hijack. Thresholds unchanged. Dense masking reference regenerated (see `docs/HANDOFF_ARYA.md`). Caches are now final for the sweeps. |
| 1 | `candidate_pool` masking + `is_active` (rate r keeps `poison_<qid>_<attack>_1..r`; RRF: mask both lists, then `fuse`); replaces `src/ire/stubs.py` | **Vansh** | -- | Can start now on the current caches. |
| 1 | Metrics: ASR (target string / codeword / URL domain), EM / F1, nDCG@5/@10, Recall@k, n_poison_top5, collateral poisoning, detection (ROC-AUC, TPR, FPR, gold dropped), backfill, latency / filter overhead | **Vansh** | -- | ASR strings: each record's `target_answer`. |
| 1 | Random filter + oracle filter controls | **Vansh** | -- | Random drops the same number of passages as the detector (plan 4.4). |
| 1 | Ada A7: LLM throughput benchmark + decide the generation batch size (docs/HANDOFF.md open issue) | **Vansh** | -- | `ada/run_generate_benchmark.sh` via `pinned_node.sh`. |
| 3 | Sweep runner: retrievers x attacks x rates x filter settings -> run records (plan section 8) via the generation cache | **Vansh** | candidate_pool, metrics | All three filters are ready (`VANSH_START.md` section 4). |
| 3 | Write the decision rule down before any sweep (plan 5.5) | All, led by **Vansh** | -- | |
| 4 | 500-request pilot | **Vansh** | sweep runner, A7 | Plan 6.1. |
| 5 | Full sweeps (about 60,000 LLM requests at most) | **Vansh** | pilot, row 2 | |
| 5 | Human calibration: 150 answers in pairs, 100 passage labels each; agreement code | All; **Vansh** owns the code | first answers | Plan section 7. |
| 6 | Cross-fitting, McNemar, paired bootstrap, Wilson intervals, Holm correction | **Vansh** | sweeps | |
| 7 | Plots | **Vansh** | stats | |
| 8 | Report write-up | All | results | |

## Ada notes for whoever runs a job
- Everything runs from Arya's account; data in `/home2/arya.mirani/ire_data_dir`, indexes on gnode056's local
  `/scratch` (submit with `-w gnode056` through `ada/pinned_node.sh`).
- Limits: 10 CPUs, 1 GPU, at most 5 queued + running jobs; memory is tied to CPUs (~3 GB each). Check what the
  node has free first: `scontrol show node gnode056 | grep -E "CfgTRES|AllocTRES"` (details in `docs/HANDOFF_ARYA.md`).
  Filter and generation jobs fit in `-c 2 --mem=5G` (the FAISS index is only loaded for searches; the LLM loads
  straight onto the GPU).
- Do not start two jobs that write the same index or cache at the same time.
