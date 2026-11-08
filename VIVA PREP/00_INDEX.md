# VIVA PREP -- Measuring RAG Safety Against Bad Data

A revision series for the viva. Read it in order the first time; afterwards use the question bank (file 16) to
test yourself and jump back to the file that answers a question you could not answer.

Every number here comes from a committed report (`reports/*.json`), `docs/HANDOFF.md`, `docs/HANDOFF_ARYA.md` or
`TEAM_STATUS.md`. Numbers marked **env=local** come from Samarth's laptop (5,000-passage corpus, small models) and
are for correctness only; everything else is from the Ada cluster on the full 2.68M-passage corpus. The sweeps
(the final experiment) have **not** run yet at the time of writing (2026-11-08), so there are no final ASR results:
never quote a result that is not in these files.

| # | File | What it covers | Read when |
|---|---|---|---|
| 01 | `01_Project_Overview.md` | problem, threat model, research questions, what is new, scope of claims, team split | first |
| 02 | `02_Architecture_and_Data_Flow.md` | the 7-stage pipeline, 150 -> 50 -> 5, one index for all poison, an end-to-end walk-through of one question | first |
| 03 | `03_Data_and_Questions.md` | BEIR NQ + nq_open, the 400 questions, folds, eligibility, answer types, local corpus | |
| 04 | `04_Retrievers.md` | BM25 (bm25s), dense E5 + FAISS Flat, SPLADE, RRF: formulas, code, choices, numbers | your lane |
| 05 | `05_Cache_Poison_Ingestion_Masking.md` | top-150 cache, fingerprints, poison ingestion, masking equivalence, candidate pool | your lane |
| 06 | `06_Attacks_and_Poison.md` | fact / hijack / url / diverse / repeat / stealth / heldout, target answers, templates | |
| 07 | `07_Defenses.md` | query-and-phrase, cluster, perplexity filters; random and oracle controls | your lane (cluster, perplexity) |
| 08 | `08_Generation.md` | Qwen2.5, prompt, chat template, greedy decoding, generation cache, batching | your lane |
| 09 | `09_Metrics_and_Statistics.md` | ASR, EM/F1/acc, nDCG, detection metrics, Wilson, McNemar, bootstrap, Holm, kappa | |
| 10 | `10_Sweep_Analysis_Decision_Rule.md` | sweep grid, run records, analysis tables, cross-fitting, frozen decision rule, human calibration | |
| 11 | `11_Infrastructure_and_Reproducibility.md` | config system, local vs Ada, job reports, seeds, pinning, Ada constraints | |
| 12 | `12_Code_Map.md` | every file in the repo, what it does and why it matters | before the viva |
| 13 | `13_Decisions_and_Tradeoffs.md` | every design decision with the alternative and the trade-off | before the viva |
| 14 | `14_Results_and_Numbers.md` | all measured numbers in one place | night before |
| 15 | `15_Bugs_Fixes_Lessons.md` | bugs found and fixed, and what they teach | likely viva material |
| 16 | `16_Viva_Question_Bank.md` | 125 deep questions with model answers | repeatedly |
| 17 | `17_Concepts_Refresher.md` | the theory: BM25, E5, SPLADE, RRF, FAISS, k-means, ROUGE-L, perplexity, statistics | if a concept is shaky |

## The project in five sentences
1. We build one RAG pipeline over BEIR Natural Questions (2,681,468 passages) with four retrievers (BM25, dense
   e5-base-v2, SPLADE, BM25+dense RRF) and a fixed LLM (Qwen2.5-3B-Instruct, greedy).
2. We inject 5,005 poisoned passages (fact distortion, instruction hijack, URL injection, three adaptive
   variants and a held-out set written blind) aimed at 200 target questions; 200 clean questions measure
   collateral damage.
3. Between retrieval and generation we put three training-free post-retrieval detectors, each aimed at a
   different property of poison: query-and-phrase (poison mimics the question / contains instructions), cluster
   (poison passages are near-duplicates), perplexity (machine text has a fluency break at the junction).
4. We compare each detector against no filter, a random filter that drops the same number of passages and an
   oracle that drops exactly the poison, in a detector view (TPR/FPR/AUC) and a defense view (ASR, nDCG, EM/F1),
   with McNemar tests, bootstrap intervals, Holm correction and cross-fitting over two question folds.
5. We do not claim robustness: the claim is "on this corpus, this LLM and these attacks, detector X changes ASR
   by Y with interval Z", and every detector has a named adaptive attack designed to defeat it.

## Who did what (be ready to say this precisely)
- **Samarth** (you): steps 0-9 of the implementation plan -- repo skeleton, config system, data loading, the
  BM25 / dense / SPLADE / RRF retrievers, the top-150 cache, poison ingestion and the masking-equivalence check,
  the generation harness and cache, the Ada report checker -- plus the held-out phrasings (written blind),
  the audit of steps 10-13 and the bug fixes listed in file 15.
- **Arya**: question set and folds, poison generation (all seven attacks), the frozen phrase list, the
  query-and-phrase filter; by agreement with Samarth also steps 10-13 (cluster filter, perplexity filter + A8,
  held-out tooling, smoke test + A9); ran every Ada job (A0-A6, A8, A9).
- **Vansh**: candidate-pool masking, metrics, random/oracle controls, sweep runner, statistics, analysis, plots,
  human-calibration tools, the decision rule draft (agreed by all three and frozen).

If asked about a part you did not write, say who wrote it and then explain it -- you are expected to understand
the whole system.
