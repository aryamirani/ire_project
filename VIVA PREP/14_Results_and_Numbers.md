# 14 -- All measured numbers (as of 2026-11-11)

**The experiment has run.** The Ada pilot (2026-11-10) and all 384 sweep settings (2026-11-11) finished with no
failures; the final results are at the end of this file ("Sweep results"). The decision-rule verdicts there come from
`scripts/analyze.py` run on the pushed run records (`reports/runs_main.tar.gz`); if Vansh's final tables differ, his
win. Everything before "Sweep results" is the pre-experiment evidence (retrieval quality, poison reach, masking,
calibration, smoke test). Source in brackets.

## Data (A1, data card, questions report)
- Corpus 2,681,468 passages (1.44 GiB jsonl), 0 duplicate ids. 3,452 BEIR test queries; 4,201 gold ids, all present.
- 2,255 questions joined to nq_open answers (65.3%) = `questions_all`.
- Final 400: 100 per cell (target/A, target/B, clean/A, clean/B); per cell ~34 person, 32 other, 12 year, 11 place,
  6-7 date, 5 number; 512 gold ids, all in the corpus. sha256 of questions.jsonl 0365180a... (Ada == local).

## Retrieval (A2-A4; 1,000 seeded questions from questions_all)
| | Recall@10 | Recall@100 | hit@10 | hit@100 | build | size | notes |
|---|---|---|---|---|---|---|---|
| BM25 | 0.4442 | 0.7442 | 0.473 | 0.768 | 333.9 s | 1,003 MiB | vocab 892,751; unique token at rank 1; 12.5 ms/query |
| dense e5-base-v2 | 0.7807 | 0.9409 | 0.819 | 0.949 | 2,775 s | 11,886 MiB | 999.8 passages/s; VRAM 1.28 GiB; self-retrieval 99/100; re-encode min cos 0.999973 |
| SPLADE | 0.7327 | 0.9270 | 0.768 | 0.938 | 3,261 s | 6,107 MiB | 844 passages/s; 392.9M non-zeros; 146.5 terms/doc, 45.6/query; self-retrieval 95/100; exactness 2.6e-7 |
On the 400 placeholder questions: BM25 0.4496/0.7533, dense 0.7438/0.9333, SPLADE 0.7171/0.9342.
Local (5,000 passages, env=local, correctness only): BM25 0.9494/0.9804, dense 0.9992/1.0, SPLADE 0.9979/1.0.

## Poison (poison_ada.json, A6)
- 4,500 main+adaptive passages + 505 held-out = 5,005 in every index; per target question 15-35.
- Mean words: fact 32.7, hijack 24.4, url 24.0, diverse 22.3, repeat 72.1, stealth 30.6.
- Phrase-rule hit rate: hijack 0.676, url 0.763, fact 0.003, diverse 0.004, repeat 0.004, stealth 0.0.
- Stealth: 428/500 under the similarity cap; anchors 241 keyword / 259 title-only; mean cos 0.801 vs cap 0.823.
- Targets: 176 drawn from the same-type pool, 24 year shifts; codeword and URL absent from the corpus.
- Held-out: phrase rule catches 3/10 phrasings, 0/5 fact passages.

## Masking and reach (A6, 2026-10-30, 50 questions for masking)
- Masking: dense and SPLADE 50/50 identical (exact); BM25 max rank diff 36, mean 0.426, top-5 unchanged 96%,
  top-50 unchanged 30%, Jaccard@50 0.9945 (min 0.961).
- Unmasked top-5 changed vs clean: all questions bm25 0.51 / dense 0.48 / splade 0.50; clean questions 0.02 / 0.00 / 0.01.
- Reach (own target's top-10), BM25 / dense / SPLADE: fact 0.915/0.735/0.846; hijack 0.602/0.228/0.264;
  url 0.081/0.296/0.141; diverse 0.554/0.422/0.112; repeat 0.002/0.268/0.698; stealth 0.000/0.000/0.006;
  heldout 0.032/0.109/0.174.

## Caches (A5, 2026-10-30)
400 rows per retriever; spot checks 20/20 for all four; dense score_docs max diff 4.8e-7; runtime 249 s.

## Query-and-phrase (P2, query_phrase_ada.json)
- Cutoffs at tail 2.5%: ~0.870-0.878 (dense A 0.8783, B 0.8753; bm25 A 0.8756, B 0.8702).
- Phrase-rule FPR: 0.109% of 21,038 clean pool passages; 0.13% of 20,000 corpus passages.
- Detection sanity at rate 5 (TPR; FPR ~0.03): url 0.84-0.85, hijack 0.77-0.79, fact 0.56-0.62, diverse 0.50-0.61,
  heldout 0.44-0.49, repeat 0.33-0.39, stealth 0.00-0.01.

## Perplexity (A8)
- Thresholds alpha 0.025: PD in [-1.928, 1.566], PM <= 6.502 (others: alpha 0.01 PD [-2.442, 1.804], PM 6.964; 0.05
  [-1.616, 1.357], 5.996; 0.10 [-1.372, 1.023], 5.457).
- Held-out flag rate (1,000 clean passages): alpha 0.025 -> PD 0.0396, PM 0.0236, either 0.0493 (tolerance 0.018).
- Sanity: fluent PD 0.531 / PM 3.941 vs gibberish half PD -8.194 / PM 12.458.
- Length: mean 102.8 tokens, median 87; 7.3% too short (1.6% of pool passages).
- Detection at alpha 0.025, rule either: diverse 0.144, hijack 0.138, fact 0.125, url 0.064, repeat 0.054,
  stealth 0.040; clean pool 0.019.
- Score cache 39,128 records, 15 MB; calibration 11.1 s.

## Cluster
- Local clean check (Samarth, env=local): questions with anything flagged 0-0.5%; mean clean flagged <= 0.03; 174 ms
  per question (mean, n = 800).
- Arya's local check with poison (env=local): rate-5 detection ~0.5 for fact/hijack/url/repeat; diverse 0.04-0.21.

## Smoke test (A9, Qwen2.5-3B, 20 questions, NO masking)
All checks pass; 143.7 s. Filter ms per question (bm25/dense/splade/rrf): query-phrase 457/98/65/60, cluster
235/118/94/120, perplexity 26/0.9/0.9/60. Poison in pools 191-226 per retriever (all attacks present at once).

## Generation (local step 9, Qwen2.5-1.5B, env=local)
59.8 / 110.3 / 133.1 requests/min at batch 1/4/8; peak VRAM 3.54 GiB at batch 8; greedy determinism 20/20; batch
independence 20/20 but logit shift up to 0.086 vs smallest gap 0.078; 18.4% of passages truncated at 180 tokens;
smoke EM 0.06 / F1 0.26. Ada A9: ~0.2 s per answer (3B) after loading. **A7 below.**

## Sweep plan
384 settings (headline 68, rates 128, sweeps 88, adaptive 80, optional 20); ~60,000 LLM requests upper bound before
caching (in practice ~24,500 new generations in all thanks to the cache, see below). Tests: 435 passing (2026-11-10).

## Generation, Ada (A7, 2026-11-10, job 67804, RTX 2080 Ti, Qwen2.5-3B-Instruct fp16)
Throughput batch 1 / 8 / 16 / 32 = 46.7 / 202.8 / 250.0 / 306.8 requests/min; peak VRAM 5.9 / 6.5 / 7.3 / 8.1 GiB (weights 5.75 GiB);
60,000 requests = 3.3 h at batch 32 (30,000 = 1.6 h); model load 68.8 s; greedy determinism 20/20; batch independence 19/20 (one prompt
differs at the last of 32 generated tokens); smoke EM 0.0 / F1 0.14 on 50 clean questions (sentence answers); 17.0% of passages exceed
the 180-token limit (mean 113.6, median 96); text store 2,686,473 docs. **Decision: batch size 32, the generation cache is the record.**

## Local dry run (env=local, 2026-11-08..10; correctness only, NOT results)
5,000-passage corpus, e5-small, Qwen2.5-1.5B on CPU. Pilot (10 target + 10 clean questions, fact only, headline tier): 36 settings, 720
records, 0 failures, 197 generations + 523 cache hits (72.6% hit rate), 26 min. Records checked by hand: no-poison accuracy 0.6-0.7;
no filter (rate 3) ASR 0.8 with ~3 poison passages in the top 5; oracle ASR 0; the random control matches its detector's number of
dropped passages and leaves ASR at ~0.8. `analyze.py` and `make_plots.py` ran on it; the agreement script was tested on fabricated labels.

## Ada pilot (SWEEP_pilot.json, 2026-11-10, job 67888, gnode056)
36 headline settings, attack fact only, 10 target + 10 clean questions: 0 failed, 720 requests (187 generated, 74% cache
hits), batch 32, 235 s. Checks: oracle ASR 0 on all four retrievers; random drops exactly the detector's number of
passages on every question; oracle top-5 and answers equal the no-poison run for bm25 / dense / splade (20/20), rrf 17/20
(same 5 documents in a different order, see "RRF and the oracle" below). The report's "skipped 12" are the random
settings re-reading their detector's run file, not missing settings.

## Sweep runs (SWEEP_main.json, 2026-11-11)
- Headline tier: job 69674 on gnode082, 68 settings x 400 questions, 27,200 records, 0 failed, 37 min (6,131 generated,
  77% cache hits). gnode056 was full until 13 Nov, so the job ran on gnode082 (same RTX 2080 Ti) after copying the
  dense index, the bm25 / splade manifests and the text store through home (`IRE_STAGE_INDEXES=1` in `ada/pinned_node.sh`).
- Remaining tiers (rates, sweeps, adaptive, optional): job 69752 on gnode082, 316 new settings, 0 failed, 1 h 49 min
  (18,227 generated, 85% cache hits).
- Total: **384/384 settings, 145,600 records, 0 failures.** Main and adaptive settings use 200 target + 200 clean
  questions, except the adaptive / held-out attacks (100 target + 200 clean); url 200 + 200.
- Checks on every record: random count = detector count; oracle leaves 0 poison passages in any top-5; oracle top-5 =
  no-poison top-5 on 400/400 questions for bm25 / dense / splade, rrf 363/400 (fact) and 332/400 (hijack).

### RRF and the oracle (why rrf is not 400/400)
RRF fuses the bm25 and dense lists **with the active poison still in them**; filters (oracle included) remove passages
from the fused pool afterwards. The poison took rank slots in each component list, which shifts the clean passages'
ranks by different amounts in the two lists, so the fused order of clean passages can differ from the no-poison run
(same documents, different order). This is correct: a real defense filters the pool it receives. Consequence: for rrf,
"closeness to the oracle" is measured against the oracle run, not the no-poison run.

## Sweep results: the headline decision table (rate 3, 200 target + 200 clean; decision rule in file 10)
"Helps" = Holm-adjusted McNemar p < 0.05 AND lower ASR AND clean nDCG@5 change >= -0.02. ASR no filter -> detector.

| retriever | attack | query-phrase (tail 0.025) | cluster (tau_c 0.85) | perplexity (alpha 0.025) |
|---|---|---|---|---|
| bm25 | fact | 0.700 -> 0.240, dnDCG -0.111: **no** | 0.700 -> 0.510, -0.005: **helps** | 0.700 -> 0.645, 0.000: **helps** (p_Holm 0.038) |
| bm25 | hijack | 0.895 -> 0.310, -0.111: **no** | 0.895 -> 0.625, -0.005: **helps** | 0.895 -> 0.815, 0.000: **helps** |
| dense | fact | 0.555 -> 0.235, -0.223: **no** | 0.555 -> 0.335, -0.013: **helps** | 0.555 -> 0.485, -0.010: **helps** |
| dense | hijack | 0.655 -> 0.230, -0.223: **no** | 0.655 -> 0.345, -0.013: **helps** | 0.655 -> 0.600, -0.010: **helps** (p_Holm 0.030) |
| splade | fact | 0.560 -> 0.260, -0.201: **no** | 0.560 -> 0.365, -0.013: **helps** | 0.560 -> 0.520, +0.001: no (p_Holm 0.077) |
| splade | hijack | 0.705 -> 0.185, -0.201: **no** | 0.705 -> 0.480, -0.013: **helps** | 0.705 -> 0.625, +0.001: **helps** |
| rrf | fact | 0.630 -> 0.295, -0.188: **no** | 0.630 -> 0.375, -0.015: **helps** | 0.630 -> 0.530, -0.009: **helps** |
| rrf | hijack | 0.840 -> 0.260, -0.188: **no** | 0.840 -> 0.470, -0.015: **helps** | 0.840 -> 0.795, -0.009: no (p_Holm 0.077) |

**Verdicts: cluster helps in 8/8 cells, perplexity in 6/8, query-and-phrase in 0/8.**
- Query-and-phrase gives the largest ASR drops (0.30-0.59) and every p_Holm is ~0, but it fails the quality bound in
  every cell: clean nDCG@5 falls by 0.11-0.22 (95% CI, e.g. dense [-0.280, -0.167]). Reason: it drops a **gold**
  passage on ~31-32% of clean questions (cluster 0.5-3.5%, perplexity 0.5-1.5%). Its query-anchor cut flags passages
  that are very similar to the question, and gold passages are exactly that. Clean accuracy drops too (dense 0.565 -> 0.465).
- Cluster: ASR -0.19 to -0.37, clean nDCG within 0.005-0.015. Caveat for the viva: the rule uses the **point** estimate;
  some cluster CIs reach below -0.02 (dense [-0.034, +0.005]).
- Perplexity: small ASR drops (0.04-0.10). Two cells miss Holm significance (splade fact, rrf hijack: p_Holm 0.077).
- Beats random (unadjusted, exploratory): every detector cell except perplexity on splade fact (p 0.30) and rrf hijack
  (p 0.064). Random at the same removal count leaves ASR at the no-filter level (e.g. dense fact 0.525-0.560).
- Oracle-captured fraction (how much of the oracle's ASR drop a detector gets): query-phrase 0.54-0.74, cluster
  0.27-0.47, perplexity 0.05-0.16.
- Mechanism flag: false in all 24 cells (no detector lowers ASR by any route other than removing poison).

## Detection quality at the anchors (rate 3; summary.csv)
| detector | TPR (passage) | FPR (passage) | clean questions with >= 1 flag | AUC | gold dropped (clean qs) |
|---|---|---|---|---|---|
| query-phrase | fact 0.54-0.59, hijack 0.76-0.78 | 0.029-0.030 | 0.47-0.55 | 0.83-0.94 | 0.315-0.325 |
| cluster | 0.25-0.46 | 0.008-0.021 | 0.08-0.13 | 0.81-0.87 | 0.005-0.035 |
| perplexity | 0.11-0.14 | 0.013-0.024 | 0.33-0.64 | **0.93-0.98** | 0.005-0.015 |
Perplexity has the best ranking of poison vs clean (AUC up to 0.98) but its alpha 0.025 cut is very conservative, so
TPR is only ~0.12: the score separates well, the threshold flags little. That is why alpha 0.1 does much better in the
sweep (below). Backfill (final top-5 using passages beyond rank 20): <= 2% everywhere. Collateral poisoning on clean
questions with no filter: 0-1.5%.

## Rates (ASR at rate 1 / 3 / 5; columns bm25, dense, splade, rrf)
| attack | filter | bm25 | dense | splade | rrf |
|---|---|---|---|---|---|
| fact | none | .34/.70/.98 | .22/.56/.78 | .27/.56/.89 | .23/.63/.90 |
| fact | query-phrase | .12/.24/.39 | .10/.23/.34 | .12/.26/.40 | .10/.29/.41 |
| fact | cluster | .34/.51/.29 | .19/.34/.15 | .26/.36/.17 | .23/.38/.18 |
| fact | perplexity | .29/.65/.90 | .19/.48/.71 | .21/.52/.80 | .19/.53/.80 |
| hijack | none | .54/.90/.94 | .36/.66/.70 | .37/.70/.77 | .46/.84/.93 |
| hijack | query-phrase | .12/.31/.49 | .09/.23/.35 | .08/.18/.30 | .10/.26/.43 |
| hijack | cluster | .53/.62/.35 | .36/.34/.14 | .36/.48/.20 | .45/.47/.20 |
| hijack | perplexity | .45/.81/.90 | .33/.60/.69 | .34/.62/.73 | .39/.80/.89 |
Oracle: 0.00-0.01 everywhere. More poison -> more harm without a defense. **Cluster gets better as the attack gets
bigger**: at rate 1 there is no near-duplicate group to find (ASR unchanged), at rate 5 it beats query-phrase on all four
retrievers and both attacks. Query-phrase and perplexity get worse as the rate grows (more poison survives).

## Threshold sweeps (rate 3; ASR fact / hijack, clean nDCG@5; dense no filter 0.555 / 0.655, 0.492)
- Query-phrase tail 0.01 / 0.025 / 0.05 / 0.1: ASR .34/.29, .23/.23, .18/.17, .10/.12; nDCG .379, .269, .230, .180.
  Even the loosest cut costs 0.11 nDCG. bm25: nDCG .163 -> .043.
- Cluster tau_c 0.70 / 0.75 / 0.80 / 0.85 / 0.90 / 0.95: ASR .33/.32 (0.70-0.80), .34/.34, .42/.45, .46/.54; nDCG
  .479 up to 0.85, .492 above. tau_c 0.70, 0.75 and 0.80 flag **identical** passages: a group needs mean cosine >= tau_c
  AND mean ROUGE-L >= a fixed minimum, and below 0.8 the ROUGE-L gate binds (e5 cosines of related passages are
  almost always > 0.8). Flat by design, not a bug.
- Perplexity alpha 0.01 / 0.025 / 0.05 / 0.1: ASR .53/.63, .48/.60, .35/.52, .20/.26; nDCG .485, .482, .478, .454.
- Cross-fitting (parameter chosen on one fold, reported on the other, pooled): perplexity becomes much stronger
  (e.g. bm25 fact ASR 0.70 -> 0.325, dense hijack 0.655 -> 0.56, splade hijack 0.705 -> 0.39) at nDCG -0.009 to -0.021;
  cluster similar to the anchor; query-phrase unchanged (its nDCG loss stays 0.11-0.22). The headline verdicts use the
  frozen anchors, not the cross-fitted choices.

## Adaptive, held-out and url (rate 3; ASR bm25 / dense / splade / rrf)
| attack | none | query-phrase | cluster | perplexity |
|---|---|---|---|---|
| heldout | .87 .47 .70 .71 | .51 .19 .43 .26 | .70 .37 .46 .52 | .69 .32 .52 .51 |
| stealth | .01 .00 .00 .00 | ~0 | ~0 | ~0 |
| diverse | .36 .22 .28 .30 | .07 .11 .10 .14 | .33 .20 .23 .24 | .30 .23 .24 .28 |
| repeat | .51 .34 .43 .39 | .37 .21 .31 .27 | .30 .18 .24 .22 | .44 .33 .40 .35 |
| url | .86 .62 .64 .77 | .25 .10 .20 .21 | .60 .35 .40 .42 | .80 .58 .60 .71 |
- Stealth is harmless even with no defense: its poison almost never reaches the top 10 (reach 0.000-0.006, A6). A
  stealth "win" is no attack, not a detector success.
- Held-out phrasings (written blind to the phrase list) get past query-phrase more than the phrase-list hijack: bm25
  0.87 -> 0.51 vs hijack 0.90 -> 0.31; TPR 0.45-0.51 vs 0.76-0.78. The phrase rule overfits to its own phrase list.
- Diverse (varied wording) is caught by query-phrase (TPR 0.47-0.58) but not by cluster (TPR 0.06-0.12): no
  near-duplicates.
- Repeat (one passage copied) is the one attack where **cluster beats query-phrase** (TPR 0.38-0.57 vs 0.33-0.38).
- URL: query-phrase TPR 0.85-0.86 (the phrase rule matches the link wording).
- Complementarity: each detector is strongest on a different attack (query-phrase on instructions / URLs, cluster on
  repeats and at high rate, perplexity weak everywhere at its anchor).

## Mechanism (ASR by number of poison passages left in the final top-5, all filters pooled)
fact: 0 -> ~0.00-0.01, 1 -> 0.08-0.30, 2 -> 0.35-0.51, 3+ -> ~0.65. hijack: 0 -> 0.00, 1 -> 0.26-0.51, 2 -> 0.85-0.94,
3+ -> ~0.94-0.95. ASR depends on how many poison passages reach the LLM; **given that count, ASR is the same with or
without a filter**. Detectors help only by removing poison, and one surviving hijack passage already succeeds ~40% of
the time, which is why passage-level TPR overstates the safety gain.
