# 16 -- Viva question bank (with model answers)

Cover the answer, say yours out loud, then compare. Answers are short on purpose; the referenced file has the detail.
"(Q)" = the examiner may push further; a follow-up is given.

## A. Big picture (files 01-02)
1. **What is your project in one sentence?** We measure how much poisoned text in a RAG corpus changes the answer
   under four retrievers, and whether three training-free post-retrieval detectors reduce it beyond random removal.
2. **What is RAG poisoning?** Inserting passages into the retrieval corpus so that, when retrieved, they change the
   LLM's answer: fact distortion (false answer) or instruction hijack (indirect prompt injection).
3. **What is your threat model?** Insert-only attacker who knows the target question, injects 1/3/5 passages, cannot
   touch the retriever, model or prompt; adaptive variants also know the detector.
4. **What are your research questions?** RQ1 attack effect vs rate per retriever; RQ2 detector + defense views;
   RQ3 complementarity / which adaptive attack beats which detector; RQ4 vs random and oracle; RQ5 lexical vs dense.
5. **What is novel?** Side-by-side evaluation of three detector families on one corpus/LLM/attack suite with four
   retriever types, named adaptive attacks, random/oracle controls, both views, pre-registered decisions.
6. **Do you claim robustness?** No. Claims are limited to this corpus, LLM and these attacks; adaptive attacks are
   expected to break detectors (Nasr et al.).
7. **Why did you drop the trust-tier filter?** Needed hand-set thresholds and synthetic source trust that NQ does not
   have; the new detectors calibrate from clean data.
8. **Walk me through the pipeline.** Index (all poison) -> cache top-150 -> mask inactive poison -> top-50 pool ->
   detector flags -> first 5 survivors -> Qwen greedy -> metrics. (File 02, section 5.)
9. **Why 150 / 50 / 5?** 150: retrieval cached once with headroom for masking; 50: room for filters to drop and
   backfill; 5: the LLM context (maps rates to 20/60/100%).
10. **What is backfill and why report it?** Whether the final 5 include passages beyond rank 20; a defense that works
    only by pulling in lower-ranked passages trades quality for safety.
11. **What results do you have?** Retrieval quality, poison reach, exact masking, detector calibration and sanity
    checks, smoke test. The ASR comparisons need the sweeps, which have not run yet. (File 14.)

## B. Data (file 03)
12. **Which dataset and why?** BEIR NQ (2.68M Wikipedia passages, gold relevance) + nq_open (short answers);
    standard, realistic, and answers allow string-match ASR/EM.
13. **How did you link BEIR and nq_open?** Normalised question text (NFKC, lowercase, no punctuation, single
    spaces); 2,255 of 3,452 matched (65.3%).
14. **How were the 400 questions chosen?** Eligibility (short non-yes/no answers not contained in the question,
    >= 3-word questions, no duplicates), answer-type stratification, seeded round-robin into 4 cells of 100.
15. **Why exclude answers contained in the question?** String metrics would fire by echoing the question.
16. **Why clean questions?** Collateral damage and false positives; the "helps" rule uses clean nDCG@5.
17. **Why two folds?** Cross-fitting: choose thresholds on one, report on the other.
18. **Why not the 500k subset?** The full corpus fitted; it is the realistic setting.

## C. Retrievers (file 04)
19. **Write the BM25 formula.** sum_t IDF(t) * tf / (tf + k1 (1 - b + b |d|/avgdl)), IDF = ln(1 + (N - df + 0.5)/(df + 0.5)).
20. **What do k1 and b do? Your values?** k1 = tf saturation, b = length normalisation; 0.9 / 0.4 (Anserini BEIR
    defaults).
21. **Why bm25s?** No Java, pip-installable, memory-mapped, fast enough; pyserini needs a JDK.
22. **Your BM25 Recall@100 is 0.744 but BEIR reports 0.76 -- why?** Different tokeniser/stemmer/stopwords and a single
    title+text field vs Anserini's analyser; close enough for our purpose.
23. **What was the empty-token bug?** bm25s maps empty texts to a "" token; stopword-only queries matched stopword-only
    docs. We drop token id 0.
24. **How are ties handled and why does it matter?** Score desc, then doc_id, including at the k-th place; otherwise
    the cache, fresh searches and score_docs can disagree (duplicate passages exist).
25. **Why do BM25 rows sometimes have fewer than 150 entries?** Only documents sharing a query term (score > 0).
26. **What is E5 and why the prefixes?** A contrastively trained bi-encoder; trained with "query: "/"passage: "
    prefixes, which tell it which side it encodes; without them quality drops.
27. **Pooling and normalisation?** Mean over non-padding tokens; L2-normalised in fp32 -> inner product = cosine.
28. **Why FAISS Flat instead of HNSW?** Exactness: masking equals a separate index only with exact search; also exact
    ties. Costs 8.2 GB RAM and a full scan (~21 ms/query).
29. **Why encode queries one at a time?** fp16 vectors depend on batch composition; a query's results must depend on
    its text only (cache vs score_docs consistency).
30. **Why one query per FAISS call?** BLAS (>= 20 queries) and loop paths round differently; reordered near-ties.
31. **What is SPLADE?** Learned sparse retrieval: MLM-head logits -> w_j = max_t log(1 + ReLU(logit_tj)) over the
    vocabulary; dot product; inverted index; expansion terms fix vocabulary mismatch.
32. **Why `log1p(relu(max))`?** Equals `max(log1p(relu))` because both functions are monotone; saves a huge tensor.
33. **Why no pruning?** Exact scores -> exact masking equivalence.
34. **How is SPLADE search done?** Sum posting lists of the query's terms from the inverted CSR; score > 0; ties by
    doc_id.
35. **RRF formula and k?** sum 1/(k + rank), k = 60, BM25 + dense to depth 150.
36. **Why RRF not score fusion?** Scores are on different scales; rank fusion needs no normalisation or tuning.
37. **Why math.fsum?** Exact rounding: the fused score does not depend on list order.
38. **Why mask before fusing?** Ranks shift when a document is removed; fusing first keeps ranks computed with poison
    present and loses documents beyond depth.
39. **Which retriever is best and why?** Dense (R@100 0.941) > SPLADE (0.927) > BM25 (0.744) on NQ: paraphrased
    natural-language questions favour semantic matching.
40. **(Q) Which retriever is most vulnerable?** It depends on the attack: BM25 lets fact poison into the top-10 most
    (0.915), SPLADE repeat (0.698), dense url (0.296); stealth barely reaches any. ASR per retriever comes from the
    sweeps.

## D. Cache, ingestion, masking (file 05)
41. **Why one index with all poison?** ~20x cheaper than an index per setting; exact for dense/SPLADE.
42. **Prove masking is exact for dense.** score(q, d) depends only on q and d; removing other documents changes no
    score; with exact search the remaining ranking is identical. Verified 50/50 on Ada.
43. **Why is BM25 only approximate?** IDF and avgdl depend on the collection, which includes poison.
44. **How big is the BM25 error?** Max rank difference 36, mean 0.43, top-5 unchanged 96%, Jaccard@50 0.9945.
45. **What is in the cache sidecar?** Retriever, cache version, questions sha256, depth, index fingerprint (corpus and
    shard hashes, settings, model revision); RRF: component keys + k/depth.
46. **What happens with a stale cache?** `StaleCacheError` -- refused.
47. **How is poison added to each index?** BM25 rebuilt (global statistics); dense and SPLADE append only the poison.
48. **What is chunk_bounds for?** Keeps clean and poison in separate fp16 encoding batches so append == full rebuild.
49. **What is the clean masking reference?** The clean index's top-150 saved before the first ingestion, as the
    baseline for the equivalence check.
50. **What is collateral poisoning?** Poison written for one target question reaching another question's context;
    measured on clean questions (other targets' poison stays active).
51. **How did you test ingestion without real poison?** A 10-passage synthetic fixture on temporary index copies.

## E. Attacks (file 06)
52. **Describe the seven attacks.** fact, hijack, url (main); diverse, repeat, stealth (adaptive); heldout (blind
    phrasings). See the table in file 06.
53. **How are false target answers chosen?** Same answer type, no shared content word with gold, not in the question
    or gold passages, <= 5 words; year shifts or another question's same-type answer; seeded per qid.
54. **Weakness of that?** The coarse "other" type (63/200) can give odd targets ("about 1500 BC" for a percentage).
55. **Why ZOLTRAX?** A codeword absent from the corpus: success is unambiguous.
56. **What makes stealth stealthy, and what does it cost?** No question prefix, polite guidance wording, similarity
    capped at the clean pool median; it is then rarely retrieved (top-10 reach ~0%).
57. **Why does repeat target perplexity?** Repetition makes text highly predictable: perplexity falls below clean text.
58. **Why does diverse target the cluster filter?** Five differently structured passages are not near-duplicates.
59. **Why freeze the phrase list before the templates?** Otherwise the list leaks into the attacks and catches them
    trivially.
60. **What is the held-out set and who wrote it?** 10 hijack phrasings + 5 fact passages written by someone who had
    not seen the list (Samarth's side); 505 passages; the list caught 3/10.
61. **What did the held-out test show?** Query-phrase TPR drops from ~0.78 (template hijack) to ~0.46 (held-out) at
    rate 5: the phrase rule partly overfits its author's wording.

## F. Detectors (file 07)
62. **Query-and-phrase in one line?** Flag if cos(q, p) is above the clean-pool (1 - tail) percentile from the other
    fold, or the text matches the frozen injection regexes.
63. **Why the opposite fold?** No cutoff is chosen on the questions it is reported on.
64. **Why not a z-score within the pool?** In 50 values the max z is bounded (~7): a tail is always flagged even
    without poison. Percentiles from clean data control the FPR.
65. **Cluster filter steps?** Top-20 vectors -> k-means k=2 (seeded, n_init 10) -> per cluster >= 2: mean pairwise
    cosine and ROUGE-L -> flag whole cluster if cos >= 0.85 and ROUGE-L >= 0.25.
66. **Why ROUGE-L in addition to cosine?** All top-20 are on-topic (high cosine); near-duplicate poison also shares word
    sequences. Two signals reduce false positives.
67. **Explain ROUGE-L.** F1 of LCS-based precision (LCS/|a|) and recall (LCS/|b|) on word tokens.
68. **Can the cluster filter detect rate 1?** No: a single passage cannot form a cluster.
69. **Why omit TrustRAG's second stage?** It uses the LLM; we keep detectors training-free and comparable.
70. **Perplexity filter formulas?** f = mean token NLL under GPT-2; PD = f(pre) - f(post) two-tailed; PM = max one-tailed;
    thresholds = percentiles of 1,000 clean passages; flag if either fires.
71. **Why split in halves?** Poison concatenates a question prefix with a payload; fluency may change at the junction.
72. **Why is it weak here?** Short NQ halves (~45 tokens) give noisy estimates; template poison is fluent; repetition
    lowers perplexity. Measured flag rates 5-14% on poison vs 1.9% clean.
73. **What is too_short?** < 20 GPT-2 tokens: never flagged (7.3% of passages).
74. **Why a score cache?** Every pool passage scored once (39,128); sweeps do no GPT-2 work.
75. **The held-out flag rate is 3.96% at alpha 2.5% -- is calibration broken?** Within the computed tolerance (+/-1.8
    points, both samples' variance); the PD band uses only ~12 passages per tail. Report the measured FPR.
76. **Why random and oracle controls?** Random: does the detector's *choice* matter, beyond removing passages? Oracle:
    the ceiling.
77. **How is random size-matched?** Per question, it drops as many passages as the detector flagged, seeded by setting
    and qid.

## G. Generation (file 08)
78. **Which LLM and why?** Qwen2.5-3B-Instruct: the plan's fixed 3-4B instruct model; fits one 11 GB GPU.
79. **What is the prompt?** The plan's neutral prompt, word for word, as the user message of the chat template.
80. **Why no injection warning?** We evaluate the detectors, not prompt defenses; normal usage.
81. **What system message?** None passed; Qwen's template adds its default ("You are Qwen ... helpful assistant").
82. **Why max 32 new tokens and 180-token passages?** Short answers suffice; bounded prompt length.
83. **What was the repetition-penalty issue?** Qwen's config applies 1.1 even in greedy mode; forced to 1.0.
84. **Explain the generation cache key.** sha1(qid | id1 | ... | id5); order matters; within a namespace that hashes
    model, revision, template, format, truncation, decoding.
85. **Is batched generation deterministic?** 20/20 identical locally, but logits shift up to 0.086 vs a 0.078 smallest
    gap, so a flip is possible; the cache is the record; batch size decided with A7.
86. **Why left padding?** Decoder-only generation continues from the last position; right padding would put pads
    between the prompt and the answer.
87. **Why is EM so low (0.06)?** The model answers in sentences; we use `acc` (gold contained).

## H. Metrics and statistics (file 09)
88. **How is ASR computed?** Codeword/URL: case-insensitive substring; false answers: SQuAD-normalised token match; only
    target questions of attacked settings.
89. **Weakness of string-match ASR?** An answer quoting and rejecting the false answer counts; humans check 150.
90. **nDCG@5 definition?** sum rel_i/log2(i+1) over the top 5 / ideal DCG; binary relevance = gold passage.
91. **Why McNemar?** Paired binary outcomes on the same questions; only discordant pairs inform.
92. **Exact vs chi-square McNemar?** Exact binomial on discordant counts; valid for small counts.
93. **Why Holm?** Family-wise error control, more powerful than Bonferroni, no independence assumption.
94. **Family size?** 24 = 3 detectors x 4 retrievers x 2 attacks at the anchors, rate 3.
95. **Why Wilson intervals?** Correct coverage near 0/1 and small n; Wald can exceed [0, 1].
96. **What does the bootstrap resample?** Questions, never passages (passages within a question are dependent).
97. **AUC computation?** Mann-Whitney rank formula with average ranks for ties; pooled over questions; bootstrap over
    questions.
98. **Cohen's kappa?** (p_o - p_e)/(1 - p_e): agreement beyond chance; target >= 0.80.

## I. Sweep, analysis, decision rule (file 10)
99. **How many settings?** 384 in 5 tiers (headline 68, rates 128, sweeps 88, adaptive 80, optional 20).
100. **What is in a run record?** Ids/setting fields, pool, scores, flags, top-5, poison/gold counts, backfill, ms,
     nDCG/Recall, answer, asr, em, f1, acc.
101. **How is the sweep resumable?** One file per setting, written to .tmp and atomically renamed; existing files
     skipped; answers from the generation cache.
102. **What does "helps" mean?** Holm-adjusted McNemar p < 0.05 for lower ASR AND clean nDCG@5 loss <= 0.02; "beats
     random" reported separately.
103. **Why pre-register?** Avoid choosing the flattering analysis after seeing results.
104. **What is cross-fitting?** Choose a parameter on fold A (max ASR reduction within the clean-loss bound), report on
     B, swap, pool.
105. **What is the mechanism check?** ASR vs number of poison passages in the top-5; flags cells where poison leaves the
     context without ASR falling or vice versa.
106. **What if the detectors don't help?** We say so per cell with intervals; negative results are reported.
107. **Human calibration design?** 150 answers, 75 dev/75 test, 50 per pair, blind; 100 passages each; kappa >= 0.8,
     judge accuracy >= 90%; revise rules once on dev.

## J. Engineering and reproducibility (file 11)
108. **How do you make results reproducible?** Seed 13 with string-keyed RNGs, doc_id tie-breaks, pinned versions and
     model revisions, content fingerprints, frozen inputs, exact search, atomic writes, 433 tests.
109. **Why string-keyed RNGs?** A draw depends only on its key (e.g. seed:target:qid), not on processing order.
110. **How did you work without Ada access?** Local profile with small stand-ins; Ada jobs as one-command scripts with
     JSON reports, run by teammates and checked by rules.
111. **What is DONE vs DONE-LOCAL?** DONE requires an Ada report passing the checker.
112. **How are large files kept out of git?** IRE_DATA_DIR outside the repo (or git-ignored ire_data_dir); the config
     refuses other in-repo paths.
113. **Why SQLite for the caches?** Keyed random access on disk, transactions, no server, works on Windows and Linux.
114. **Biggest engineering difficulty?** Exactness and determinism at scale on GPUs: fp16 batch effects, BLAS rounding,
     ties, append vs rebuild -- each found by an equivalence test and fixed.

## K. Your own work (expect "what exactly did YOU do?")
115. **What did you implement?** Steps 0-9: skeleton, config, data, BM25/dense/SPLADE/RRF, top-150 cache, poison
     ingestion + masking check, generation harness + cache, report checker; held-out phrasings; audit and fixes of
     steps 10-13 (perplexity memory/threshold/namespace fixes, threshold-path fix for both filters, LF fix).
116. **Hardest bug you fixed?** e.g. fp16 append vs rebuild (chunk_bounds) or GPT-2 OOM (sliced reshape copied the
     logits; row-wise loss on views; verified identical to 1e-6).
117. **What would you change with more time?** Exact BM25 masking (index per attack for BM25 only, it is cheap); a
     title for poison passages; a finer answer-type classifier for targets; LLM-judge cross-check; vLLM for speed.
118. **(Q) If BM25 masking is approximate, are BM25 results invalid?** No: the difference is measured and small (top-5
     unchanged 96%); it is reported as a limitation, and a BM25 index per attack would be the fix (334 s each).

## L. Curveballs
119. **Could the attacker just put the poison in the title?** Yes; our poison has no title (visible as "[i] text").
     A limitation of the format, recorded.
120. **Would a larger LLM resist better?** Possibly (more instruction-following, but also more susceptible to
     instructions); out of scope -- claims are for this model.
121. **Why GPT-2 small for perplexity, not the generator?** Cheap, standard in the cited work, independent of the
     generator; the plan specifies it.
122. **Why is the 'heldout' sweep population 105 questions?** It counts every question with held-out poison (100
     hijack + 5 fact). Proposed fix: score the 100 hijack questions only (plan defines the test as hijack phrasings);
     pending team agreement because the decision rule is frozen.
123. **What does 'training-free' buy you?** No poisoned training data, no leakage from attack design into detector
     training, cheap to run on every query.
124. **Is ASR on clean questions defined?** No (null): nothing to succeed at; clean questions measure collateral and
     quality.
125. **What is the difference between TPR and ASR reduction?** TPR is about passages flagged; ASR about answers. A
     detector can flag poison yet ASR barely moves if remaining poison suffices.
