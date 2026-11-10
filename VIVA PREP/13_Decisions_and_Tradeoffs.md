# 13 -- Decisions and trade-offs

Format: decision -> alternative(s) -> why we chose it -> what it costs. Know the "costs" column: examiners like
candidates who know the weaknesses of their own choices.

## A. Research design
| # | Decision | Alternative | Why | Cost / risk |
|---|---|---|---|---|
| 1 | Three training-free post-retrieval detectors | provenance/trust weighting (earlier proposal); trained classifiers; LLM-based checks (TrustRAG stage 2) | thresholds calibrated from clean data, no hand-set trust tiers (NQ has no source metadata); cheap; comparable on equal terms | weaker than trained or LLM-based detectors; no robustness claim |
| 2 | Every detector paired with a named adaptive attack | only standard attacks | defenses must be tested against adaptive attackers (Nasr et al.) | results will look worse -- by design |
| 3 | Random + oracle controls | report detector vs no filter only | separates "the detector chose well" from "removing passages helps"; oracle = ceiling | 2 extra settings per cell |
| 4 | Pre-registered decision rule, frozen before sweeps | decide after seeing results | prevents choosing the flattering comparison | cannot change thresholds later even if a better choice appears |
| 5 | Cross-fitting over folds A/B | tune on all questions | no setting is chosen on the questions it is reported on | half the data per choice (noisier) |
| 6 | 200 target + 200 clean questions | more questions, fewer settings | budget: ~60,000 LLM requests upper bound; Wilson half-width ~7 points at n = 200 | small effects (< ~10 points) may be undetectable |
| 7 | Rates 1/3/5 of 5 numbered passages, nested | random subsets per rate | nested settings; maps to 20/60/100% of a 5-passage context (TrustRAG) | rate 1 is a single fixed template per question |
| 8 | String-match ASR + human calibration of 150 answers | LLM-as-judge | deterministic, cheap, reproducible; target strings absent from clean context | quotes-but-rejects answers count as success; human labels measure this |
| 9 | Neutral prompt, no injection warning | a defensive prompt | measure the detectors, not prompt engineering; normal usage | higher baseline ASR (intended) |
| 10 | Full 2.68M corpus | the plan's 500k fallback subset | realistic competition for the poison; it fitted on one node's /scratch | 46-54 min index builds, 19 GB, pinned node |

## B. Retrieval (Samarth)
| # | Decision | Alternative | Why | Cost |
|---|---|---|---|---|
| 11 | One index containing all poison, mask per setting | an index per (attack, rate) | ~20x less indexing; exact for dense/SPLADE | BM25 approximate (IDF): max rank diff 36, top-5 96% unchanged |
| 12 | bm25s (Python) | pyserini (Lucene, Java) | no JDK on Windows/Ada, pip install, memory-mapped, fast enough | slightly different tokenisation than Anserini (R@100 0.744 vs BEIR 0.76) |
| 13 | k1 = 0.9, b = 0.4 | 1.2 / 0.75 (textbook) | Anserini defaults used for BEIR baselines -> comparable | none significant |
| 14 | Exact BM25 scoring of every document with doc_id ties | bm25s `retrieve` (partial sort) | deterministic, cache == fresh search == score_docs | slower search (full score vector per query; 13 ms) |
| 15 | Drop the empty token id 0 from queries | trust the library | stopword-only queries matched every stopword-only document | -- |
| 16 | e5-base-v2 (768 d) | BGE, Contriever, larger E5 | plan allowed E5 or BGE; strong on BEIR, small enough for one GPU; prefixes enforced | not state of the art; e5-small locally (different cutoffs) |
| 17 | FAISS Flat (exact) on Ada | HNSW | exact masking equivalence and exact ties | 8.2 GB RAM, full scan per query (~21 ms) |
| 18 | fp16 vector storage, fp32 normalisation | fp32 storage | halves disk/RAM (4.1 GB); cosine to fresh vectors 0.99997 | tiny precision loss |
| 19 | Queries encoded one at a time; one query per FAISS call | batch queries | a query's vector/results must not depend on its batch (fp16, BLAS rounding) | slower query encoding (irrelevant: 400 questions) |
| 20 | SPLADE without pruning | prune low weights (common) | exact scores -> exact masking | 6 GB index, 392.9M non-zeros |
| 21 | `log1p(relu(max logit))` | `max(log1p(relu(logit)))` | identical (monotone), avoids a second huge tensor | -- |
| 22 | RRF k = 60 over BM25 + dense | weighted score fusion; SPLADE in the fusion | rank-based, no tuning, standard; plan's hybrid | ignores score magnitudes |
| 23 | Mask before fusing | fuse then mask | exact; fusing first keeps poisoned ranks and loses documents at depth | must fuse at pool time (cheap) |
| 24 | Top-150 cache with fingerprinted sidecars | recompute or trust files | experiments never retrieve; stale caches refused automatically | must rebuild after any index/question change (by design) |
| 25 | Save the clean masking reference before ingestion | rebuild a clean index for the check | the clean index no longer exists after ingestion; rebuilding costs ~1 h | small extra file |
| 26 | Append poison to dense/SPLADE, rebuild BM25 | rebuild everything | encoding 5,005 passages instead of 2.68M | `chunk_bounds` needed to keep fp16 vectors identical |

## C. Detectors
| # | Decision | Alternative | Why | Cost |
|---|---|---|---|---|
| 27 | Percentile cutoffs from clean pools (query-phrase), clean corpus passages (perplexity) | hand-set thresholds; z-scores within the pool | false-positive rate controlled by construction; z-score max is bounded (~7 for n = 50) so a tail is always flagged | calibration noise (e.g. PD 3.96% vs 2.5%) |
| 28 | Phrase list frozen before templates; held-out written blind | write list after attacks | avoids leakage that would make the phrase rule look perfect | the list is generic, misses novel wording (3/10 held-out caught) |
| 29 | Cluster: top-20, k = 2, cosine AND ROUGE-L >= 0.25 | cosine only; larger k; whole pool | two independent signals cut false positives; TrustRAG design | cannot detect a single poison passage (rate 1) |
| 30 | Omit TrustRAG's LLM stage | full TrustRAG | training-free, cheap, comparable | weaker detector |
| 31 | Perplexity on halves (PD, PM) with GPT-2 small | whole-text perplexity; larger LM | junction effect (RAGuard); GPT-2 small is cheap (CPU-able) | NQ halves ~45 tokens -> noisy; weak detection (5-14%) |
| 32 | Score cache for every passage that can enter a pool | score on the fly | sweeps do no GPT-2 work (~1 ms per question) | 39,128 records scored up front |
| 33 | Rule "either" for perplexity and query-phrase | pd-only / pm-only / similarity-only | catches either failure mode | higher false-positive rate (alpha per rule adds up) |

## D. Generation
| # | Decision | Alternative | Why | Cost |
|---|---|---|---|---|
| 34 | Qwen2.5-3B-Instruct, greedy, max 32 tokens | larger model, sampling | plan: one fixed 3-4B model; fits one GPU; determinism | results specific to this model |
| 35 | repetition_penalty forced to 1.0 | Qwen's default 1.1 | plan says greedy; 1.1 changes answers even with do_sample=False | -- |
| 36 | Chat template with the model's default system message | raw prompt; custom system message | instruct models expect the template; default is neutral | the default system text is part of the setup (recorded) |
| 37 | Passages truncated to 180 tokens | no truncation | bounded prompt length and memory; uniform batches | 18% of passages cut (locally) |
| 38 | Cache keyed by (qid, ordered top-5) within a namespace | prompt-text key | the plan's key; namespace covers everything else | -- |
| 39 | Batched generation, cache as the record | batch size 1 | 2.2x faster locally; one sweep internally consistent | a rare answer could differ from a batch-1 run (logit shifts up to 0.086 vs 0.078 smallest gap); decided 2026-11-10 from A7: batch 32 (306.8 req/min vs 46.7 at batch 1), batch independence 19/20 |

## E. Engineering
| # | Decision | Why |
|---|---|---|
| 40 | Two profiles (local/ada) in one code base | Samarth had no Ada access; develop and test locally, run at scale on Ada with identical code |
| 41 | Every Ada job = one command, resumable, JSON report, checked by rules | non-expert teammates run the jobs; acceptance is objective |
| 42 | Strict schemas | interface errors surface at the boundary with a line number |
| 43 | SQLite for text store, generation cache, perplexity cache | random access by key on disk; transactions; no server |
| 44 | Frozen files never edited; LF line endings forced | hashes recorded in reports must stay valid on every OS |
| 45 | Off-Ada calibrations never touch the tracked threshold files | a local run must not overwrite the Ada calibration used for results (bug found and fixed) |
