# 14 -- All measured numbers (as of 2026-11-10)

**There are no final experiment results yet** (the Ada pilot and sweeps have not run; A7 and a local dry run have). If asked "what did you find?", say what
is measured below (retrieval quality, poison reach, masking exactness, detector sanity checks, smoke test) and that
the ASR comparisons come from the sweeps. Source in brackets.

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
caching. Tests: 435 passing (2026-11-10).

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
