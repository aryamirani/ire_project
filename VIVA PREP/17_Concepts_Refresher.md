# 17 -- Concepts refresher (the theory behind the code)

## Retrieval
**Inverted index.** term -> posting list of (doc, weight). Query time = merge the posting lists of the query terms.
Used by BM25 (bm25s stores sparse score matrices) and SPLADE (our `inverted.*.npy` CSR transpose).

**TF-IDF -> BM25.** BM25 = probabilistic relevance framework. Term frequency saturates (k1), long documents are
normalised (b), rare terms matter more (IDF). Lucene IDF `ln(1 + (N - df + 0.5)/(df + 0.5))` is always positive.

**Dense bi-encoder.** Two passes of the same encoder: e(q), e(d) in R^d; relevance = dot product (cosine if
normalised). Trained contrastively (InfoNCE: the positive passage must score above in-batch negatives). Documents are
encoded offline, so search = nearest-neighbour in vector space.

**FAISS.** `IndexFlatIP` = brute-force inner product (exact, O(N d) per query). `IndexHNSW` = graph-based
approximate search (sub-linear, recall < 1). Exactness matters for us (masking equivalence).

**SPLADE.** BERT MLM head gives a logit for every vocabulary term at every position; `w_j = max_t log(1 + ReLU(l_tj))`
produces a sparse, non-negative, vocabulary-sized vector with expansion; trained with ranking loss + FLOPS
regularisation for sparsity. Score = sparse dot product.

**Reciprocal rank fusion.** `RRF(d) = sum_r 1/(k + rank_r(d))`; k = 60 dampens the advantage of top ranks; robust,
parameter-light (Cormack, Clarke, Buettcher 2009).

**Recall@k, hit@k, nDCG@k.** Recall@k = fraction of gold docs in the top-k; hit@k = at least one gold in top-k;
nDCG@k = position-discounted gain normalised by the ideal ranking.

## Text similarity and clustering
**Cosine similarity** = dot product of L2-normalised vectors, in [-1, 1].
**k-means** minimises within-cluster squared distance; Lloyd iterations; `n_init=10` runs from 10 seeds and keeps the
best (k-means is sensitive to initialisation); `random_state=13` makes it deterministic.
**LCS / ROUGE-L.** Longest common subsequence (order kept, gaps allowed), dynamic programming O(|a||b|);
ROUGE-L F1 = 2PR/(P + R), P = LCS/|a|, R = LCS/|b|.

## Language-model scores
**NLL and perplexity.** For tokens r_1..r_n: NLL = -(1/n) sum log p(r_i | r_<i); perplexity = exp(NLL). Lower = the
text is more predictable for the model. GPT-2 small: 124M parameters, BPE vocabulary of 50,257.
**Why halves?** A spliced text (fluent prefix + unrelated payload) can have a different NLL in each half; PD captures
that difference. **Repetition** makes later tokens nearly certain -> very low NLL.

## Statistics
**Wilson score interval** for a proportion: inverts the score test; good coverage for small n or p near 0/1.
**McNemar's test**: paired binary data; discordant counts b (yes->no), c (no->yes); H0: b and c equally likely;
exact p from Binomial(b + c, 0.5).
**Bootstrap**: resample units with replacement many times, recompute the statistic, take percentiles for an interval.
The unit must be the independent unit (questions).
**Paired bootstrap**: resample per-question differences (same questions in both conditions).
**ROC-AUC**: area under TPR vs FPR over all thresholds = probability that a random positive scores above a random
negative (Mann-Whitney U / (n_pos n_neg)).
**Multiple comparisons**: with 24 tests at 0.05, ~1.2 false positives are expected by chance. Bonferroni: p * m.
Holm: step-down, p_(i) * (m - i + 1), monotone; same FWER guarantee, more power. Benjamini-Hochberg: controls the false
discovery rate instead (weaker).
**Cohen's kappa**: (p_observed - p_chance)/(1 - p_chance); 0.8+ = strong agreement.
**Cross-fitting**: sample splitting to avoid selection bias (choose on one half, evaluate on the other, swap).
**Pre-registration**: fixing hypotheses and analysis before data, against the "garden of forking paths".

## LLM generation
**Greedy decoding**: argmax token at each step (temperature 0). **Sampling parameters** (temperature, top-k, top-p)
are ignored when do_sample=False, but **repetition_penalty** still rescales logits -> must be set to 1.0 for pure
greedy. **Chat template**: the model-specific wrapping of system/user/assistant turns (`<|im_start|>user ...`).
**Left padding** for batched decoder-only generation. **fp16 non-determinism**: different batch shapes choose
different kernels/reduction orders -> tiny numeric differences that can flip near-ties.

## Security background
**Indirect prompt injection** (Greshake et al. 2023): instructions hidden in data the model reads (web pages, emails,
retrieved passages). **PoisonedRAG** (Zou et al. 2024): passages = retrieval condition (the question) + generation
condition (a statement of the false answer); high ASR with few passages. **TrustRAG**: k-means clustering of retrieved
passages to remove near-duplicate poison, then LLM self-knowledge. **RAGuard**: combines similarity, perplexity and
other signals; reports per-signal ablations. **PoisonCraft**: injections phrased as helpful guidance to evade phrase
filters. **Nasr et al.**: adaptive attacks bypass 12 defenses. **RobustRAG**: certified robustness via isolate-then-
aggregate (not run here).
