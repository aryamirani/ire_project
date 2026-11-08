# 07 -- The three detectors and the two controls

Common contract (section 2): `apply_filter(pool, qid, question, ctx) -> FilterResult(flagged: set, scores: dict)`.
`pool` = the masked top-50 ranked doc_ids; `ctx = FilterContext(texts=TextStore, vectors=VectorAPI, config,
thresholds)`. The runner sends the **first 5 unflagged** passages, in rank order, to the LLM. Every detector is
**training-free** (no learning on poisoned data) and returns a continuous score per passage, used for ROC-AUC.

---------------------------------------------------------------------------------------------------------------
## 1. Query-and-phrase filter (Arya; `src/ire/defenses/query_phrase.py`; plan 4.1)

**Why:** poison written to be retrieved must look like the question; poison written to hijack must contain an
instruction. The first is measurable with the embedding model, the second with a short phrase list.
**Evidence:** RAGuard -- similarity-only detection 0.695 accuracy on NQ vs 0.962 for the full method: useful but
incomplete. PoisonCraft rewords injections as helpful guidance to evade phrase rules -> our stealth variant.

**Signal 1 -- similarity.** `s(q, p) = cos(e(q), e(p))` with the dense encoder (`get_query_vector` . `get_doc_vector`,
both L2-normalised). Flag if `s > cutoff`, the passage is *unusually* close to the question.
- **Calibration** (`scripts/calibrate_query_phrase.py`, job P2): for each retriever and fold, collect s over the
  **clean** questions' top-50 pools with **all poison masked**; cutoff = the `(1 - tail)` percentile; tails 1, 2.5
  (anchor), 5, 10%.
- **Cross-fitting:** a question in fold A is filtered with the cutoff calibrated on fold B's clean questions, and
  vice versa (`thresholds_for(table, retriever, report_fold)`), so no threshold is tuned on the reporting fold.
- Ada cutoffs at tail 2.5%: about 0.870-0.878 (e.g. dense A 0.8783, B 0.8753). Cross-fold flag rate on clean pools
  ~= the tail (sanity).
- Since 2026-11-08: the loader refuses cutoffs calibrated with another dense model (cosines are model-specific).

**Signal 2 -- phrase rule.** Flag if the normalised text matches any of the 24 frozen regexes (file 06 section 4).
FPR 0.11% on clean pool passages.

**Rule:** `query_phrase.rule: either` (flag if either fires); `similarity` / `phrase` for ablations.
**Ablation only:** distance from the pool centroid. **Not** a within-pool z-score rule: in a pool of 50 values the
maximum z-score is bounded (about 7 = (n-1)/sqrt(n)), so a z-score tail always flags *something* whether or not poison
exists -- a percentile from clean data does not have that problem.
**Score for AUC:** s(q, p).

**Detection sanity on Ada (rate 5, anchor tail, TPR on poison / FPR on clean passages of target pools):**
| attack | bm25 | dense | splade | rrf |
|---|---|---|---|---|
| url | 0.85 | 0.84 | 0.84 | 0.84 |
| hijack | 0.79 | 0.77 | 0.78 | 0.77 |
| fact | 0.62 | 0.57 | 0.58 | 0.56 |
| diverse | 0.61 | 0.51 | 0.54 | 0.50 |
| heldout | 0.47 | 0.49 | 0.44 | 0.44 |
| repeat | 0.39 | 0.35 | 0.34 | 0.33 |
| stealth | 0.01 | 0.01 | 0.00 | 0.00 |
FPR on clean passages ~0.03 throughout. Stealth defeats it completely (as designed); held-out wording costs ~30 points.

---------------------------------------------------------------------------------------------------------------
## 2. Cluster filter (`src/ire/defenses/cluster.py`; plan 4.2; step 10, written by Arya, Samarth's lane)

**Why:** an attacker who wants a false answer to win must inject several passages saying it; they tend to be
near-duplicates, while genuine passages on a topic vary. Stage 1 of TrustRAG.
**Evidence:** TrustRAG clustering on NQ with BGE: detection F1 ~97-98 at 60% and 100% poison, but ~3 at 20% (one
poison passage cannot form a cluster); clean retention 66-93%; diverse poison left ~22% ASR vs ~1% for
question-prefixed poison. RAGuard reports a high false-positive rate for TrustRAG.

**Specification (exact code behaviour):**
1. Take the top **20** pool passages and their **stored** dense vectors (`get_doc_vector`).
2. **k-means with k = 2** (`sklearn KMeans(n_clusters=2, n_init=10, random_state=13)`); if fewer than 2 distinct
   points, one cluster.
3. For each cluster with **>= 2 members**: mean pairwise cosine of the member vectors (upper triangle of `V V^T`) and
   mean pairwise **ROUGE-L F1** of the member texts (own LCS implementation on lowercase word tokens).
4. Flag the **whole cluster** only if `cosine >= tau_c` **and** `ROUGE-L >= 0.25`. Anchor tau_c = 0.85; sweep 0.70,
   0.75, 0.80, 0.85, 0.90, 0.95.
5. Survivors: first 5 unflagged of the whole 50-pool; backfill from ranks 21-50 is reported.
Score for AUC: the mean cosine of the passage's cluster (0 for singletons and ranks > 20).

**Design choices:**
- **Why k = 2?** The question is binary -- "is there a tight group (poison) separate from the rest?" -- and the plan
  follows TrustRAG. Larger k would split genuine passages into small tight groups and raise false positives.
- **Why the ROUGE-L gate?** Embedding cosine alone is high for any passages on the same topic (all top-20 are about
  the question). ROUGE-L >= 0.25 requires *lexical* overlap too (shared word sequences), which near-duplicate poison
  has and topically related genuine passages usually lack. Two independent signals -> fewer false positives.
- **Why only the top-20?** Poison that matters ranks high; clustering 50 would cost more and dilute.
- **TrustRAG's second stage (LLM self-knowledge) is deliberately omitted**: keeps the filter training-free, cheap and
  comparable with the other two.
- **Known blind spot:** at rate 1 there is a single poison passage, which can never form a >= 2 cluster (TrustRAG's
  ~3 F1 at 20%).
Local check (Samarth, env=local, clean questions, no poison): fraction of questions with anything flagged 0-0.5%,
mean clean passages flagged <= 0.03, ~174 ms per question. Arya's local check with poison: rate-5 detection ~0.5 for
fact/hijack/url/repeat, diverse 0.04-0.21 (diverse evades it, as designed). Tests: 5 near-duplicates + 15 diverse
-> the 5 flagged; 20 diverse -> nothing; a single-member cluster never flagged.

---------------------------------------------------------------------------------------------------------------
## 3. Perplexity filter (`src/ire/defenses/perplexity.py`; plan 4.3; step 11, written by Arya, fixed by Samarth)

**Why:** poison is machine-written and typically concatenates a question-matching prefix with a payload; fluency may
change at the junction. Whole-text perplexity is weak, a chunk-wise version looks at the junction.
**Evidence:** RAGuard: perplexity-only variant 0.893 detection accuracy on NQ (FPR 4.3%) vs 0.593 for whole-text
perplexity. **Counter-evidence:** PoisonedRAG and TrustRAG found overlapping perplexity distributions; Chang et al.
show repetition drives perplexity *below* clean text (-> our repeat attack).

**Specification** (GPT-2 small, `openai-community/gpt2`, revision pinned):
```
split the passage's GPT-2 tokens at the midpoint into pre and post
f(R)  = -(1/|R|) * sum_i log p(r_i | r_<i)      mean NLL per token, each half scored on its own after <|endoftext|>
PD(p) = f(pre) - f(post)                        two-tailed: a fluency break moves it either way
PM(p) = max(f(pre), f(post))                    one-tailed: high = unusually unfluent
```
- Passages with **< 20 GPT-2 tokens** are never flagged (`too_short`; 7.3% of NQ passages, 1.6% of pool passages).
- **Thresholds** from **1,000 seeded random clean corpus passages** (another 1,000 held out): PD band
  `[q(alpha/2), q(1 - alpha/2)]`, PM cutoff `q(1 - alpha)`; alpha 1, 2.5 (anchor), 5, 10%. Rule `either`.
  Ada thresholds at alpha 0.025: PD in [-1.928, 1.566], PM <= 6.502.
- **Score cache** (SQLite `caches/perplexity.sqlite`, key (namespace, doc_id)): every passage that can ever enter a
  pool (all questions' unmasked top-50 for every retriever + every poison passage) is scored once in A8 (39,128
  records, 15 MB); the sweeps then do no GPT-2 work (A9: perplexity filter ~1 ms per question on cache hits).
- **Namespace** = sha1(model id, revision, split rule, min_tokens, max_tokens): another GPT-2 or split rule never
  reuses old scores, and (since the 2026-10-27 audit) thresholds of another namespace are refused.
- Score for AUC: PM.

**Why each half is scored after `<|endoftext|>`:** so the first token of each half has a defined context (GPT-2's
document-start token) and both halves are scored the same way; scoring "post" with "pre" as context would measure
conditional fluency, not the half's own fluency, and the junction effect would leak into both.

**Ada results (A8):**
- Held-out flag rate on 1,000 clean passages at alpha 2.5%: PD 3.96%, PM 2.36%, either 4.93% (tolerance +/-1.8 points;
  both pass, but PD is noisy: ~12 passages per tail define the band).
- Sanity: a fluent passage PD 0.53 / PM 3.94 vs the same passage with a gibberish second half PD -8.19 / PM 12.46.
- Passage length: mean 102.8 GPT-2 tokens, median 87 -> each half ~45-50 tokens: noisy (the plan's caveat).
- **Detection sanity (alpha 2.5%, rule either): flag rate on poison** diverse 14.4%, hijack 13.8%, fact 12.5%, url
  6.4%, repeat 5.4%, stealth 4.0% vs 1.9% on clean pool passages -- **weak**, as the counter-evidence predicted.

**Samarth's fixes (audit 2026-10-27, file 15):** local calibration no longer overwrites the tracked Ada thresholds;
foreign-namespace thresholds refused; cache hits load no GPT-2; memory-safe scoring (per-row loss, VRAM cap, halving
on OOM) -- checked equal to the old code to < 1e-6.

---------------------------------------------------------------------------------------------------------------
## 4. Controls (Vansh; `src/ire/defenses/controls.py`; plan 4.4)
```python
def random_filter(pool, n_drop, seed, key):          # drops as many passages as the detector flagged
    rng = random.Random(f"{seed}:{key}")              # key = setting id + qid -> reproducible, order-independent
    return set(rng.sample(list(pool), min(n_drop, len(pool))))

def oracle_filter(pool):                              # drops every poison passage: the best any detector can do
    return {d for d in pool if is_poison_id(d)}
```
- **Random** answers RQ4: if a detector lowers ASR, is that because it *chose* the poison or just because removing
  passages helps (e.g. by backfilling)? The random control removes the **same number of passages for the same
  question** (count taken from the detector's run records), uniformly at random from the pool.
- **Oracle** gives the ceiling; `oracle_captured = (ASR_none - ASR_det) / (ASR_none - ASR_oracle)`.
- **No poison** (attack = None) and **no filter** are the other two reference settings.
- Combined filter (flag if any detector fires): optional, first to be cut after url; not in the grid.

## 5. Smoke test numbers (A9, Ada, 20 questions, stub pool = NO masking, all 5,005 poison present)
| retriever | poison in pools | flagged (poison) q-phrase | cluster | perplexity | mean ms q-phrase / cluster / ppl |
|---|---|---|---|---|---|
| bm25 | 226 | 154 (118) | 187 (182) | 36 (26) | 457 / 235 / 26 |
| dense | 191 | 132 (97) | 129 (122) | 35 (23) | 98 / 118 / 0.9 |
| splade | 220 | 148 (113) | 168 (158) | 36 (26) | 65 / 94 / 0.9 |
| rrf | 221 | 142 (108) | 200 (173) | 36 (26) | 60 / 120 / 60 |
These are **not results** (no masking: every attack's poison is present at once, which makes clusters easy); they
show the pipeline runs end to end with the real 3B model (143.7 s). Query-phrase (and generation, 75 s vs ~4 s) is
slowest on BM25, the first retriever the smoke test runs; most likely one-off loading (query encoder, LLM, cold
disk caches), not a property of BM25 -- not separately measured.

## 6. Likely viva questions
- *Why percentile thresholds from clean data instead of hand-set ones?* No poisoned data is used, nothing is tuned on
  the attack, and the false-positive rate on clean data is controlled by construction (alpha / tail).
- *Why is the perplexity filter weak here?* NQ passages are short (~45 tokens per half: noisy estimates), fact poison
  is fluent template English (no junction break), and repetition lowers perplexity.
- *Can the cluster filter catch rate-1 poison?* No -- one passage cannot form a cluster of >= 2.
- *Why does a detector need a random control?* Removing passages changes the context even when the wrong ones are
  removed; a detector must beat random removal of the same size to show its *choice* matters.
- *Why first 5 unflagged from 50 rather than re-retrieving?* Re-retrieval would cost a search per setting; the 50-pool
  is cached and gives room to backfill.
