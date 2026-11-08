# 04 -- Retrievers: BM25, dense (E5), SPLADE, RRF  (Samarth's lane, steps 3-6)

All four share one interface: `search(queries, k) -> [[(doc_id, score), ...]]` (score descending, **ties by
doc_id**), and BM25 / dense / SPLADE also have `score_docs(query, doc_ids)` (the score of any document under the
current index -- needed for the masking check and for spot checks). Indexes live in
`$IRE_DATA_DIR/indexes/<env>/<retriever>/` (local and Ada never mix) with a `manifest.json` that records a
**fingerprint** of what the index was built from.

## Ada numbers at a glance (full corpus, 1,000 seeded questions with gold in the index)
| Retriever | Recall@10 | Recall@100 | Build time | Size | Query latency |
|---|---|---|---|---|---|
| BM25 (bm25s) | 0.444 | 0.744 | 334 s | 1.0 GiB | 12.5 ms median |
| Dense e5-base-v2 + FAISS Flat | 0.781 | 0.941 | 2,775 s (1,000 passages/s) | 11.6 GiB | 21 ms/query in batch (341 ms single, incl. model) |
| SPLADE cocondenser-ensembledistil | 0.733 | 0.927 | 3,261 s (844 passages/s) | 6.0 GiB | 59 ms/query in batch |
| RRF(BM25, dense) | fused at query time from the cached lists | | | | |
BM25's Recall@100 of 0.744 is close to BEIR's published 0.76 for NQ (the gap: single title+text field and bm25s's
tokeniser vs Anserini's multi-field Lucene analyser). Dense > SPLADE > BM25 on NQ is the expected order.

---------------------------------------------------------------------------------------------------------------
## 1. BM25 (`src/ire/retrieval/bm25.py`, step 3)

### What it computes (Lucene variant, as bm25s `method="lucene"`)
```
score(q, d) = sum over query terms t in d of  IDF(t) * tf(t,d) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )
IDF(t)      = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )
```
k1 = 0.9, b = 0.4 (the Anserini/pyserini defaults used for the BEIR BM25 baselines). Lucene drops BM25's
`(k1 + 1)` numerator factor; it is a constant per query, so the ranking is unchanged.
- **k1** controls term-frequency saturation (0.9 = saturates fast: the 5th occurrence adds little).
- **b** controls length normalisation (0.4 = mild: long passages are only partly penalised).

### Settings (`bm25:` in `configs/experiment.yaml`, stored in the manifest and checked on load)
lowercase; token pattern `(?u)\b\w\w+\b` (runs of 2+ word characters); bm25s English stopwords; PyStemmer Snowball
English stemmer; indexed text = `title + " " + text`.

### Why bm25s and not pyserini
The plan allowed either. bm25s is pure Python (numpy/scipy sparse) -- **no Java**, installs with pip on Windows and on
Ada, memory-mapped index loading, and it is fast enough (334 s to index 2.68M passages). pyserini needs a JDK and
Lucene; on the cluster and on Windows that was extra risk for no gain in a project about *poison*, not BM25.

### Implementation decisions you must be able to defend
1. **Exact scoring of every document, ties by doc_id.** `search` calls bm25s `get_scores(tokens)` (a score for every
   document) and picks the top-k with `top_k_rows`:
   ```python
   cand = np.flatnonzero(scores > 0)                        # only documents sharing a query term
   kth  = np.partition(scores[cand], len(cand) - k)[len(cand) - k]
   cand = cand[scores[cand] >= kth]                         # keep EVERY tie at the k-th place
   return cand[np.lexsort((id_rank[cand], -scores[cand]))[:k]]   # score desc, then doc_id asc
   ```
   bm25s's own `retrieve` uses a partial sort whose choice among tied documents depends on index position. The
   local corpus already has identical passages (exact ties), so without the doc_id tie-break the cache and a fresh
   search could disagree. Now the output is a pure function of (query, corpus), identical across builds.
2. **Score > 0 only.** A document sharing no query term has score 0 and is never returned, so a row can be
   **shorter than 150** (recorded under "Interface changes"; on the full corpus every question still gets 150).
3. **The empty-token pitfall.** bm25s maps an empty document or query to a special `""` token (id 0). Without
   care, a stopword-only query would *match every stopword-only document*. Fix: query token id 0 is dropped
   (`EMPTY_TOKEN_ID = 0`), so an empty/stopword-only/unknown-word query returns `[]`.
4. **Atomic build.** The index is written to `<dir>.building` and swapped in at the end (`swap_in`), so a crash never
   leaves a half index that looks complete.
5. **`score_docs` under current statistics** -- this is how the BM25 masking approximation is measured.
6. **Unique-token test** (plan A2 acceptance): build a throw-away 50,001-passage index with one passage containing
   the invented word "zorblaxquintavel", search it, expect rank 1 (Ada: rank 1, 1 hit).

### BM25 and poison (important for RQ5)
Poison starts with the question text, so it shares *all* query terms -> BM25 ranks it very high: fact poison is in
its question's top-10 **91.5%** of the time (A6) vs 73.5% dense. Stealth poison (no question prefix) reaches BM25's
top-10 **0%** of the time. Repeat poison (question once, then a phrase repeated ~70 words) -- BM25's length
normalisation and tf saturation punish it: 0.2% in top-10 vs 69.8% for SPLADE.

---------------------------------------------------------------------------------------------------------------
## 2. Dense retriever: E5 + FAISS (`src/ire/retrieval/dense.py`, step 4)

### Model
`intfloat/e5-base-v2` on Ada (768 dims), `e5-small-v2` locally (384 dims). E5 is a contrastively trained bi-encoder.
**E5 requires prefixes**: queries are encoded as `"query: " + q`, passages as `"passage: " + title + " " + text`.
Without them quality drops noticeably (the model card says so); the encoder refuses to run an E5 model with other
prefixes, and a test checks the prefixes change the vector.

### Encoding
- Mean pooling over non-padding tokens of the last hidden state, then L2 normalisation (in float32), so the inner
  product equals cosine similarity.
- fp16 on GPU; a per-process VRAM cap (`compute.max_vram_gb`, 5.0 GB locally) and **automatic batch halving on CUDA
  out-of-memory**; max_length 512 tokens.
- Stored as `doc_vectors.npy` in **float16** (2.68M x 768 x 2 B = 4.1 GB), opened as a read-only memmap.
- **Resumable**: checkpoints every 50,000 passages in a `.building` folder; an interrupted build with the same
  fingerprint resumes from the last checkpoint (Ada time limits).
- **Queries are encoded one at a time** (`encode_queries` uses `batch_size=1`). In fp16, a vector depends slightly on
  the other texts in its batch (padding, kernel choice). A query's top-150 in the cache and its `score_docs` later
  must agree exactly, so a query vector must depend only on the query text. Passages are still batched (they are
  encoded once and stored).

### Index: FAISS `IndexFlatIP` (exact) on both machines
- **Decision (Samarth, before step 5): Ada uses exact Flat, not HNSW.** Masking is only exactly equivalent to a
  separate index when search is exact (plan 2.1). HNSW is approximate: whether a document is found depends on graph
  navigation, which changes when poison is added. Cost: the Flat index holds float32 vectors (8.2 GB RAM) and every
  search is a full scan (~21 ms per query on Ada) -- acceptable for 400 questions x a few retrieval passes.
- **Exact top-k with ties by doc_id**: FAISS fetches k + 16 candidates; if the k-th score equals the last fetched
  score (a tie may continue), fetch doubles until it is strictly above; then sort by (score desc, doc_id asc).
- **One query per FAISS call** (Arya's fix, kept): FAISS uses BLAS for batches of >= 20 queries and a direct loop for
  fewer; they round differently (~1e-7), which reordered near-ties (A5 spot check 17/20 with batches of 32).
- **The FAISS index is loaded lazily** (only when a search runs). The filters only need `get_doc_vector` (read from the
  memmap), so they do not hold the 8.2 GB index in RAM.

### Vector API (section 2 contract; `src/ire/vectors/__init__.py`)
```python
get_doc_vector(["doc12", "poison_q1_fact_1"])  # float32 [n, d], L2-normalised, READ from disk, never re-encoded
get_query_vector(["who wrote hamlet"])          # float32 [n, d], "query: " prefix
```
Used by the cluster filter (vectors of the top-20) and the query-and-phrase filter (s(q,p) = q . p). Reading stored
vectors guarantees the filter sees exactly the vectors the retriever used.

### Checks (A3 on Ada)
re-encode 200 stored passages: min cosine 0.999973 (fp16 storage loses almost nothing); norms 0.99994-1.00006;
self-retrieval (a passage's own text as the query) top-1 for 99/100; VRAM peak 1.28 GiB.
(The unique-token test is meaningless for a dense encoder -- it does not match an invented word exactly -- so
self-retrieval replaces it; recorded under "Interface changes".)

---------------------------------------------------------------------------------------------------------------
## 3. SPLADE (`src/ire/retrieval/splade.py`, step 5)

### Model and representation
`naver/splade-cocondenser-ensembledistil` (BERT vocabulary, 30,522 terms). A *learned sparse* model: for each
vocabulary term j it outputs a weight
```
w_j = max over tokens t of  log(1 + ReLU(logit_{t,j}))        (max pooling, SPLADE v2 / ++)
score(q, d) = sum_j  w_j(q) * w_j(d)                           (sparse dot product)
```
The MLM head means a passage also gets weight on terms it does **not** contain (*expansion*: "car" -> "vehicle"),
which is why SPLADE fixes BM25's vocabulary mismatch while keeping an inverted index.

### Implementation decisions
1. **`log1p(relu(max_t logit))` instead of `max_t log1p(relu(logit))`** -- identical because log1p and ReLU are
   monotone non-decreasing, but it avoids allocating a second [batch, length, 30522] tensor (a test checks the two
   formulas agree).
2. **Lengths**: documents truncated to 256 tokens, queries to 64.
3. **No pruning** of document terms (`prune_doc_terms: false`; true is refused). Pruning makes scores approximate,
   and then masking would no longer equal a separate index. Cost: ~146.5 non-zero terms per passage, 392.9M
   non-zeros in total, 6.0 GiB on disk.
4. **Storage**: the document matrix as CSR (`doc_matrix.{data,indices,indptr}.npy`) and its transpose, an inverted
   index (`inverted.*.npy`: rows = vocabulary terms = posting lists), all memory-mapped.
5. **Exact search**: sum the posting lists of the query's terms (queries average 45.6 terms) -> a score for every
   document that shares a term -> top-k with ties by doc_id, score > 0 only.
6. **Empty query guard**: an empty/whitespace query returns `[]`; otherwise SPLADE would "expand" the bare
   [CLS]/[SEP] into 126 terms and return arbitrary documents.
7. Queries one at a time (fp16 batch effect: up to 0.01 in a score, enough to reorder near-ties).
8. Resumable checkpoints as for dense.

### Checks (A4 on Ada)
exactness: `search` scores vs a manual dot product of stored vectors, max relative difference 2.6e-7 over 2,000
pairs; `score_docs` equals `search` exactly; self-retrieval 95/100 (5 misses are near-duplicate passages);
every weight non-negative; no empty document rows.

---------------------------------------------------------------------------------------------------------------
## 4. RRF hybrid (`src/ire/retrieval/rrf.py`, step 6)

```python
def fuse_many(lists, k=60, depth=150):
    contributions = {}
    for ranked in lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):     # only the ORDER is used
            contributions.setdefault(doc_id, []).append(1.0 / (k + rank))
    scores = {d: math.fsum(c) for d, c in contributions.items()}       # exactly rounded: order-independent
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:depth]  # ties by doc_id
```
- k = 60 (Cormack et al. 2009, and the plan). Components: BM25 + dense (the plan's hybrid), each to depth 150.
- **Why RRF and not a weighted score sum?** BM25 scores are unbounded (~10-40), cosines are in [-1, 1]: a linear
  combination needs normalisation and a tuned weight. RRF uses ranks only -- no tuning, robust, standard.
- **Why `math.fsum`?** Floating-point addition is not associative; fsum is exactly rounded, so the fused score of a
  document does not depend on the order of the lists.
- **Masking must happen before fusing** (section 2 of file 02). The module never masks; `candidate_pool` masks the
  BM25 and dense rows and then calls `fuse`. The cached `top150_rrf.jsonl` is an *unmasked, no-poison reference
  only*, and `load_cache(["rrf"])` refuses it unless `allow_unmasked_reference=True`.
- A toy test shows the failure mode: fusing first and deleting poison afterwards swaps two clean passages (c2, c3).

## 5. Likely viva questions (short answers; more in file 16)
- *Why is dense better than BM25 on NQ?* NQ questions are natural-language and often paraphrase the passage;
  dense models capture meaning beyond exact words. BM25 needs lexical overlap.
- *Why do poison passages rank so well under BM25?* They contain the question verbatim: maximal term overlap.
- *Why is SPLADE so vulnerable to the repeat attack (69.8% top-10 vs BM25 0.2%)?* Measured fact (A6); the
  explanation is our hypothesis, not tested: SPLADE's max pooling ignores how often a term occurs, so a repetitive
  passage loses nothing for its length, while BM25 saturates term frequency and its length normalisation penalises
  the ~72-word repetitive passage relative to the short question-prefixed attacks. Say "we observed / we think".
- *Why fp16 storage but float32 normalisation?* Normalising in fp32 avoids rounding in the norm; the stored fp16
  vectors still have cosine 0.99997 with fresh float32 encodings.
- *What breaks if you use HNSW?* The equivalence "mask after retrieval == separate index" (approximate search can
  miss different documents when the index changes), and exact ties.
