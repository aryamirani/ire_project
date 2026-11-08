# 15 -- Bugs found, how they were fixed, and the lesson

These are excellent viva material: they show you understand the system at the level where things break. Each entry:
symptom -> cause -> fix -> lesson.

## Retrieval and indexing
1. **BM25 empty-token match.** Symptom: a stopword-only query returned stopword-only documents. Cause: bm25s gives
   every empty document and query a special `""` token (id 0), and those match each other. Fix: drop token id 0
   from queries (`EMPTY_TOKEN_ID`). Lesson: test edge-case queries (empty, stopwords, unknown words).
2. **Ties ordered by index position.** Symptom: the cache and a fresh search disagreed on near-identical passages.
   Cause: libraries break ties by internal position; the local corpus has duplicate passages (exact ties). Fix:
   `doc_id_ranks` + `top_k_rows` -- ties by doc_id everywhere, including at the k-th place (keep every tie at the
   boundary, then sort). Lesson: "deterministic" must include ties.
3. **fp16 batch-dependent query vectors.** Symptom: `score_docs` and `search` disagreed by up to 0.01 (SPLADE). Cause:
   in fp16 a vector depends on the other texts in the batch (padding, kernels). Fix: encode queries one at a time.
   Lesson: numerical results can depend on batch composition on GPUs.
4. **FAISS BLAS vs loop rounding (Arya, Ada).** Symptom: A5 dense spot check 17/20 with batches of 32 queries. Cause:
   FAISS uses BLAS for >= 20 queries and a direct loop otherwise; they round differently (~1e-7). Fix: one query per
   FAISS call. Lesson: the same as 3, for the search library.
5. **Append != rebuild in fp16.** Symptom: appending poison vectors gave clean vectors that differed in the last bits
   from a full rebuild. Cause: mixed clean/poison encoding batches. Fix: `chunk_bounds` keeps clean and poison rows in
   separate encoding chunks. Lesson: equivalence claims need bit-level tests.
6. **Model loading took 111 s.** Cause: transformers checked the Hub on every load. Fix: `resolve_model` resolves the
   local snapshot and pins the revision (no network). Lesson: offline-by-default for clusters.
7. **Windows locks memory-mapped files.** A throw-away BM25 index could not be deleted. Fix: `mmap=False` for
   throw-away indexes.
8. **FAISS index held in RAM by the filters (Arya).** The filters need only stored vectors; the 8.2 GB index is now
   loaded lazily (only for searches).

## Data
9. **Slow / failing original download.** UKP server 0.06-0.4 MB/s and a TLS error. Fix: pinned Hugging Face copy, with
   a test that both sources give identical outputs.
10. **Placeholder questions vs final set.** After Arya froze the final 400, the local data and the poison fixture still
    used placeholder qids; `prepare_data.py --force` would even overwrite `questions.jsonl` with the placeholder.
    Fix (Arya + Samarth): prepare_data builds the frozen set; local corpus, indexes and caches rebuilt; fixture
    re-targeted. Checked byte-identical to Ada.
11. **Line endings change hashes (2026-11-08).** On Windows, git converted the frozen files to CRLF, so
    `question_set.json` hashed 2f2e7... locally vs fb6f5... committed. Fix: `.gitattributes` forces LF for
    `configs/frozen/**` and `heldout/**`. Lesson: hash bytes only of files whose bytes you control.

## Generation
12. **Qwen's hidden repetition penalty.** Qwen's generation_config sets sampling and `repetition_penalty 1.1`;
    `do_sample=False` disables sampling but the penalty still changes greedy outputs (1 of 12 answers differed).
    Fix: set every decoding parameter explicitly (penalty 1.0). Lesson: library defaults from model repos override
    what you think "greedy" means.
13. **Memory in the batch-independence probe.** Computing first-token logits for all positions ran out of memory; fix:
    `logits_to_keep=1`.
14. **Host RAM on Ada (Arya).** Loading the 3B model on CPU then moving it held a full copy in RAM; A9 was OOM-killed on
    a node with 5 GB free. Fix: `device_map` loads straight onto the GPU.

## Perplexity filter (Samarth's audit, 2026-10-27)
15. **Local calibration overwrote the tracked Ada thresholds.** `calibrate_perplexity.py` always wrote
    `configs/perplexity_thresholds.json`; one local run + commit would silently replace the results' thresholds. Fix:
    on Ada the tracked file; elsewhere `$IRE_DATA_DIR/reports/perplexity_thresholds_<env>.json` unless
    `--write-config`.
16. **Thresholds of another scorer accepted.** Thresholds are percentiles of one GPT-2 revision's scores. Fix: the
    filter refuses a table whose namespace differs from the scorer's.
17. **GPT-2 loaded on pure cache hits.** `namespace()` loaded the weights just to read the revision. Fix: `_resolve()`
    gets the revision without loading the model.
18. **CUDA out-of-memory on a 6 GB card.** The batched loss `cross_entropy(logits[:, :-1].reshape(-1, V))` copied the
    [tokens x 50,257] logits (a reshape of a sliced tensor makes a copy) and the loss needed ~3x their size; next to
    the LLM in the smoke test it crashed. Fix: loss computed row by row on views, VRAM cap, halve the token budget and
    retry on OOM (batches computed once, re-split only after an OOM). Verified on 300 passages: PD/PM equal to the old
    code within 1e-6; namespace unchanged, so Ada's A8 stays valid.

## Query-and-phrase thresholds (2026-11-08)
19. **Same overwrite bug in `calibrate_query_phrase.py`**, and worse: local runs used the Ada cutoffs (cosines of
    e5-base-v2) with e5-small-v2 vectors, which is meaningless. Fix: shared `Config.thresholds_path` /
    `thresholds_write_paths`; local runs read their own table; `load_thresholds(path, cfg)` refuses a table calibrated
    with another dense model. Lesson: thresholds are tied to the model that produced the scores.

## Process
20. **A hung job (laptop sleep)** and **a job killed for low RAM** -- local jobs need the machine awake and memory free;
    long runs go to Ada.
21. **Stale caches after ingesting the held-out shard** -- exactly what the fingerprint design is for: the loader
    refused them until A5 was re-run.

## Open issues to mention honestly
- BM25 masking is approximate (numbers in file 05).
- Batch-size decision pending A7 (file 08 section 4).
- `heldout` attack mixes 100 hijack and 5 fact questions in the sweep; proposed: score the 100 only (team decision,
  frozen rule).
- Some false targets are semantically odd (coarse `other` type).
- Perplexity PD clean flag rate 3.96% vs alpha 2.5% (within tolerance, noisy).
- Poison passages have no title, so they look different in the prompt ("[i] text" vs "[i] Title: text").
