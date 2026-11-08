# 05 -- Top-150 cache, poison ingestion and masking  (Samarth's lane, steps 7-8)

## 1. The top-150 cache (`src/ire/retrieval/cache.py`, `scripts/build_cache.py`, job A5)

### What it is
For every retriever and every one of the 400 questions: the top-150 `(doc_id, raw score)` list, retrieved once
from the index that contains **all** poison. One JSONL file per retriever:
`$IRE_DATA_DIR/caches/<env>/top150_<retriever>.jsonl`, one `RetrievalCacheRecord {qid, retriever, top150}` per line
in question-file order. Every experiment afterwards reads only these files: no retrieval runs during the sweeps.

### Why it is safe: the fingerprinted sidecar
Next to each cache file, `top150_<r>.jsonl.manifest.json` records the cache **key**:
```
key = { retriever, cache_version, questions_sha256 (bytes of questions.jsonl), depth (150),
        index fingerprint (copied from the index manifest: corpus sha256, every poison shard's sha256,
                           retriever settings, model id + exact revision) }
rrf: the keys of both component caches + rrf k and depth, and "unmasked_reference": true
```
`load_cache` recomputes the key from the **current** index manifests and questions file and raises
**`StaleCacheError`** if it differs. So a cache built before the held-out shard was ingested, or for an older
question set, *cannot* be used by mistake -- it fails loudly instead of silently giving wrong numbers. This is
exactly what happened in practice: after Arya ingested the held-out shard (A6 re-run), the old caches were refused
until A5 was re-run.

`load_cache(["rrf"])` raises **`UnmaskedReferenceError`** unless `allow_unmasked_reference=True`, because the fused
file must never be masked (file 04, section 4).

### Building it
- **Resumable**: rows are appended to a partial file; a re-run with the same key continues where it stopped.
- Rows sorted by score descending, ties by doc_id; BM25/SPLADE rows may be shorter than 150 (score > 0 only);
  dense rows must be exactly 150 (the builder enforces it).
- **Spot checks** (A5 acceptance): 20 random rows per retriever are compared with a fresh search and with
  `score_docs`. Ada A5 (2026-10-30, after the held-out ingestion): 400 rows per retriever, spot checks 20/20 for
  BM25, dense, SPLADE and RRF (dense `score_docs` max difference 4.8e-7, within float tolerance). Runtime 249 s.
- The questions file is fingerprinted by its **bytes**: re-saving it with CRLF line endings makes every cache stale
  (safe behaviour, worth knowing).

## 2. Poison ingestion (`scripts/index_poison.py`, `src/ire/retrieval/poison.py`, job A6)

### What it does
Adds every record of every `poison_*.jsonl` shard in `data/poison/` (Arya's `poison_main`, `poison_adaptive`,
`poison_heldout`; 5,005 passages in total) to each retriever's index:
| Retriever | How | Why |
|---|---|---|
| BM25 | **rebuild** with the shards (334 s) | bm25s cannot append: IDF and average length are global statistics |
| dense | **append**: clean rows copied bit for bit, only the 5,005 poison passages encoded | encoding 2.68M passages again would cost 46 minutes |
| SPLADE | **append**, same idea (CSR rows appended, inverted index rebuilt) | same |
After ingestion the index manifest fingerprint equals that of a full build with the shards, plus a
`poison_ingestion` block. Re-running is a no-op (A6 re-run: `noop: true` for all three).

### Validation before ingestion (`load_shards`)
every record must parse as a `PoisonRecord` (strict schema; `doc_id == poison_<qid>_<attack>_<n>` of its own fields);
no duplicate ids within or across shards or against the corpus; every qid must be a **target** question (a poison
passage aimed at a clean question would break the design). Counts per attack are reported.

### The subtle fp16 problem and `chunk_bounds` (a good viva story)
In fp16 a passage's vector depends very slightly on which other passages share its encoding batch (padding length,
kernel choice). If a full rebuild encoded clean and poison passages mixed in the same batches, its clean vectors
would differ in the last bits from a clean-only build, and "append == full rebuild" would fail. Fix:
```python
def chunk_bounds(n, n_clean, size, start=0):
    """Encoding chunks [s, e) of at most `size` rows that never mix clean rows (< n_clean) with poison rows."""
```
Clean rows are chunked exactly as in a clean-only build; poison rows get their own chunks. Now appending poison
later gives bit-identical vectors to a full build with shards (checked locally for dense and SPLADE).

### The masking-equivalence check (plan 2.1; `masking_equivalence`)
Question: is "retrieve from the poisoned index and drop every poison id" the same as "retrieve from a clean index"?
- Before the *first* ingestion, each clean index's top-150 for every question is saved as
  `masking_reference_<retriever>.json` (keyed by the clean fingerprint and the questions sha256). If an index already
  holds poison and no reference exists, a temporary clean copy is built for the comparison.
- After ingestion: poisoned top-150 with every `poison_` id removed vs the clean reference, on 50 questions.

**Ada A6 results (5,005 poison passages, 50 questions):**
| | identical ids+order | max rank diff | mean rank diff | top-5 unchanged | Jaccard@50 mean (min) |
|---|---|---|---|---|---|
| dense | 50/50 | 0 | 0 | 100% | 1.0 |
| SPLADE | 50/50 | 0 | 0 | 100% | 1.0 |
| BM25 | 0/50 | 36 | 0.43 | 96% | 0.9945 (0.961) |
BM25 differs because poison changes IDF and avgdl; the change is tiny in the region that matters (top-5 and
top-50), and it is a documented limitation.

### Poison reach (A6): fraction of poison passages in their OWN target question's unmasked top-10
| attack | BM25 | dense | SPLADE |
|---|---|---|---|
| fact | 0.915 | 0.735 | 0.846 |
| hijack | 0.602 | 0.228 | 0.264 |
| url | 0.081 | 0.296 | 0.141 |
| diverse | 0.554 | 0.422 | 0.112 |
| repeat | 0.002 | 0.268 | 0.698 |
| stealth | 0.000 | 0.000 | 0.006 |
| heldout | 0.032 | 0.109 | 0.174 |
(top-150 reach is 1.0 for fact/hijack/url on all three; stealth 0.51-0.65.) This table already answers part of RQ5:
fact poison (question + answer sentence) reaches the top-10 most often everywhere, and most under BM25. Hijack
reaches BM25's top-10 far more than dense's (0.60 vs 0.23); url the reverse (0.08 vs 0.30). Our reading (an
interpretation, not tested): instruction text pulls the passage's *meaning* away from the question, which matters
for dense, while BM25 only needs the question words, which every hijack passage contains. We have no tested
explanation for url's low BM25 reach (hijack and url passages are about the same length, 24 words). Say "we
observed ...; one possible explanation is ...", never present it as established.

Unmasked top-5 changed vs clean: about half of all questions (bm25 0.51, dense 0.48, splade 0.50 -- driven by the
target questions); for **clean** questions only 2% / 0% / 1%: poison written for one question rarely reaches
another question's top-5 even with *all* attacks active.

### Local verification with a synthetic fixture
`tests/fixtures/poison_fixture.jsonl`: 10 passages (5 fact + 5 hijack with codeword ZEBRAFISH) written by Samarth for
5 final target questions -- **test fixtures only**, never put into `data/poison/` (index_poison refuses a tests/
folder unless `--work-copy`). `--work-copy` runs everything on temporary copies so the real indexes stay clean.
Local results: every fixture passage in its target's top-10 for all three retrievers; dense/SPLADE masking 50/50
exact; BM25 max rank difference 10; append == full rebuild; re-run is a no-op; removing the shard restores the clean
index.

## 3. Masking at experiment time (`candidate_pool`, Vansh) -- recap
`is_active(doc_id, attack, rate)` = same attack and `n <= rate`. Poison of other target questions of the same attack
stays active (collateral). `attack=None` = no-poison setting (all poison masked). RRF: mask both lists, then fuse.
Full code in file 02.

## 4. Likely viva questions
- *Why not build one index per setting?* ~20 settings x (46 min dense + 54 min SPLADE + 6 min BM25) and ~19 GB each:
  impossible on our quota. Masking gives the same result (exactly for dense/SPLADE) at the cost of one index.
- *Why is the BM25 equivalence only approximate, and why is it acceptable?* IDF(t) = ln(1 + (N - df + 0.5)/(df + 0.5))
  and avgdl depend on all documents; 5,005 extra passages among 2.68M (0.19%) move IDF of question words slightly.
  Measured: top-5 unchanged 96%, mean rank difference 0.43.
- *How do you know the cache is not stale?* The sidecar key includes content hashes of the corpus, every shard and
  the questions file plus model revisions; the loader recomputes and compares.
- *What happens if a teammate re-runs A6 but forgets A5?* `load_cache` raises `StaleCacheError`; the sweep refuses to
  start. (`check_ada_report.py A5` also warns if the indexes hold 0 poison documents.)
- *Why save the clean reference before the first ingestion?* After ingestion the clean index no longer exists; the
  masking check needs a clean baseline. Rebuilding a clean dense index just for the check would cost 46 minutes.
