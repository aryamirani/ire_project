# docs/IMPLEMENTATION_PLAN.md -- Samarth's lane

Design source of truth: `docs/IRE_Final_Project_Plan.pdf`. This file says *what to build, in what order, and how to check it*. Progress is recorded in `docs/HANDOFF.md`.

## 0. How to work (read first)

For every step:
1. Set the step to IN PROGRESS in `docs/HANDOFF.md`.
2. Implement it (small, readable modules; type hints; docstrings stating inputs/outputs).
3. Run every check under **Verify** on the local profile. Paste real outputs (numbers, counts, timings) into the HANDOFF step log, labelled `env=local`.
4. If a check fails: fix and re-run. If stuck after a reasonable attempt: mark BLOCKED with the exact error and stop.
5. Add or update a test in `tests/` for the step (pytest). `pytest -q` must pass before marking DONE.
6. Set the step to DONE-LOCAL (never DONE; see section 0.5), raise the Ada request if the step has one (section 5), and update `docs/HANDOFF.md`: status board row, "What teammates can use right now", step log, open issues.
7. Do not commit; leave changes uncommitted and tell the user which files changed and a suggested commit message (`step N: <title>`). The user commits and pushes. Then start the next step.

Rules: no fake data in verify checks (small real subsets are fine; label them as subsets). Develop and test on the **local profile** (section 0.5) first. The 'dev subset' mentioned in step texts means the local profile on Samarth's machine, and a larger sample (1,000 questions + 50,000 passages) on Ada. Anything large is an **[ADA]** job run by a teammate (section 5), never run locally.

## 0.5 Execution model: two environments (IMPORTANT)

Samarth has **no Ada access**. Arya and Vansh do. So the work runs in two places:

- **[LOCAL]** (Samarth's machine, CPU or a free Colab/Kaggle GPU): write all code, unit tests, and verify everything on a **tiny dev subset** with **small stand-in models**. Nothing here needs Ada.
- **[ADA]** (a teammate with Ada access): run the heavy jobs on the full or large data with the real models, using the job scripts in `ada/` and the steps in `docs/ADA_RUNBOOK.md`.

Rules that follow from this:
1. Code is written once and runs in both places. The only differences live in two config files: `configs/env_local.yaml` and `configs/env_ada.yaml`, selected with `IRE_ENV=local|ada`. No code branches on the environment except through config.
2. **Local profile** (defaults, change only if the machine cannot cope): corpus = 5,000 passages (all gold passages for the 100 dev questions + random others, seed 13); 100 dev questions; dense model = a small variant of the chosen family (e.g. `intfloat/e5-small-v2` or `BAAI/bge-small-en-v1.5`); SPLADE on the 5,000 passages (CPU is fine, slow); LLM stand-in = a small instruct model (0.5B-1.5B, e.g. a Qwen2.5-0.5B/1.5B-Instruct class model) behind the same `generate()` interface; GPT-2 small runs fine on CPU. **Local numbers are for correctness only, never for results.** Every local report must be labelled `env=local`.
3. **Ada profile**: real models and the full or 500k corpus, as in the project plan.
4. A step has two completion states:
   - **DONE-LOCAL**: code written, tests pass, Verify checks pass on the local profile.
   - **DONE**: DONE-LOCAL plus the matching Ada job ran and its report passes the Ada checks.
   Claude Code may only set DONE-LOCAL by itself. DONE needs a pasted Ada report (see section 5).
5. Never block on Ada. When a step is DONE-LOCAL, immediately create the Ada request (section 5), record it in `docs/HANDOFF.md` under "Ada requests", and continue with the next step. Steps that need real Ada output as input (nothing in steps 0-13 does, because each step can be tested locally) are marked `needs-ada-output`.
6. Every Ada job script must be: idempotent and resumable, write to `$IRE_DATA_DIR`, write a small JSON `reports/<job>.json` (counts, timings, metrics, git commit, env, GPU, library versions), and print the exact command to re-run. Teammates should only have to run one command and send back the report file.
7. Write teammates' instructions so a non-expert can follow them: exact commands, expected runtime, expected output files, and what "success" looks like.

## 1. Fixed decisions (from the project plan)

- Dataset: BEIR Natural Questions (retrieval) + `nq_open` (short answers). 200 target + 200 clean questions, two folds (A/B), chosen by Arya; until then use a placeholder list of 400 questions with seed 13.
- Corpus: full BEIR NQ (~2.7M) if storage allows, else ~500k subset (all gold + hard negatives). Controlled by `corpus.mode` in config (`full` | `subset`).
- Retrievers: `bm25` (pyserini or bm25s), `dense` (E5 or BGE with correct query/passage prefixes; model name in config), `splade` (`naver/splade-cocondenser-ensembledistil`), `rrf` (BM25 + dense, score = sum 1/(60+rank)).
- One index per retriever over clean corpus **plus all poison ever generated**. Poison masking is done later by Vansh's `candidate_pool`; you never build per-setting indexes.
- Cache: raw top-150 `(doc_id, score)` per (retriever, qid).
- Generator: one fixed 3-4B instruct model, temperature 0, greedy. Neutral prompt (see 3.9). Cache key = (qid, ordered tuple of the 5 passage IDs).
- Filters in my lane: cluster filter and perplexity filter. Both are training-free and run on the masked top-50 pool.
- Seeds: global seed 13 everywhere unless config says otherwise.

## 2. Contracts with teammates (do not break; record changes in HANDOFF)

All files are JSONL unless stated. IDs: clean docs keep their BEIR `_id` (string); poison docs are `poison_<qid>_<attack>_<n>` with attack in `{fact, hijack, url, diverse, repeat, stealth, heldout}`.

**Corpus record** (`corpus.jsonl`): `{"doc_id": str, "title": str, "text": str, "is_poison": false}`
**Poison record** (Arya's `poison_*.jsonl`): `{"doc_id": "poison_q123_fact_1", "qid": "q123", "attack": "fact", "title": "", "text": str, "is_poison": true, "target_answer": str}`
**Question record** (`questions.jsonl`): `{"qid": str, "question": str, "answers": [str], "gold_doc_ids": [str], "role": "target"|"clean", "fold": "A"|"B"}`
**Retrieval cache** (one file per retriever, `top150_<retriever>.jsonl`): `{"qid": str, "retriever": str, "top150": [[doc_id, score], ...]}` sorted by score descending, length 150 (fewer only if the index is smaller).
**Vector API** (mine, used by cluster filter and Arya's similarity filter): `get_doc_vector(doc_ids: list[str]) -> np.ndarray [n, d]` (L2-normalised dense vectors from the dense retriever's passage encoder) and `get_query_vector(texts: list[str]) -> np.ndarray [n, d]`.
**Filter API** (mine and Arya's, so Vansh's runner can call any filter the same way):
```python
def apply_filter(pool: list[str], qid: str, question: str, ctx: FilterContext) -> FilterResult
# pool: ranked doc_ids (top-50 after masking). FilterResult.flagged: set[str]; FilterResult.scores: dict[str, float]
```
`FilterContext` carries texts-by-id, vector API, config, and fold-specific thresholds. The runner takes the first 5 unflagged IDs in rank order.
**Candidate pool** (Vansh): `candidate_pool(qid, retriever, attack, rate, cache, pool_size=50)`. Until his version is merged, use the stub in `src/ire/stubs.py` that returns the first 50 cached IDs without masking.
**Generation call** (mine): `generate(qid, question, top5_doc_ids) -> {"answer": str, "cached": bool}`.

## 3. Steps

### Step 0 -- Repo skeleton and environment
Build (tags: [LOCAL] + creates the `ada/` and `configs/env_*.yaml` scaffolding):
- `configs/env_local.yaml` and `configs/env_ada.yaml` with the profiles from section 0.5; `IRE_ENV` selects one.
- `ada/` directory with `setup_env.sh` (conda/venv creation, pinned installs, model pre-download to `$IRE_DATA_DIR/models`, Java check if pyserini is used) and `common.sh` (sources `.env`, activates the environment, prints GPU info). Add `ada/README.md` pointing to `docs/ADA_RUNBOOK.md`.
- `docs/ADA_RUNBOOK.md` already exists in the repo; keep it in sync with the job scripts.
- Layout: `src/ire/{config,data,retrieval,vectors,generation,defenses,utils}/`, `defenses/cluster.py`, `defenses/perplexity.py`, `scripts/`, `tests/`, `configs/`, `docs/`, `data/` (gitignored), `.env.example`, `requirements.txt` (pinned), `pyproject.toml`, `.gitignore` (data, indexes, caches, `.env`, `__pycache__`).
- Put `IRE_Final_Project_Plan.pdf` into `docs/` if missing.
- `scripts/check_env.py` printing: Python version, torch version, CUDA availability and GPU name/memory, free disk under `$IRE_DATA_DIR`, and whether pyserini/Java, transformers, sentence-transformers, faiss are importable.
Verify ([LOCAL]):
- `python scripts/check_env.py` runs and shows GPU (or states clearly that none is available) and disk space.
- `pytest -q` collects and passes a trivial test.
- `git status` clean after commit; confirm no data files tracked.

### Step 1 -- Config, paths, logging, contract schemas
Build:
- `configs/paths.yaml`, `configs/experiment.yaml` (seed, corpus.mode, models, pool_size=50, top_k_cache=150, final_k=5), loaded by `src/ire/config.py` with `$IRE_DATA_DIR` expansion.
- `src/ire/schemas.py`: dataclasses/pydantic models for every record in section 2 plus read/write JSONL helpers with validation.
- `src/ire/utils/log.py`: logger writing to console and `$IRE_DATA_DIR/logs/`.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- `scripts/check_ada_report.py <job>`: reads `reports/<job>.json` and checks it against the Ada acceptance checks in section 5 (stub the rules now, fill them in as each job is added).
- Unit tests: round-trip each record type; invalid records (missing field, wrong type) raise.
- Loading config with a missing path key fails with a clear message.

### Step 2 -- Load NQ corpus and questions
Build:
- `scripts/prepare_data.py`: download BEIR NQ (corpus, queries, qrels) and `nq_open` answers, normalise into `corpus.jsonl`, `questions_all.jsonl` (with gold doc IDs from qrels and answer strings from nq_open where matched). Support `--dev-subset` (1,000 questions + their gold + 50,000 random passages, seed 13) and `--corpus-mode subset|full`.
- Placeholder question set: 200 `target` + 200 `clean`, split into folds A/B (stratified so each fold has 100+100), saved as `questions.jsonl` with a note in HANDOFF that Arya's final set will replace it.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Print counts: corpus size, questions with at least one gold passage, answer coverage rate.
- Assert no duplicate doc_ids; assert every gold ID exists in the corpus; assert fold balance (target/clean x A/B each 100 questions).
- Show 3 sample records.

### Step 3 -- BM25 index and retriever
Build: `retrieval/bm25.py` with `build_index(corpus_path, out_dir)` and `search(queries, k) -> list[list[(doc_id, score)]]`. Title + text indexed. Include `include_poison_shards` option that appends any `poison_*.jsonl` found under the poison dir.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- On the dev subset: Recall@100 for gold passages reported (expect clearly above random; record the number).
- Determinism: the same query returns identical results twice.
- A passage inserted with a unique made-up word is found at rank 1 for a query using that word.

### Step 4 -- Dense index and retriever + vector API
Build: `retrieval/dense.py` and `vectors.py`.
- Encoder from config (E5 or BGE) with the **correct prefixes** (E5: `query: ` / `passage: `; BGE query instruction as per model card). Batch encoding with fp16 on GPU, L2-normalised, stored as float16 memmap + `doc_ids.npy`.
- FAISS inner-product index (flat for dev subset; HNSW or IVF-flat for full corpus; parameters in config).
- `get_doc_vector`, `get_query_vector` as in section 2, reading from the stored matrix (no re-encoding).
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Prefix test: assert encoding with and without prefix gives different vectors (guards against forgetting them).
- Recall@100 on dev subset recorded; compare to BM25 and note which is higher.
- `get_doc_vector(ids)` row i matches freshly re-encoded passage i (cosine > 0.999).
- Vector norms are 1 +/- 1e-3.

### Step 5 -- SPLADE index and retriever
Build: `retrieval/splade.py` using `naver/splade-cocondenser-ensembledistil`: sparse encode (max-pooled log(1+ReLU)), store as an inverted index (pyserini impact index or a scipy CSR matrix with a dot-product search on dev; document the choice). Keep top terms count and query max length in config.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Recall@100 on dev subset recorded.
- Sparsity report: mean non-zero terms per passage and per query.
- A query's top result for its own gold passage text is the same passage (sanity).

### Step 6 -- RRF hybrid retriever
Build: `retrieval/rrf.py`: fuse BM25 and dense **ranked lists** with score = sum over lists of 1/(60 + rank). Expose `fuse(list_a, list_b, k)`, because Vansh's masking must mask each component list before fusing; do not store only fused output.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Unit test with hand-built lists (including a doc in only one list) and the expected fused order.
- Recall@100 on dev subset is at least as good as the worse of BM25/dense (report all three).

### Step 7 -- Top-150 cache builder
Build: `scripts/build_cache.py --retriever {bm25,dense,splade,rrf} --questions questions.jsonl`. For rrf, cache the two component top-150 lists as well as the fused list (`top150_rrf.jsonl` plus the component files). Resumable (skips qids already written), writes atomically.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Row count equals number of questions for each retriever; each row has 150 entries sorted descending.
- Re-running is a no-op (resume works).
- Spot check: cached scores equal fresh `search` scores for 10 random questions.
- **[ADA]** the full-corpus run is job A5 (section 5): write `ada/run_cache.sh`, request it, and record time and disk usage from the returned report.

### Step 8 -- Poison ingestion hook
Build: `scripts/index_poison.py` that adds every record from the poison shards (`poison_*.jsonl`, any attack) to each retriever's index (append for BM25/dense/SPLADE; rebuild if the library cannot append). Until Arya provides shards, use a **10-passage synthetic test shard** under `tests/fixtures/` for testing only, and say so in HANDOFF.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- After ingestion, a fixture poison passage written to match question X appears in X's top-10 for at least one retriever, and its `is_poison` flag is recoverable from `doc_id`.
- Index size before/after reported.
- Clean-question results change minimally (report fraction of the 400 questions whose top-5 changed).

### Step 9 -- Generation harness and cache
Build: `generation/llm.py` and `generation/cache.py`.
- Model from config (3-4B instruct, local via vLLM or transformers; temperature 0, max new tokens 32). Neutral prompt:
```
Answer the question using the passages below. Be brief.
Passages:
[1] <text 1>
...
[5] <text 5>
Question: <question>
Answer:
```
(no warning about injections). Passage text truncated to a fixed token budget from config.
- Cache: key = `sha1(qid + "|" + "|".join(top5_ids))`, stored in a SQLite or JSONL store; `generate(...)` returns `{"answer", "cached"}`.
- Batch generation for sweeps with GPU batching.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Same inputs twice: second call returns `cached=True` and identical text.
- Changing the order of the 5 IDs produces a different key (order matters).
- Greedy determinism: 20 prompts run twice give identical output.
- Throughput report (requests/min) on the available GPU; this feeds the 60,000-request budget in the plan.
- Show 3 sample answers on clean questions (EM against gold reported on 50 clean questions as a smoke metric).

### Step 10 -- Cluster filter (`defenses/cluster.py`)
Spec (plan section 4.2): take top-20 of the pool, vectors via `get_doc_vector`, k-means with k=2 (seeded), for each cluster with >=2 members compute mean pairwise cosine and mean pairwise ROUGE-L between member texts; flag the cluster only if cosine >= `tau_c` (default 0.85) **and** ROUGE-L >= 0.25. Return flagged IDs and the cluster stats in `scores`. `tau_c` is a config value so Vansh can sweep 0.70, 0.75, 0.80, 0.85, 0.90, 0.95. Also report backfill: whether the first 5 survivors come from beyond rank 20.
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Synthetic test: 5 near-duplicate passages + 15 diverse ones in a pool -> the 5 are flagged; 20 diverse passages -> nothing flagged.
- Test that a single-member cluster is never flagged.
- On clean dev questions (no poison), report the fraction of questions where anything is flagged and the mean number of clean passages flagged (false-positive rate).
- On the synthetic poison fixture, report detection rate.
- Runtime per question reported.

### Step 11 -- Perplexity filter and score cache (`defenses/perplexity.py`)
Spec (plan section 4.3): GPT-2 small; split each passage at its midpoint (by tokens); f(R) = mean negative log-likelihood per token; PD = f(pre) - f(post) (two-tailed), PM = max(f(pre), f(post)) (one-tailed). Cache `{doc_id: {ppl_pre, ppl_post, pd, pm}}` in a persistent store keyed by `doc_id`. Thresholds: percentile (alpha = 2.5% default; also 1/5/10 for sweep) of the PD/PM distribution over **1,000 random clean corpus passages** (seeded), saved to `configs/perplexity_thresholds.json`. A passage is flagged if PD is outside the two-tailed band or PM exceeds the one-tailed cutoff (document which rule is used as anchor in config; default: flag if either rule fires).
Verify ([LOCAL] first; the Ada check is listed in section 5):
- Score a known fluent passage vs the same passage with a gibberish second half: PD must differ sharply in the expected direction.
- Threshold calibration: on a held-out 1,000 clean passages, the flag rate is close to the configured alpha (report the number).
- Passages shorter than 20 tokens are handled (flag nothing, record `too_short`).
- Cache hit test: second call for the same doc_id does no GPU work.
- Report the mean passage length in tokens on NQ and how many passages are "too short".

### Step 12 -- Held-out phrasing tool
Build: `scripts/heldout_template.py` that creates `heldout/phrasings_template.jsonl` with 10 empty hijack-phrasing slots and 5 empty fact-passage slots for Samarth to fill **by hand without looking at Arya's frozen phrase list**. The script must not read the phrase list. Add a `scripts/validate_heldout.py` that checks format, fills `doc_id` as `poison_<qid>_heldout_<n>`, and later (when Arya's list exists) reports overlap with that list.
Verify ([LOCAL] first; the Ada check is listed in section 5): template creation works; validator rejects empty slots; validator passes on a filled fixture.

### Step 13 -- End-to-end smoke test
Build: `scripts/smoke_test.py`: for 20 questions and each of the 4 retrievers: load cached top-150 -> stub `candidate_pool` -> apply cluster filter and perplexity filter -> first 5 survivors -> `generate` -> print question, top-5 IDs, flagged IDs, answer.
Verify ([LOCAL] first; the Ada check is listed in section 5): the script finishes without error for all 4 retrievers and both filters; HANDOFF lists the exact command and expected output shape; update "What teammates can use right now" with every module and CLI.

## 4. When all steps are DONE
Write a final HANDOFF section "Ready for integration" listing: commands to rebuild each cache, module entry points for Vansh's runner, config keys Vansh needs for sweeps, known limitations (BM25 masking approximation; perplexity noise on short halves), and measured numbers (Recall@100 per retriever on the dev subset, generation throughput, perplexity cache size, filter runtimes). Include the table of Ada jobs with status (REQUESTED / RUNNING / PASSED / FAILED) and owner. Then stop and wait for instructions.

## 5. Ada requests (what teammates run)

After each step reaches DONE-LOCAL, Claude Code does three things: (a) makes sure the job script exists in `ada/`, (b) appends an entry to `docs/ADA_RUNBOOK.md` if it is new, (c) adds a row to "Ada requests" in `docs/HANDOFF.md` with status REQUESTED and the named owner.

| Job | After step | Script | What it produces | Suggested owner | Rough size |
|---|---|---|---|---|---|
| A0 setup | 0 | `ada/setup_env.sh` | Environment, models downloaded, `check_env` report | whoever starts first | 10-30 min |
| A1 data | 2 | `ada/run_prepare_data.sh` | `corpus.jsonl`, `questions.jsonl` (Arya's final set replaces the placeholder) | Arya | download + minutes |
| A2 bm25 | 3 | `ada/run_index.sh bm25` | BM25 index over clean + poison | Arya | CPU, <1 h |
| A3 dense | 4 | `ada/run_index.sh dense` | Dense vectors + FAISS index | Vansh | GPU, 1-3 h on full corpus |
| A4 splade | 5 | `ada/run_index.sh splade` | SPLADE index | Vansh | GPU, several hours on full corpus |
| A5 cache | 7 | `ada/run_cache.sh <retriever>` | `top150_*.jsonl` for all retrievers | Vansh | GPU, ~1 h per retriever |
| A6 poison | 8 | `ada/run_index_poison.sh` | Poison shards added to every index | Arya | minutes to 1 h |
| A7 generate | 9 | `ada/run_generate_benchmark.sh` | Throughput of the real 3-4B LLM, cache check | Vansh | 15 min |
| A8 perplexity | 11 | `ada/run_ppl.sh` | GPT-2 thresholds + score cache for pool docs | Vansh | GPU, <1 h |
| A9 smoke | 13 | `ada/run_smoke.sh` | End-to-end report with real models | Arya or Vansh | 30 min |

Each script prints a one-line summary and writes `$IRE_DATA_DIR/reports/<job>.json`. The teammate pastes that JSON (or commits it under `reports/`) and tells Samarth the job is done. Claude Code then runs `scripts/check_ada_report.py <job>` (build this in Step 1), which compares the report with the Ada acceptance checks below; only if it passes may the step go from DONE-LOCAL to DONE.

**Ada acceptance checks (per job)**
- A1: counts match the project plan (corpus size ~2.7M or the chosen subset; 200 target + 200 clean; every gold ID exists).
- A2-A4: index builds without error; Recall@100 on 1,000 questions is reported; a unique-token test document is found at rank 1.
- A5: row count equals number of questions; every row has 150 sorted entries; 10 random rows match a fresh search.
- A6: a poison passage written for question X appears in X's top-10 for at least one retriever; index sizes reported.
- A7: throughput in requests/min and projected hours for 60,000 requests; cache hit on repeat; greedy determinism on 20 prompts.
- A8: flag rate on 1,000 held-out clean passages close to alpha; cache size; fraction of passages marked too_short.
- A9: all four retrievers and both filters run end to end on 20 questions.

**Request message template** (Claude Code writes this into `docs/HANDOFF.md` and `docs/ADA_RUNBOOK.md` for each job):
```
ADA REQUEST <job>
Owner: <name>          Needs: <earlier jobs done>
Run:   git pull && bash ada/<script>.sh
Time:  <estimate>      Disk: <estimate>
Success looks like: <one sentence>
Send back: $IRE_DATA_DIR/reports/<job>.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of the log and the report file if one exists.
```

## 6. Working without waiting
Order of work for Samarth: finish DONE-LOCAL for steps 0-13 in sequence, raising Ada requests as soon as each step is ready. The Ada owners run jobs in parallel as they arrive. If an Ada result contradicts a local result (for example recall much lower than expected), mark that step REOPENED in `docs/HANDOFF.md` with the evidence and fix it before continuing.
