# VANSH_START -- where your lane starts and how to continue (2026-10-26)

Your lane (plan section 9): **`candidate_pool` masking; metrics including detection quality; random and oracle
controls; cross-fitting and statistics; judge calibration; sweep runner; plots.** Plus the Ada runs A7 and A8.
The order across the team is in `TEAM_STATUS.md`; this file is only what you need to start.

## 1. What already exists for you
| What | Where | Notes |
|---|---|---|
| Final 400 questions | `data/questions.jsonl` (`QuestionRecord`: qid, question, answers, gold_doc_ids, role, fold) | 200 target + 200 clean, folds A/B with 100 + 100 each. Answer types per qid in `configs/frozen/question_set.json`. |
| Poison shards | `data/poison/poison_main.jsonl` (fact, hijack, url) and `poison_adaptive.jsonl` (diverse, repeat, stealth) | `PoisonRecord`; doc_id `poison_<qid>_<attack>_<n>`, n = 1..5. Only target questions are poisoned. Held-out shard `poison_heldout.jsonl` (attack `heldout`, 505 passages, written blind) is ingested too (A6 re-run 2026-10-30). |
| ASR target strings | each record's `target_answer`; per qid in `configs/frozen/poison_targets.json` | fact / diverse / repeat: a false answer; hijack / stealth: codeword `ZOLTRAX`; url: domain `answer-archive.example`. |
| Top-150 caches over the poisoned indexes | `caches/ada/top150_{bm25,dense,splade,rrf}.jsonl` | Built on the final questions, after poison ingestion (A5 passed). The `rrf` file is an unmasked reference only (see section 3). |
| Text of any doc_id | `ire.data.textstore.TextStore.open_for(cfg)` | Read-only mapping doc_id -> "title text"; use as `FilterContext(texts=...)`. |
| Dense vectors | `ire.vectors.default_vectors()` | `get_doc_vector`, `get_query_vector`; use as `FilterContext(vectors=...)`. |
| Query-and-phrase filter (Arya) | `ire.defenses.query_phrase` | Ready; thresholds calibrated on Ada in `configs/query_phrase_thresholds.json`. |
| Cluster / perplexity filters (Samarth) | `ire.defenses.cluster`, `ire.defenses.perplexity` | Ready (steps 10-11, done by Arya); perplexity thresholds from A8 in `configs/perplexity_thresholds.json`. |
| Generation | `ire.generation.llm.Generator.from_config(cfg)` | `generate(qid, question, top5_ids)` / `generate_batch([(qid, question, top5_ids), ...])` -> `{"answer", "cached"}`; answers cached in `caches/generation.sqlite`. |
| Report checker | `python scripts/check_ada_report.py A7 --file ...` | Rules exist for A0-A9. |

## 2. Getting a working setup
**On your laptop (develop and test here first; local numbers are for correctness only):**
```
git pull
python -m venv .venv  &&  pip install torch==2.6.0 -r requirements.txt  &&  pip install -e . --no-deps
cp .env.example .env            # IRE_ENV=local, IRE_DATA_DIR=<a folder with ~10 GB free>, HF_HOME=<that>/hf_cache
python scripts/download_models.py
python scripts/prepare_data.py                      # downloads NQ; builds the final questions + a 5,000-passage corpus
python scripts/build_index.py --retriever bm25      # then dense, then splade
python scripts/make_poison.py
python scripts/index_poison.py
python scripts/build_cache.py --retriever all
```
On macOS put `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` in front of every command (faiss + torch can
deadlock otherwise). After this you have the same file layout as Ada, just small.

**On Ada (Arya's account; everything big is on node gnode056):**
- Code: `~/ire_project`. To get your latest code there, `git pull` with your own GitHub HTTPS token (never
  write the token into a file in the repo), or ask Arya to sync it.
- Jobs: `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t <time> -o ~/ire_data_dir/logs/<name>-%j.log --wrap "bash \$HOME/ire_project/ada/pinned_node.sh <script> <args>"`.
  `pinned_node.sh` copies the code to `~/ire_node` (whose `.env` points at gnode056's `/scratch`), runs the
  script there, and copies reports, logs, caches and poison back to `~/ire_data_dir`.
- Interactive work on the node:
  `srun -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 04:00:00 --pty bash`, then
  `cd ~/ire_node && export PATH=$HOME/miniforge3/bin:$PATH && source ada/common.sh && ire_activate`.
  (`~/ire_node` holds the code of the last job; run any job once, or `rsync -a --exclude .env ~/ire_project/ ~/ire_node/`, to refresh it.)
- Limits: 10 CPUs, 1 GPU, 29 GB RAM, at most 5 queued + running jobs. `/share1` is not visible on compute nodes.
- `load_cache` checks the caches against the index manifests, which exist only on gnode056, so anything that
  reads caches must run there.

## 3. Your first task: `candidate_pool` (plan section 2.3)
Contract (docs/IMPLEMENTATION_PLAN.md section 2): `candidate_pool(qid, retriever, attack, rate, cache, pool_size=50)`
-> ranked list of doc_ids after masking. Suggested home: `src/ire/retrieval/pool.py` (plan layout: retrieval/).
```python
from ire.retrieval.cache import load_cache
from ire.retrieval.rrf import fuse
from ire.schemas import is_poison_id, parse_poison_id

cache = load_cache(["bm25", "dense", "splade"])          # {retriever: {qid: [(doc_id, score), ...150]}}

def is_active(doc_id, attack, rate):
    _, a, n = parse_poison_id(doc_id)
    return a == attack and n <= rate                     # rate r keeps poison_<qid>_<attack>_1..r

def masked(row, attack, rate):
    return [(d, s) for d, s in row if not is_poison_id(d) or is_active(d, attack, rate)]
```
- `attack=None` (no-poison setting) masks every poison passage.
- **RRF**: mask the bm25 and dense lists first, then `fuse(bm25_masked, dense_masked, k=cfg["rrf.k"],
  depth=cfg["rrf.depth"])`. Never mask the cached `top150_rrf.jsonl` (docs/HANDOFF.md, step 6 explains why).
- BM25 masking is not exactly a separate index; A6 measured it: mean rank shift 0.40, top-5 unchanged for
  96% of questions (`reports/A6.json`). Report this in the paper.
- Should poison of OTHER target questions stay active in a setting? With `is_active` as above it does (same
  attack, any qid), which is what produces "collateral poisoning" on clean questions. Decide and write it down.

## 4. Filters (all have the same API)
```python
from ire.schemas import FilterContext
from ire.defenses.query_phrase import apply_filter, load_thresholds, thresholds_for
table = load_thresholds(cfg.path("query_phrase_thresholds"))
ctx = FilterContext(texts=store, vectors=vectors, config=cfg,
                    thresholds=thresholds_for(table, retriever, q.fold, tail=0.025))   # cutoff from the OTHER fold
res = apply_filter(pool, q.qid, q.question, ctx)
top5 = [d for d in pool if d not in res.flagged][:5]                                   # first 5 survivors
```
`res.scores[doc_id]` is the continuous score for ROC-AUC (similarity for query-phrase). Sweep tails:
`cfg["query_phrase.tail_sweep"]`; ablations: `cfg["query_phrase.rule"]` = `either` | `similarity` | `phrase`.
Random control: drop the same number of passages as the detector, chosen with a seeded RNG. Oracle: drop every
`is_poison_id`.

## 5. What a run produces (plan section 8, "Run record")
`{"qid", "retriever", "attack", "rate", "filter", "fold", "pool", "flagged", "top5", "n_poison_top5",
"n_gold_dropped", "answer", "asr", "em", "f1"}` -- one per (setting, question). Metrics to compute from them:
ASR, EM / F1 (SQuAD normalisation; `ire.generation.smoke_metrics` has a smoke version), nDCG@5 / @10 and
Recall@k against `gold_doc_ids`, collateral poisoning (poison in clean questions' top-5), detection (ROC-AUC,
TPR on poison, FPR on clean passages, gold passages dropped), backfill rate (survivors from beyond rank 20),
latency / filter overhead.

## 6. Ada jobs that are yours
| Job | When | Command (inside the sbatch / pinned_node pattern above) | Check |
|---|---|---|---|
| A7 LLM benchmark | now | `ada/run_generate_benchmark.sh` | `check_ada_report.py A7`; then decide the generation batch size (docs/HANDOFF.md "Decision needed") |
| A8 perplexity scores | after Samarth's step 11 | `ada/run_ppl.sh` (Samarth writes it) | `check_ada_report.py A8` |
| Pilot (500 requests) | after your runner + A7 | your runner | throughput, cache hit rate, failure modes (plan 6.1) |
| Sweeps | after the pilot and after the held-out shard is ingested | your runner | |

Qwen2.5-3B-Instruct is already downloaded to `~/ire_data_dir/hf_cache` on Ada (6.2 GB fits the 2080 Ti's 11 GB).

## 7. Before the sweeps (do not skip)
1. Samarth's held-out shard must be ingested first (Arya re-runs A6 -> A5 -> P2); caches built before that are
   stale and `load_cache` will refuse them (`StaleCacheError`).
2. Write the decision rule down before running (plan 5.5): headline anchor, what "helps" means (ASR interval
   excludes zero and clean nDCG@5 loss under a stated bound), what we say if the answer is no.
3. Thresholds and cutoffs: choose on fold A, report on B, swap, pool (plan 5.2).
