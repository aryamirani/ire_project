# 12 -- Code map: every file, what it does, why it matters

Owner in brackets: S = Samarth, A = Arya, V = Vansh. Line counts as of 2026-11-08 (~14,900 lines incl. tests).
If the examiner opens a file, know its first docstring line and its one key function.

## src/ire/ -- the package
| File | Owner | What / key function | Why it matters |
|---|---|---|---|
| `schemas.py` (344) | S | section-2 data contracts: `CorpusRecord`, `PoisonRecord`, `QuestionRecord`, `QuestionAllRecord`, `RetrievalCacheRecord`, `FilterContext`, `FilterResult`, `VectorAPI`, `GenerationOutput`; `read_jsonl/iter_jsonl/write_jsonl`; `is_poison_id`, `make_poison_id`, `parse_poison_id` | strict validation (all fields required, unknown fields rejected) so teammates' files fail loudly, with the line number |
| `ada_report.py` (614) | S (A8/A9: A) | acceptance rules per Ada job; `check_report(job, report)` -> PASS/FAIL/LOCAL_DRY_RUN | the gate from DONE-LOCAL to DONE |
| `stubs.py` | A | `candidate_pool` stub (first 50, no masking) for the smoke test | lets step 13 run before Vansh's masking existed |
| **config/** `env.py`, `loader.py` | S | `.env` loading, `get_env_name`, `get_data_dir` (refuses data inside the repo), `load_config` deep merge, `Config.path`, `Config.thresholds_path` | one source for every setting and path; no absolute paths |
| **data/** `download.py` | S | pinned downloads of BEIR NQ (HF or UKP) and nq_open, sha256 | reproducible raw inputs |
| `nq.py` (283) | S | stream the corpus (zip or parquet), load queries/qrels/nq_open, `normalize_question` (join key) | 2.68M passages never all in memory |
| `prepare.py` | S | build corpus / questions_all / questions (modes full, random, subset) and verify them | step 2 |
| `question_set.py` | A | eligibility, `answer_type`, stratified selection into target/clean x A/B, frozen file I/O | the 400 questions |
| `textstore.py` | S | SQLite doc_id -> (title, text) for corpus + shards; read-only Mapping | LLM prompts and filter texts; 2.68M passages on disk |
| **retrieval/** `common.py` | S | documents + shards iteration, `inputs_fingerprint`, `swap_in` (atomic), `doc_id_ranks`, `top_k_rows` (ties by doc_id), `chunk_bounds`, `resolve_model` | shared determinism and safety helpers |
| `bm25.py` (236) | S | `BM25Settings`, `build_index`, `BM25Retriever.search/score_docs`, `unique_token_test` | lexical retriever; empty-token fix |
| `dense.py` (477) | S | `E5Encoder` (prefixes, mean pooling, fp16, OOM halving, queries one at a time), resumable build, `ingest_poison`, `DenseRetriever` (exact Flat search with tie widening, lazy FAISS, vector API) | dense retriever + vectors for two filters |
| `splade.py` (465) | S | SPLADE encoder (max log1p ReLU), CSR + inverted index, exact search, `ingest_poison` | learned sparse retriever |
| `rrf.py` | S | `fuse`, `fuse_many` (fsum, ties by doc_id), `RRFRetriever` (no masking) | hybrid; mask-then-fuse rule |
| `cache.py` (271) | S | build/resume/validate/load top-150 caches; sidecar key; `StaleCacheError`, `UnmaskedReferenceError` | experiments never retrieve; stale data cannot be used |
| `poison.py` | S | `load_shards` (validation), `masking_equivalence`, `poison_reach`, `top_k_changed`, clean reference | A6 checks |
| `eval.py` | S | `recall_at_k`, eval-set sampling | retrieval sanity only |
| `pool.py` | V | `is_active`, `mask_row`, `masked_row` (RRF: mask then fuse), `candidate_pool` | the masking step of every experiment |
| **vectors/** `__init__.py` | S | `get_doc_vector`, `get_query_vector`, `default_vectors()` | the Vector API used by the cluster and query-phrase filters |
| **poison/** `targets.py` | A | type-matched false target answers (`choose_target`, `is_valid_target`) | fact/diverse/repeat targets |
| `templates.py` | A | all attack templates, codeword ZOLTRAX, URL | written after the phrase list was frozen |
| `build.py` | A | `build_main`, `build_adaptive`, seeded template order, stealth candidates | 4,500 passages |
| `heldout.py` | A | template, `validate`, `build_heldout`, `phrase_overlap` (only after the shard is final) | the blind test |
| **defenses/** `query_phrase.py` | A | similarity + frozen phrase rule, `calibrate_cutoffs`, `thresholds_for` (opposite fold), `load_thresholds` (dense-model guard) | detector 1 |
| `cluster.py` | A (S lane) | k-means k=2 on top-20 vectors, ROUGE-L via LCS, flag if cosine >= tau_c and ROUGE-L >= 0.25 | detector 2 |
| `perplexity.py` (321) | A (S lane, S fixes) | `PerplexityScorer` (GPT-2 halves), `PerplexityCache` (SQLite), `calibrate`, `flags`, namespace guard | detector 3 |
| `controls.py` | V | `random_filter` (size-matched, keyed seed), `oracle_filter` | RQ4 baselines |
| **generation/** `llm.py` (266) | S | `GenerationSettings`, `Generator` (render via chat template, truncation, greedy, length-sorted left-padded batches, OOM halving, cache), `generate` | the fixed LLM |
| `cache.py` | S | `GenerationCache` (SQLite WAL), `cache_key`, `namespace_id` | ~60,000 requests become far fewer |
| `smoke_metrics.py` | S | SQuAD EM/F1 for the step-9 smoke check | not the project metrics |
| **eval/** `metrics.py` | V | ASR (literal vs token match), EM, F1, acc, nDCG, Recall, gold dropped, backfill | per-question metrics |
| `stats.py` | V | Wilson, exact McNemar, paired bootstrap, Holm, ROC-AUC, question-level AUC bootstrap, Cohen's kappa | inference |
| `sweep.py` (310) | V | `Setting`, `expand` (grid), `load_targets`, `retrieval_part`, `answer_part`, `SweepRunner` | runs the experiment |
| `analysis.py` | V | `summarize`, `paired`, `headline` (Holm, helps, beats_random, mechanism flag), `cross_fit`, `mechanism`, `complementarity` | turns records into answers to RQ1-RQ5 |
| **utils/** `log.py`, `run_info.py` | S | env-tagged logger; git commit + dirty flag | traceability |

## scripts/ -- one command per stage
| Script | Owner | Job | Purpose |
|---|---|---|---|
| `check_env.py` | S | A0 | Python/torch/GPU/disk/libraries/models report |
| `download_models.py` | S | | models into HF_HOME (resumable) |
| `prepare_data.py` | S | A1 | data files + data card |
| `select_questions.py`, `build_local_corpus.py` | A | | apply the frozen 400; rebuild the local corpus around them |
| `build_index.py` (546) | S | A2-A4 | build + all checks (recall, unique token / self-retrieval, exactness, determinism, latency) |
| `check_rrf.py` | S | | step-6 checks |
| `build_cache.py` | S | A5 | top-150 caches + spot checks |
| `make_poison.py` | A | P1 | poison shards + report (stealth cap needs the dense index) |
| `index_poison.py` (425) | S | A6 | ingestion, masking check, reach; `--work-copy`, `--check-masking` |
| `generate_benchmark.py` | S | A7 | cache/determinism/batch-independence checks, throughput, smoke EM/F1 |
| `check_cluster.py` | A | | step-10 sanity on real caches |
| `calibrate_perplexity.py` | A (S fixes) | A8 | thresholds, held-out rate, sanity, score cache; `--write-config` |
| `calibrate_query_phrase.py` | A (S fix) | P2 | cutoffs per retriever and fold, phrase FPR, detection sanity; `--write-config` |
| `heldout_template.py`, `validate_heldout.py` | A | | step 12 |
| `smoke_test.py` | A | A9 | 20 questions x 4 retrievers x 3 filters -> LLM |
| `run_sweep.py` | V | sweep | `--list`, `--tier`, `--pilot`, resumable |
| `analyze.py`, `make_plots.py` | V | | CSV tables, figures |
| `calibration_sample.py`, `calibration_agreement.py` | V | | human-label sheets, kappa and judge accuracy |
| `check_ada_report.py` | S | | runs the acceptance rules on a report |

## ada/ -- cluster wrappers (bash)
`common.sh` (SLURM defaults, env activation), `setup_env.sh` (A0), `run_prepare_data.sh`, `run_index.sh`,
`run_make_poison.sh`, `run_index_poison.sh`, `run_cache.sh`, `run_query_phrase.sh`, `run_ppl.sh`,
`run_generate_benchmark.sh`, `run_smoke.sh`, `run_sweep.sh`, and `pinned_node.sh` (run any job on gnode056 with data on
its local /scratch; stage in, run, stage out).

## tests/ -- 435 test cases (433 on 2026-11-08, +2 on 2026-11-10; 27 files)
The numbers below count test *functions*; parametrisation turns them into 433 cases (435 with the two added on 2026-11-10). Biggest: `test_ada_report.py` (29 rule cases), `test_bm25.py` (17: tokenisation, empty-token, ties, unique token,
manifest checks), `test_splade.py` (16: pooling formula equivalence, exactness, empty query), `test_dense.py` (14:
prefixes, normalisation, resumable build, append == rebuild), `test_config.py` (15: merging, paths, errors, threshold
paths), `test_schemas.py` (12, many parametrised), `test_data_nq.py` (14), `test_poison.py` (10), `test_rrf.py` (10:
formula, ties, mask-then-fuse toy case), `test_cache.py` (9: stale key, unmasked reference, resume),
`test_generation.py` (8: exact prompt, truncation, namespace, OOM halving, killed run, real-model determinism),
`test_cluster.py` (7), `test_perplexity.py` (7), `test_query_phrase.py` (7), Vansh's `test_pool/metrics/stats/
controls/sweep/analysis` (28 functions), others. Models are only needed by a few marked tests.

## configs/, docs/, reports/, heldout/
- `configs/`: experiment, env_local, env_ada, paths, sweep, the two threshold tables (Ada values), `frozen/`.
- `docs/`: the plan PDF, `IMPLEMENTATION_PLAN.md` (Samarth's step list), `HANDOFF.md` (step log, interface changes,
  open issues), `HANDOFF_ARYA.md`, `ADA_RUNBOOK.md`, `VANSH_RUNBOOK.md`, `DECISION_RULE.md`, `mid_report/`.
- `reports/`: committed Ada reports (A0-A6, A8, A9, poison, questions, query_phrase, data cards).
- `heldout/`: instructions for the blind writer, the template, the filled `phrasings.jsonl`.
