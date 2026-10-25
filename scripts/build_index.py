"""Steps 3-5 / jobs A2-A4: build a retriever index, then check it and write a report.

Usage (repo root; PowerShell or bash):
    python scripts/build_index.py --retriever bm25                   # build if needed, then evaluate
    python scripts/build_index.py --retriever dense                  # E5 vectors + FAISS (GPU if available)
    python scripts/build_index.py --retriever splade                 # SPLADE sparse vectors, exact search (GPU)
    python scripts/build_index.py --retriever bm25 --force           # rebuild even if the index is current
    python scripts/build_index.py --retriever dense --check-rebuild  # also rebuild a temp copy and compare results

The index goes to $IRE_DATA_DIR/indexes/<env>/<retriever>. Resumable: if the index exists and was built from the
same corpus (sha256), poison shards and settings (dense/SPLADE: also the exact model revision), the build is
skipped and only the checks run; an interrupted dense/SPLADE encoding resumes from its last checkpoint.
Report: $IRE_DATA_DIR/reports/<A2|A3|A4>[_local].json with sizes, timings, Recall@10/100 and the retriever's sanity
checks. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.retrieval import bm25 as bm25_mod  # noqa: E402
from ire.retrieval.common import dir_size, find_poison_shards, index_dir_for, iter_documents, read_manifest  # noqa: E402
from ire.retrieval.eval import questions_with_gold_in_index, recall_at_k, sample_eval_set  # noqa: E402
from ire.schemas import QuestionAllRecord, QuestionRecord, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402

JOB_NAMES = {"bm25": "A2", "dense": "A3", "splade": "A4"}
EDGE_QUERIES = ["", "the of and is", "qwertzuiopasdfghjk"]  # empty, only stopwords, unknown word

Item = tuple[str, str, list[str]]  # (qid, question, gold ids in the index)


# ---------------------------------------------------------------- shared checks


def evaluate(search: Callable[[list[str], int], list], items: list[Item], ks: list[int]) -> tuple[dict, list]:
    """Recall/hit@k for (qid, question, gold) items; returns metrics (+ n_questions) and the raw results."""
    if not items:
        return {"n_questions": 0}, []
    t0 = time.perf_counter()
    results = search([q for _, q, _ in items], max(ks))
    elapsed = time.perf_counter() - t0
    metrics = {"n_questions": len(items), **recall_at_k(results, [g for _, _, g in items], ks),
               "batch_ms_per_query": round(1000 * elapsed / len(items), 3)}
    return metrics, results


def eval_sets(cfg: Any, ids_in_index: set[str]) -> tuple[list[Item], list[Item], int]:
    """The 400 placeholder questions and a seeded sample of questions_all (both: gold present in the index)."""
    placeholder = questions_with_gold_in_index(read_jsonl(cfg.path("questions"), QuestionRecord), ids_in_index)
    pool_all = questions_with_gold_in_index(read_jsonl(cfg.path("questions_all"), QuestionAllRecord), ids_in_index)
    pool = sample_eval_set(pool_all, int(cfg["retrieval_eval.n_questions"]), int(cfg["seed"]))
    return placeholder, pool, len(pool_all)


def common_checks(search: Callable[[list[str], int], list], pool: list[Item]) -> dict[str, Any]:
    """Single-query latency, same-query-twice determinism and edge queries (must not crash)."""
    singles = []
    for _, q, _ in pool[:50]:
        t0 = time.perf_counter()
        search([q], 100)
        singles.append(1000 * (time.perf_counter() - t0))
    sample_q = [q for _, q, _ in pool[:50]]
    same_twice = search(sample_q, 100) == search(sample_q, 100)
    edge = {}
    for q in EDGE_QUERIES:
        try:
            edge[repr(q)] = {"hits": len(search([q], 10)[0]), "error": None}
        except Exception as exc:  # noqa: BLE001 - recorded as a failed check
            edge[repr(q)] = {"hits": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"single_query_median_ms": round(statistics.median(singles), 3) if singles else None,
            "same_twice": same_twice, "edge": edge}


def corpus_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


# ---------------------------------------------------------------- BM25 (step 3)


def run_bm25(args: argparse.Namespace, cfg: Any, log: Any) -> dict[str, Any]:
    data_dir = cfg.path("data_dir")
    settings = bm25_mod.BM25Settings.from_config(cfg)
    corpus = cfg.path("corpus")
    use_shards = cfg["bm25.include_poison_shards"] and not args.no_poison_shards
    shards = find_poison_shards(cfg.path("poison_dir")) if use_shards else []
    index_dir = index_dir_for(cfg, "bm25")
    log.info("bm25 env=%s corpus=%s poison shards=%d (%s) index=%s", cfg.env, corpus.name, len(shards),
             ", ".join(s.name for s in shards) or "none", index_dir)

    if not args.force and bm25_mod.index_is_current(index_dir, corpus, shards, settings):
        log.info("index is current (same corpus sha256, shards and settings): skipping the build (--force rebuilds)")
        built = False
    else:
        manifest = bm25_mod.build_index(corpus, index_dir, shards, settings)
        log.info("built: %d docs (%d poison) in %.1f s", manifest["n_docs"], manifest["n_poison"],
                 manifest["timings_s"]["total"])
        built = True
    manifest = read_manifest(index_dir)
    index_bytes = dir_size(index_dir)

    t0 = time.perf_counter()
    retriever = bm25_mod.BM25Retriever(index_dir, settings)
    load_s = time.perf_counter() - t0
    rows = corpus_rows(corpus)

    ks = [int(k) for k in cfg["retrieval_eval.ks"]]
    placeholder, pool, n_available = eval_sets(cfg, set(retriever.doc_ids))
    eval_placeholder, _ = evaluate(retriever.search, placeholder, ks)
    eval_pool, pool_results = evaluate(retriever.search, pool, ks)
    eval_pool["n_available"] = n_available
    log.info("recall placeholder: %s | pool: %s", eval_placeholder, eval_pool)
    common = common_checks(retriever.search, pool)
    edge_ok = all(v["error"] is None and v["hits"] == 0 for v in common["edge"].values())

    tmp_root = data_dir / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    rebuild_identical = None
    if args.check_rebuild:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            bm25_mod.build_index(corpus, Path(tmp) / "bm25_rebuild", shards, settings)
            again = bm25_mod.BM25Retriever(Path(tmp) / "bm25_rebuild", settings, mmap=False).search(
                [q for _, q, _ in pool], k=max(ks))
            rebuild_identical = again == pool_results
            del again
        log.info("rebuild gives identical top-%d results for %d queries: %s", max(ks), len(pool), rebuild_identical)

    cap = int(cfg["bm25.unique_token_test_passages"])
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        unique = bm25_mod.unique_token_test(corpus, Path(tmp), settings, max_docs=cap if rows > cap else None)
    log.info("unique-token test: %s", unique)

    checks = {
        "n_docs_matches_corpus_plus_poison": manifest["n_docs"] == rows + manifest["n_poison"],
        "recall_reported": "recall@100" in eval_pool,
        "same_query_twice_identical": common["same_twice"],
        "edge_queries_no_crash_no_hits": edge_ok,
        "unique_token_rank_1": unique["passed"],
    }
    if rebuild_identical is not None:
        checks["rebuild_identical"] = rebuild_identical
    return {
        "retriever": "bm25", "checks": checks, "built_now": built, "index_dir": index_dir.relative_to(data_dir).as_posix(),
        "build_time_s": manifest["timings_s"]["total"], "build_timings_s": manifest["timings_s"],
        "index_bytes": index_bytes, "index_mib": round(index_bytes / 2**20, 2), "load_time_s": round(load_s, 3),
        "corpus": {"path": corpus.relative_to(data_dir).as_posix(), "rows": rows,
                   "sha256": manifest["fingerprint"]["corpus_sha256"]},
        "n_docs": manifest["n_docs"], "n_clean": manifest["n_clean"], "n_poison": manifest["n_poison"],
        "n_vocab": manifest["n_vocab"], "poison_shards": [s.name for s in shards], "settings": manifest["settings"],
        "eval": {"ks": ks, "placeholder_questions": eval_placeholder, "questions_all_sample": eval_pool},
        "latency_ms": {"single_query_median_ms": common["single_query_median_ms"],
                       "batch_ms_per_query": eval_pool.get("batch_ms_per_query")},
        "determinism": {"same_query_twice": common["same_twice"], "rebuild_identical": rebuild_identical},
        "edge_queries": common["edge"], "unique_token": unique,
        "libraries": library_versions("bm25s", "PyStemmer", "numpy", "scipy"),
        "_summary": f"{manifest['n_docs']:,} docs ({manifest['n_poison']} poison), vocab {manifest['n_vocab']:,}",
    }


# ---------------------------------------------------------------- dense (step 4)


def run_dense(args: argparse.Namespace, cfg: Any, log: Any) -> dict[str, Any]:
    from ire.retrieval import dense as dense_mod

    data_dir = cfg.path("data_dir")
    settings = dense_mod.DenseSettings.from_config(cfg)
    corpus = cfg.path("corpus")
    use_shards = cfg["dense.include_poison_shards"] and not args.no_poison_shards
    shards = find_poison_shards(cfg.path("poison_dir")) if use_shards else []
    index_dir = index_dir_for(cfg, "dense")
    seed = int(cfg["seed"])

    t0 = time.perf_counter()
    encoder = dense_mod.E5Encoder.from_config(cfg, settings)
    model_load_s = time.perf_counter() - t0
    log.info("dense env=%s model=%s@%s device=%s fp16=%s batch=%d index=%s (%s) poison shards=%d", cfg.env,
             settings.model, encoder.revision[:12], encoder.device, encoder.fp16, encoder.batch_size,
             index_dir, settings.index_type, len(shards))

    encoder.reset_peak_vram()
    if not args.force and dense_mod.index_is_current(index_dir, corpus, shards, settings, encoder.revision):
        log.info("index is current (same corpus sha256, shards, settings and model revision): skipping the build")
        built, build_vram = False, None
    else:
        manifest = dense_mod.build_dense_index(corpus, index_dir, encoder, settings, shards,
                                               int(cfg["dense.checkpoint_every"]), log)
        build_vram = encoder.peak_vram_gib()
        log.info("built: %d docs (%d poison) in %.1f s (encode %.1f s, %.0f passages/s), peak VRAM %s GiB",
                 manifest["n_docs"], manifest["n_poison"], manifest["timings_s"]["total"],
                 manifest["timings_s"]["encode"], manifest["encode_passages_per_s"] or 0, build_vram)
        built = True
    manifest = read_manifest(index_dir)
    index_bytes = dir_size(index_dir)

    t0 = time.perf_counter()
    hnsw = cfg["compute"].get("hnsw") or {}
    retriever = dense_mod.DenseRetriever(index_dir, encoder=encoder, ef_search=hnsw.get("ef_search"))
    load_s = time.perf_counter() - t0
    rows = corpus_rows(corpus)

    # ---- recall (+ BM25 on the same questions, if its index exists) ----
    ks = [int(k) for k in cfg["retrieval_eval.ks"]]
    placeholder, pool, n_available = eval_sets(cfg, set(retriever.doc_ids))
    eval_placeholder, _ = evaluate(retriever.search, placeholder, ks)
    eval_pool, pool_results = evaluate(retriever.search, pool, ks)
    eval_pool["n_available"] = n_available
    bm25_cmp = None
    bm25_dir = index_dir_for(cfg, "bm25")
    if read_manifest(bm25_dir):
        bm25 = bm25_mod.BM25Retriever(bm25_dir, bm25_mod.BM25Settings.from_config(cfg))
        bm25_cmp = {"placeholder_questions": evaluate(bm25.search, placeholder, ks)[0],
                    "questions_all_sample": evaluate(bm25.search, pool, ks)[0]}
    log.info("recall placeholder: %s | pool: %s | bm25: %s", eval_placeholder, eval_pool, bm25_cmp)
    common = common_checks(retriever.search, pool)
    edge_ok = all(v["error"] is None for v in common["edge"].values())  # dense always returns neighbours

    # ---- vector checks ----
    rng = random.Random(seed)
    n_check = min(int(cfg["dense.reencode_check_passages"]), retriever.n_docs)
    check_ids = rng.sample(retriever.doc_ids, n_check)
    n_self = min(int(cfg["dense.self_retrieval_check_passages"]), retriever.n_docs)
    self_ids = rng.sample(retriever.doc_ids, n_self)
    wanted = set(check_ids) | set(self_ids)
    texts = {d: t for d, t in iter_documents(corpus, shards) if d in wanted}

    fresh = encoder.encode_passages([texts[d] for d in reversed(check_ids)])[::-1]  # different batch order
    cos = np.sum(retriever.get_doc_vector(check_ids) * fresh, axis=1)
    reencode = {"n": n_check, "min_cosine": round(float(cos.min()), 6), "mean_cosine": round(float(cos.mean()), 6),
                "passed": bool(cos.min() > 0.999)}
    norms = manifest["norms"]
    norms_ok = abs(norms["min"] - 1) <= 1e-3 and abs(norms["max"] - 1) <= 1e-3

    sample_p = [texts[d] for d in check_ids[:5]]
    sample_q = [q for _, q, _ in pool[:5]]
    pairs = [(encoder.encode_passages(sample_p), encoder._encode(sample_p)),
             (encoder.encode_queries(sample_q), encoder._encode(sample_q))]
    # "differ" = every vector changes measurably (largest element change > 1e-3); cosines are reported as well
    prefix = {"passage_cos_with_vs_without_prefix": [round(float(x), 4) for x in np.sum(pairs[0][0] * pairs[0][1], 1)],
              "query_cos_with_vs_without_prefix": [round(float(x), 4) for x in np.sum(pairs[1][0] * pairs[1][1], 1)],
              "min_max_abs_change": round(float(min(np.abs(w - wo).max(axis=1).min() for w, wo in pairs)), 4),
              "differ": bool(all(np.all(np.abs(w - wo).max(axis=1) > 1e-3) for w, wo in pairs))}

    top1 = retriever.search([texts[d] for d in self_ids], k=1)
    self_rate = sum(1 for d, hits in zip(self_ids, top1) if hits and hits[0][0] == d) / max(1, n_self)
    self_retrieval = {"n": n_self, "top1_rate": round(self_rate, 4), "passed": self_rate >= 0.9}

    rebuild_identical = None
    if args.check_rebuild:
        tmp_root = data_dir / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            dense_mod.build_dense_index(corpus, Path(tmp) / "dense_rebuild", encoder, settings, shards,
                                        int(cfg["dense.checkpoint_every"]))
            again = dense_mod.DenseRetriever(Path(tmp) / "dense_rebuild", encoder=encoder, mmap=False,
                                             ef_search=hnsw.get("ef_search"))
            same_vectors = bool(np.array_equal(np.asarray(again.vectors), np.asarray(retriever.vectors)))
            rebuild_identical = same_vectors and again.search([q for _, q, _ in pool], max(ks)) == pool_results
            del again
        log.info("rebuild: identical vectors and top-%d results: %s", max(ks), rebuild_identical)

    peak = encoder.peak_vram_gib()
    cap = cfg["compute"].get("max_vram_gb")
    checks = {
        "n_docs_matches_corpus_plus_poison": manifest["n_docs"] == rows + manifest["n_poison"],
        "recall_reported": "recall@100" in eval_pool,
        "same_query_twice_identical": common["same_twice"],
        "edge_queries_no_crash": edge_ok,
        "prefix_changes_vectors": prefix["differ"],
        "stored_matches_reencoded_cos_gt_0.999": reencode["passed"],
        "norms_within_1e-3": norms_ok,
        "self_retrieval_top1_ge_0.9": self_retrieval["passed"],
    }
    if cap and peak is not None:
        checks["vram_peak_under_cap"] = peak <= cap
    if rebuild_identical is not None:
        checks["rebuild_identical"] = rebuild_identical
    return {
        "retriever": "dense", "checks": checks, "built_now": built, "index_dir": index_dir.relative_to(data_dir).as_posix(),
        "model": settings.model, "model_revision": encoder.revision, "dim": manifest["dim"],
        "device": encoder.device, "fp16": encoder.fp16, "encode_batch_size_final": encoder.batch_size,
        "build_time_s": manifest["timings_s"]["total"], "build_timings_s": manifest["timings_s"],
        "encode_passages_per_s": manifest["encode_passages_per_s"], "resumed_from_row": manifest["resumed_from_row"],
        "vram_peak_gib": {"build": build_vram, "overall": peak, "cap": cap},
        "index_bytes": index_bytes, "index_mib": round(index_bytes / 2**20, 2),
        "load_time_s": round(load_s, 3), "model_load_time_s": round(model_load_s, 2),
        "corpus": {"path": corpus.relative_to(data_dir).as_posix(), "rows": rows,
                   "sha256": manifest["fingerprint"]["corpus_sha256"]},
        "n_docs": manifest["n_docs"], "n_clean": manifest["n_clean"], "n_poison": manifest["n_poison"],
        "poison_shards": [s.name for s in shards], "settings": manifest["settings"],
        "eval": {"ks": ks, "placeholder_questions": eval_placeholder, "questions_all_sample": eval_pool,
                 "bm25_same_questions": bm25_cmp},
        "latency_ms": {"single_query_median_ms": common["single_query_median_ms"],
                       "batch_ms_per_query": eval_pool.get("batch_ms_per_query"),
                       "note": "includes encoding the query"},
        "determinism": {"same_query_twice": common["same_twice"], "rebuild_identical": rebuild_identical},
        "edge_queries": common["edge"], "prefix_test": prefix, "reencode_check": reencode,
        "norms": {**norms, "passed": norms_ok}, "self_retrieval": self_retrieval,
        "libraries": library_versions("torch", "transformers", "faiss-cpu", "numpy", "huggingface_hub"),
        "_summary": f"{manifest['n_docs']:,} docs ({manifest['n_poison']} poison), {settings.model} dim "
                    f"{manifest['dim']}, {settings.index_type}",
    }


# ---------------------------------------------------------------- SPLADE (step 5)

FULL_CORPUS_PASSAGES = 2_681_468  # BEIR NQ; used to scale the local sparsity into an Ada RAM/disk estimate


def splade_scale_estimate(nnz_per_doc: float, n_docs: int = FULL_CORPUS_PASSAGES) -> dict[str, Any]:
    """Disk/RAM for the full corpus from the measured mean non-zeros per passage (8 B per non-zero per copy:
    float32 weight + int32 column; the index stores the matrix twice: rows = docs and the inverted transpose)."""
    nnz = nnz_per_doc * n_docs
    one_copy = nnz * 8 + (n_docs + 1) * 4
    return {"n_docs": n_docs, "nnz_per_doc": round(nnz_per_doc, 1), "nnz_total": int(nnz),
            "index_disk_gib": round(2 * one_copy / 2**30, 2),
            "build_peak_ram_gib": round(3.5 * one_copy / 2**30 + 1.5, 1),  # chunks + matrix + transpose + texts
            "note": "estimate scaled from the local corpus, not measured on Ada"}


def run_splade(args: argparse.Namespace, cfg: Any, log: Any) -> dict[str, Any]:
    from ire.retrieval import splade as splade_mod

    data_dir = cfg.path("data_dir")
    corpus = cfg.path("corpus")
    use_shards = cfg["splade.include_poison_shards"] and not args.no_poison_shards
    shards = find_poison_shards(cfg.path("poison_dir")) if use_shards else []
    index_dir = index_dir_for(cfg, "splade")
    seed = int(cfg["seed"])

    t0 = time.perf_counter()
    encoder = splade_mod.SpladeEncoder.from_config(cfg)
    model_load_s = time.perf_counter() - t0
    settings = encoder.settings
    log.info("splade env=%s model=%s@%s device=%s %s batch=%d doc/query max length %d/%d index=%s poison shards=%d",
             cfg.env, settings.model, encoder.revision[:12], encoder.device, settings.encode_dtype,
             encoder.batch_size, settings.doc_max_length, settings.query_max_length, index_dir, len(shards))

    encoder.reset_peak_vram()
    if not args.force and splade_mod.index_is_current(index_dir, corpus, shards, settings, encoder.revision):
        log.info("index is current (same corpus sha256, shards, settings and model revision): skipping the build")
        built, build_vram = False, None
    else:
        manifest = splade_mod.build_splade_index(corpus, index_dir, encoder, settings, shards,
                                                 int(cfg["splade.checkpoint_every"]), log)
        build_vram = encoder.peak_vram_gib()
        log.info("built: %d docs (%d poison) in %.1f s (encode %.1f s, %.0f passages/s), nnz/doc %.1f, peak VRAM %s GiB",
                 manifest["n_docs"], manifest["n_poison"], manifest["timings_s"]["total"],
                 manifest["timings_s"]["encode"], manifest["encode_passages_per_s"] or 0,
                 manifest["nnz_per_doc"]["mean"], build_vram)
        built = True
    manifest = read_manifest(index_dir)
    index_bytes = dir_size(index_dir)

    t0 = time.perf_counter()
    retriever = splade_mod.SpladeRetriever(index_dir, encoder=encoder)
    load_s = time.perf_counter() - t0
    rows = corpus_rows(corpus)

    # ---- recall, side by side with BM25 and dense on the same questions (if their indexes exist) ----
    ks = [int(k) for k in cfg["retrieval_eval.ks"]]
    placeholder, pool, n_available = eval_sets(cfg, set(retriever.doc_ids))
    eval_placeholder, _ = evaluate(retriever.search, placeholder, ks)
    eval_pool, pool_results = evaluate(retriever.search, pool, ks)
    eval_pool["n_available"] = n_available
    others: dict[str, Any] = {}
    if read_manifest(index_dir_for(cfg, "bm25")):
        bm25 = bm25_mod.BM25Retriever(index_dir_for(cfg, "bm25"), bm25_mod.BM25Settings.from_config(cfg))
        others["bm25_same_questions"] = {"placeholder_questions": evaluate(bm25.search, placeholder, ks)[0],
                                         "questions_all_sample": evaluate(bm25.search, pool, ks)[0]}
    if read_manifest(index_dir_for(cfg, "dense")):
        from ire.retrieval.dense import DenseRetriever

        dense = DenseRetriever.from_config(cfg)
        others["dense_same_questions"] = {"placeholder_questions": evaluate(dense.search, placeholder, ks)[0],
                                          "questions_all_sample": evaluate(dense.search, pool, ks)[0]}
    log.info("recall placeholder: %s | pool: %s | others: %s", eval_placeholder, eval_pool, others)
    common = common_checks(retriever.search, pool)
    edge = common["edge"]
    edge_ok = all(v["error"] is None for v in edge.values()) and edge["''"]["hits"] == 0

    # ---- sparsity and non-negativity ----
    q_vecs = retriever.query_vectors([q for _, q, _ in placeholder])
    q_nnz = np.diff(q_vecs.indptr)
    sparsity = {"doc": manifest["nnz_per_doc"],
                "query": {"mean": round(float(q_nnz.mean()), 2), "max": int(q_nnz.max()), "min": int(q_nnz.min()),
                          "n": len(q_nnz)},
                "vocab_size": manifest["vocab_size"], "vocab_terms_used_by_docs": manifest["vocab_terms_used"]}
    nonneg = {"doc_min_weight": float(retriever.docs.data.min()), "query_min_weight": float(q_vecs.data.min()),
              "passed": bool(retriever.docs.data.min() > 0 and q_vecs.data.min() >= 0)}

    # ---- exactness: search() == manual dot product of stored vectors; score_docs() == search() ----
    n_exact = int(cfg["splade.exactness_check_queries"])
    exact_q = [q for _, q, _ in pool[:n_exact]]
    exact_hits = retriever.search(exact_q, 100)
    qmat = retriever.query_vectors(exact_q)
    max_rel = max_sd = 0.0
    for i, hits in enumerate(exact_hits):
        ids = [d for d, _ in hits]
        got = np.array([s for _, s in hits], dtype=np.float64)
        manual = retriever.doc_vectors(ids).toarray().astype(np.float64) @ qmat[i].toarray().ravel().astype(np.float64)
        max_rel = max(max_rel, float(np.max(np.abs(got - manual) / np.maximum(np.abs(manual), 1e-12))))
        max_sd = max(max_sd, float(np.max(np.abs(retriever.score_docs(exact_q[i], ids) - got))))
    exactness = {"n_queries": len(exact_q), "n_pairs": sum(len(h) for h in exact_hits),
                 "max_rel_diff_search_vs_manual": max_rel, "max_abs_diff_score_docs_vs_search": max_sd,
                 "passed": max_rel < 1e-5 and max_sd < 1e-4}

    # ---- self-retrieval ----
    rng = random.Random(seed)
    n_self = min(int(cfg["splade.self_retrieval_check_passages"]), retriever.n_docs)
    self_ids = rng.sample(retriever.doc_ids, n_self)
    texts = {d: t for d, t in iter_documents(corpus, shards) if d in set(self_ids)}
    top1 = retriever.search([texts[d] for d in self_ids], k=1)
    self_rate = sum(1 for d, hits in zip(self_ids, top1) if hits and hits[0][0] == d) / max(1, n_self)
    misses = [d for d, hits in zip(self_ids, top1) if not hits or hits[0][0] != d]
    self_retrieval = {"n": n_self, "top1_rate": round(self_rate, 4), "passed": self_rate >= 0.9, "misses": misses[:10]}

    rebuild_identical = None
    if args.check_rebuild:
        tmp_root = data_dir / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            splade_mod.build_splade_index(corpus, Path(tmp) / "splade_rebuild", encoder, settings, shards,
                                          int(cfg["splade.checkpoint_every"]))
            again = splade_mod.SpladeRetriever(Path(tmp) / "splade_rebuild", encoder=encoder, mmap=False)
            same_vectors = bool((again.docs != retriever.docs).nnz == 0)
            rebuild_identical = same_vectors and again.search([q for _, q, _ in pool], max(ks)) == pool_results
            del again
        log.info("rebuild: identical vectors and top-%d results: %s", max(ks), rebuild_identical)

    peak = encoder.peak_vram_gib()
    cap = cfg["compute"].get("max_vram_gb")
    checks = {
        "n_docs_matches_corpus_plus_poison": manifest["n_docs"] == rows + manifest["n_poison"],
        "recall_reported": "recall@100" in eval_pool,
        "same_query_twice_identical": common["same_twice"],
        "edge_queries_no_crash_empty_returns_nothing": edge_ok,
        "weights_non_negative": nonneg["passed"],
        "search_exact_and_score_docs_agree": exactness["passed"],
        "self_retrieval_top1_ge_0.9": self_retrieval["passed"],
    }
    if cap and peak is not None:
        checks["vram_peak_under_cap"] = peak <= cap
    if rebuild_identical is not None:
        checks["rebuild_identical"] = rebuild_identical
    return {
        "retriever": "splade", "checks": checks, "built_now": built,
        "index_dir": index_dir.relative_to(data_dir).as_posix(),
        "model": settings.model, "model_revision": encoder.revision, "device": encoder.device,
        "encode_dtype": settings.encode_dtype, "encode_batch_size_final": encoder.batch_size,
        "build_time_s": manifest["timings_s"]["total"], "build_timings_s": manifest["timings_s"],
        "encode_passages_per_s": manifest["encode_passages_per_s"], "resumed_from_row": manifest["resumed_from_row"],
        "vram_peak_gib": {"build": build_vram, "overall": peak, "cap": cap},
        "index_bytes": index_bytes, "index_mib": round(index_bytes / 2**20, 2),
        "load_time_s": round(load_s, 3), "model_load_time_s": round(model_load_s, 2),
        "corpus": {"path": corpus.relative_to(data_dir).as_posix(), "rows": rows,
                   "sha256": manifest["fingerprint"]["corpus_sha256"]},
        "n_docs": manifest["n_docs"], "n_clean": manifest["n_clean"], "n_poison": manifest["n_poison"],
        "nnz": manifest["nnz"], "poison_shards": [s.name for s in shards], "settings": manifest["settings"],
        "eval": {"ks": ks, "placeholder_questions": eval_placeholder, "questions_all_sample": eval_pool, **others},
        "latency_ms": {"single_query_median_ms": common["single_query_median_ms"],
                       "batch_ms_per_query": eval_pool.get("batch_ms_per_query"),
                       "note": "includes encoding the query"},
        "determinism": {"same_query_twice": common["same_twice"], "rebuild_identical": rebuild_identical},
        "edge_queries": edge, "sparsity": sparsity, "non_negative": nonneg, "exactness": exactness,
        "self_retrieval": self_retrieval,
        "scale_estimate_full_corpus": splade_scale_estimate(manifest["nnz_per_doc"]["mean"]),
        "libraries": library_versions("torch", "transformers", "scipy", "numpy", "huggingface_hub"),
        "_summary": f"{manifest['n_docs']:,} docs ({manifest['n_poison']} poison), {settings.model}, "
                    f"{manifest['nnz_per_doc']['mean']} non-zeros/doc",
    }


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", required=True, choices=["bm25", "dense", "splade"])
    parser.add_argument("--force", action="store_true", help="rebuild even if the index is current")
    parser.add_argument("--no-poison-shards", action="store_true", help="ignore poison_*.jsonl shards")
    parser.add_argument("--check-rebuild", action="store_true",
                        help="build a second index in a temp folder and check its results are identical")
    parser.add_argument("--report", help="report name (default: A2/A3/A4 on Ada, <job>_local otherwise)")
    args = parser.parse_args()
    t_start = time.time()
    cfg = load_config()
    log = get_logger("build_index")
    job = args.report or JOB_NAMES[args.retriever] + ("" if cfg.env == "ada" else f"_{cfg.env}")
    body = {"bm25": run_bm25, "dense": run_dense, "splade": run_splade}[args.retriever](args, cfg, log)
    summary = body.pop("_summary")
    ok = all(body["checks"].values())
    report = {"job": job, "env": cfg.env, "ok": ok, "git_commit": git_commit(),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body,
              "runtime_s": round(time.time() - t_start, 1),
              "rerun": "python scripts/build_index.py " + " ".join(sys.argv[1:])}
    report_path = cfg.path("reports_dir") / f"{job}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n== {args.retriever} env={cfg.env}: {summary}, index {report['index_mib']} MiB, build "
          f"{report['build_time_s']} s{'' if report['built_now'] else ' (earlier build)'}, load {report['load_time_s']} s")
    ev = report["eval"]
    for name, key in (("placeholder questions", "placeholder_questions"),
                      ("questions_all (gold in index)", "questions_all_sample")):
        print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in ev[key].items()))
    for other in ("bm25", "dense"):
        if ev.get(f"{other}_same_questions"):
            for name, key in ((f"  {other}, placeholder", "placeholder_questions"),
                              (f"  {other}, questions_all", "questions_all_sample")):
                print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in ev[f"{other}_same_questions"][key].items()))
    print(f"latency: {report['latency_ms']}")
    for key in ("vram_peak_gib", "prefix_test", "reencode_check", "norms", "sparsity", "non_negative", "exactness",
                "self_retrieval", "unique_token", "scale_estimate_full_corpus"):
        if key in report:
            print(f"{key}: {report[key]}")
    print(f"edge queries: {report['edge_queries']}")
    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {report_path.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
