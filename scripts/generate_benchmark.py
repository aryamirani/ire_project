"""Step 9 / job A7: generation harness checks and throughput benchmark.

Usage (repo root):  python scripts/generate_benchmark.py [--batch-sizes 1,4,8] [--n-throughput 24] [--report NAME]

Runs on the active environment's LLM (local: Qwen2.5-1.5B-Instruct stand-in; Ada: Qwen2.5-3B-Instruct):
cache hit / key order / namespace tests, greedy determinism on 20 prompts, batch independence (one by one vs batches
of 8 mixed with other prompts vs another order, plus how close the first answer token is to flipping), throughput
at several batch sizes with projections for 60,000 and 30,000 requests, passage truncation statistics, and a smoke
EM/F1 on 50 clean questions. The tests use a TEMPORARY generation cache; the real one is not touched.
Contexts: the dense top-150 cache's top-5 if it is current, else (e.g. on Ada before A3/A5) each question's gold
passages plus seeded random passages; the report says which.
Report: $IRE_DATA_DIR/reports/<A7|A7_local>.json. Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.textstore import TextStore  # noqa: E402
from ire.generation.cache import GenerationCache, cache_key  # noqa: E402
from ire.generation.llm import Generator, GenerationSettings  # noqa: E402
from ire.generation.smoke_metrics import best_em_f1  # noqa: E402
from ire.retrieval.cache import CacheError, load_cache  # noqa: E402
from ire.schemas import QuestionRecord, read_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit, library_versions  # noqa: E402

N_DETERMINISM = 20
N_EM = 50


def power_info() -> dict[str, Any]:
    try:
        import psutil

        b = psutil.sensors_battery()
        return {"battery": b is not None, "on_mains": None if b is None else bool(b.power_plugged),
                "charge_percent": None if b is None else round(b.percent)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def contexts(cfg: Any, questions: list[QuestionRecord], seed: int) -> tuple[dict[str, list[str]], str]:
    """qid -> top-5 doc_ids: the dense cache if it is current, else gold passages + seeded random passages."""
    try:
        dense = load_cache(["dense"], cfg=cfg)["dense"]
        return {q.qid: [d for d, _ in dense[q.qid][:5]] for q in questions}, "dense top-150 cache (no poison)"
    except CacheError as exc:
        reason = str(exc).splitlines()[0][:120]
    with cfg.path("corpus").open("r", encoding="utf-8") as fh:
        ids = [json.loads(line)["doc_id"] for line in fh if line.strip()]
    rng = random.Random(seed)
    out = {}
    for q in questions:
        gold = list(q.gold_doc_ids[:5])
        fill = [d for d in rng.sample(ids, 10) if d not in gold][:5 - len(gold)]
        out[q.qid] = gold + fill
    return out, f"gold passages + seeded random passages (dense cache unavailable: {reason})"


def truncation_stats(gen: Generator, store: TextStore, seed: int, max_docs: int = 20_000) -> dict[str, Any]:
    """Token length of formatted passages ("<title>: <text>") with the LLM tokenizer (all, or a seeded sample)."""
    ids = list(store)
    if len(ids) > max_docs:
        ids = random.Random(seed).sample(ids, max_docs)
    lengths = []
    for start in range(0, len(ids), 2000):
        recs = store.records(ids[start:start + 2000])
        texts = [f"{t}: {x}" if gen.settings.include_title and t else x for t, x in recs]
        lengths += [len(v) for v in gen.tokenizer(texts, add_special_tokens=False)["input_ids"]]
    limit = gen.settings.passage_max_tokens
    hit = sum(1 for n in lengths if n > limit)
    return {"n_passages": len(lengths), "sampled": len(lengths) < len(store), "limit_tokens": limit,
            "mean_tokens": round(statistics.mean(lengths), 1), "median_tokens": statistics.median(lengths),
            "max_tokens": max(lengths), "n_over_limit": hit, "frac_over_limit": round(hit / len(lengths), 4)}


def first_token_margins(gen: Generator, prompts: list[str], batch: int) -> dict[str, Any]:
    """Logits of the first answer token: one prompt at a time vs inside left-padded batches. Reports how much
    batching moves them and the smallest gap between the best and second-best token (a flip needs a move > gap)."""
    import torch

    model = gen.model

    def last_logits(group: list[str]) -> Any:
        enc = gen.tokenizer(group, return_tensors="pt", padding=True).to(gen.device)
        with torch.inference_mode():
            return model(**enc, logits_to_keep=1).logits[:, -1, :].float().cpu()  # last position only

    single = torch.cat([last_logits([p]) for p in prompts])
    batched = torch.cat([last_logits(prompts[i:i + batch]) for i in range(0, len(prompts), batch)])
    top2 = torch.topk(single, 2, dim=1).values
    return {"max_abs_logit_diff": round(float((single - batched).abs().max()), 5),
            "min_top1_top2_gap": round(float((top2[:, 0] - top2[:, 1]).min()), 4),
            "same_first_token": int((single.argmax(1) == batched.argmax(1)).sum()), "n": len(prompts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", default="1,4,8", help="batch sizes for the throughput test")
    parser.add_argument("--n-throughput", type=int, default=24, help="prompts per throughput run")
    parser.add_argument("--report", help="report name (default: A7 on Ada, A7_local otherwise)")
    args = parser.parse_args()

    t_start = time.time()
    cfg = load_config()
    log = get_logger("generate_benchmark")
    seed = int(cfg["seed"])
    job = args.report or ("A7" if cfg.env == "ada" else f"A7_{cfg.env}")
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    clean = [q for q in questions if q.role == "clean"]
    need = N_DETERMINISM + 4 + args.n_throughput + N_EM
    if len(clean) < need:
        raise SystemExit(f"need {need} clean questions, found {len(clean)}")
    t0 = time.time()
    store = TextStore.open_for(cfg)
    store_s = time.time() - t0
    ctx, ctx_source = contexts(cfg, clean, seed)
    log.info("contexts: %s; text store %d docs (%.1f s)", ctx_source, len(store), store_s)

    data_tmp = cfg.path("data_dir") / "tmp"
    data_tmp.mkdir(parents=True, exist_ok=True)
    with (tempfile.TemporaryDirectory(dir=data_tmp, ignore_cleanup_errors=True) as tmp,
          closing(GenerationCache(Path(tmp) / "benchmark_cache.sqlite")) as cache):  # closed even on errors
        gen = Generator.from_config(cfg, texts=store, cache=cache)
        import torch

        t0 = time.time()
        gen.model  # load
        load_s = time.time() - t0
        vram_weights = round(torch.cuda.memory_allocated() / 2**30, 3) if gen.device == "cuda" else None
        gpu = torch.cuda.get_device_name(0) if gen.device == "cuda" else "cpu"
        log.info("model %s@%s on %s (%s), loaded in %.1f s, weights %s GiB", gen.settings.model, gen.revision[:12],
                 gpu, gen.settings.dtype, load_s, vram_weights)

        prompt_of = {q.qid: gen.build_prompt(q.question, ctx[q.qid]) for q in clean[:need]}
        det_q = clean[:N_DETERMINISM]
        det_p = [prompt_of[q.qid] for q in det_q]
        other_p = [prompt_of[q.qid] for q in clean[N_DETERMINISM:N_DETERMINISM + 4]]

        # ---- greedy determinism: 20 prompts, one by one, twice
        run1 = gen.generate_raw(det_p, 1)
        run2 = gen.generate_raw(det_p, 1)
        determinism = {"n": len(det_p), "identical": sum(a == b for a, b in zip(run1, run2))}

        # ---- batch independence: vs batches of 8 mixed with other prompts, vs another order, vs the default path
        mixed = det_p + other_p
        order = list(range(len(mixed)))
        random.Random(seed).shuffle(order)
        res = gen.generate_raw([mixed[i] for i in order], 8, sort_by_length=False)
        by_pos = dict(zip(order, res))
        batch_mixed = [by_pos[i] for i in range(len(det_p))]
        batch_reversed = gen.generate_raw(det_p[::-1], 8, sort_by_length=False)[::-1]
        batch_default = gen.generate_raw(det_p, 8)
        same = lambda xs: sum(a == b for a, b in zip(run1, xs))  # noqa: E731
        diffs = [{"qid": det_q[i].qid, "single": run1[i], "batch": xs[i]}
                 for xs in (batch_mixed, batch_reversed, batch_default) for i in range(len(det_p)) if run1[i] != xs[i]]
        independence = {"n": len(det_p), "batch8_mixed_shuffled": same(batch_mixed),
                        "batch8_reversed_order": same(batch_reversed), "batch8_length_sorted": same(batch_default),
                        "differences": diffs[:10], "first_token_logits": first_token_margins(gen, det_p, 8)}
        independence["all_identical"] = not diffs

        # ---- cache tests (temporary cache)
        q0 = clean[0]
        top5 = ctx[q0.qid]
        a = gen.generate(q0.qid, q0.question, top5)
        b = gen.generate(q0.qid, q0.question, top5)
        rev = gen.generate(q0.qid, q0.question, top5[::-1])
        gen33 = Generator(replace(gen.settings, max_new_tokens=gen.settings.max_new_tokens + 1), store, cache,
                          batch_size=gen.batch_size)
        gen33.share_model_from(gen)
        c = gen33.generate(q0.qid, q0.question, top5)
        cache_tests = {
            "first_call_cached": a["cached"], "second_call_cached": b["cached"], "same_answer": a["answer"] == b["answer"],
            "reversed_ids_key_differs": cache_key(q0.qid, top5) != cache_key(q0.qid, top5[::-1]),
            "reversed_ids_cached": rev["cached"], "other_max_new_tokens_namespace_differs": gen33.namespace != gen.namespace,
            "other_max_new_tokens_cached": c["cached"]}
        cache_ok = (not a["cached"] and b["cached"] and a["answer"] == b["answer"] and cache_tests["reversed_ids_key_differs"]
                    and not rev["cached"] and cache_tests["other_max_new_tokens_namespace_differs"] and not c["cached"])

        # ---- throughput (no cache)
        tp_p = [prompt_of[q.qid] for q in clean[N_DETERMINISM + 4:N_DETERMINISM + 4 + args.n_throughput]]
        throughput = {}
        for bs in sizes:
            if gen.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            gen.generate_raw(tp_p, bs)
            dt = time.perf_counter() - t0
            rpm = len(tp_p) / dt * 60
            throughput[str(bs)] = {"requests": len(tp_p), "seconds": round(dt, 2), "requests_per_min": round(rpm, 1),
                                   "peak_vram_gib": gen.peak_vram_gib(), "final_batch_size": min(bs, gen.batch_size)}
        best = max(throughput.values(), key=lambda x: x["requests_per_min"])
        projection = {f"hours_for_{n}": round(n / best["requests_per_min"] / 60, 2) for n in (60_000, 30_000)}
        projection["at_requests_per_min"] = best["requests_per_min"]

        # ---- smoke EM / F1 on 50 clean questions
        em_q = clean[N_DETERMINISM + 4 + args.n_throughput:need]
        answers = gen.generate_batch([(q.qid, q.question, ctx[q.qid]) for q in em_q])
        scores = [best_em_f1(r["answer"], q.answers) for r, q in zip(answers, em_q)]
        smoke = {"n": len(em_q), "em": round(statistics.mean(s[0] for s in scores), 4),
                 "f1": round(statistics.mean(s[1] for s in scores), 4), "contexts": ctx_source}
        samples = []
        for q, r in list(zip(em_q, answers))[:3]:
            p = gen.build_prompt(q.question, ctx[q.qid])
            samples.append({"qid": q.qid, "question": q.question, "gold": q.answers[:3], "answer": r["answer"],
                            "prompt_shortened": p[:260] + " ... " + p[-120:]})
        trunc = truncation_stats(gen, store, seed)
        example_prompt = gen.build_prompt(clean[0].question, ctx[clean[0].qid])
        peak = gen.peak_vram_gib()
        namespace = {"id": gen.namespace, "settings": gen.namespace_settings()}

    checks = {"cache_hit_and_keys": cache_ok, "greedy_determinism_20_of_20": determinism["identical"] == determinism["n"],
              "smoke_em_f1_computed": smoke["n"] == N_EM}
    cap = cfg["compute"].get("max_vram_gb")
    if cap and peak is not None:
        checks["vram_peak_under_cap"] = peak <= cap
    ok = all(checks.values())
    report = {
        "job": job, "env": cfg.env, "ok": ok, "checks": checks, "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu, "power": power_info(), "model": gen.settings.model, "model_revision": gen.revision,
        "dtype": gen.settings.dtype, "model_load_s": round(load_s, 1), "vram_gib": {"weights": vram_weights,
                                                                                   "peak": peak, "cap": cap},
        "settings": GenerationSettings.from_config(cfg).__dict__, "namespace": namespace,
        "text_store": {"n_docs": len(store), "open_or_build_s": round(store_s, 1)},
        "throughput": throughput, "projection": projection, "cache_tests": cache_tests,
        "determinism": determinism, "batch_independence": independence, "smoke_em_f1": smoke,
        "samples": samples, "truncation": trunc, "example_prompt": example_prompt,
        "libraries": library_versions("torch", "transformers", "accelerate", "tokenizers"),
        "runtime_s": round(time.time() - t_start, 1),
        "rerun": "python scripts/generate_benchmark.py " + " ".join(sys.argv[1:]),
    }
    out = cfg.path("reports_dir") / f"{job}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== generation env={cfg.env}: {gen.settings.model}@{gen.revision[:12]} on {gpu} ({gen.settings.dtype}), "
          f"power {report['power']}")
    print(f"VRAM: weights {vram_weights} GiB, peak {peak} GiB (cap {cap})")
    for bs, r in throughput.items():
        print(f"batch {bs}: {r['requests_per_min']} requests/min ({r['requests']} in {r['seconds']} s), peak VRAM "
              f"{r['peak_vram_gib']} GiB")
    print(f"projection at {projection['at_requests_per_min']} req/min: 60,000 -> {projection['hours_for_60000']} h, "
          f"30,000 -> {projection['hours_for_30000']} h")
    print(f"cache tests: {cache_tests}")
    print(f"determinism: {determinism}")
    print("batch independence:", {k: v for k, v in independence.items() if k != "differences"})
    for d in independence["differences"]:
        print(f"   DIFF {d['qid']}: single {d['single'][:70]!r} | batch {d['batch'][:70]!r}")
    print(f"smoke EM/F1 on {smoke['n']} clean questions: EM {smoke['em']}, F1 {smoke['f1']} ({ctx_source})")
    print(f"truncation: {trunc}")
    for s in samples:
        print(f"sample {s['qid']}: {s['question']!r} -> {s['answer']!r} (gold {s['gold']})")
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}   runtime {report['runtime_s']} s")
    print(f"REPORT: {out.as_posix()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
