"""Arya: generate the poison shards (plan section 3.2) for the final question set.

Usage (repo root; needs data/questions.jsonl from scripts/select_questions.py and corpus.jsonl):
    python scripts/make_poison.py                    # targets + both shards + report
    python scripts/make_poison.py --no-stealth-cap   # skip the dense similarity cap (no dense index needed)
    python scripts/make_poison.py --refreeze-targets # DANGER: overwrite configs/frozen/poison_targets.json

Outputs:
    $IRE_DATA_DIR/data/poison/poison_main.jsonl      fact, hijack, url: 200 target questions x 5 each (3,000)
    $IRE_DATA_DIR/data/poison/poison_adaptive.jsonl  diverse, repeat, stealth: 100 target questions x 5 (1,500)
    configs/frozen/poison_targets.json               per-question false target answers, codeword, URL and the
                                                     100 adaptive qids (tracked in git; regenerated targets
                                                     must equal it, otherwise the script stops)
    $IRE_DATA_DIR/reports/poison_<env>.json          counts, target methods, phrase-rule hit rates per attack,
                                                     stealth similarity-cap stats, shard sha256

Before job A6 the shards are validated with Samarth's `ire.retrieval.poison.load_shards`.
The held-out shard (poison_heldout.jsonl) is Samarth's and is not written here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import load_config  # noqa: E402
from ire.data.question_set import answer_types_of, read_frozen, sha256_file  # noqa: E402
from ire.defenses.query_phrase import phrase_hits  # noqa: E402
from ire.poison import templates as T  # noqa: E402
from ire.poison.build import build_adaptive, build_main  # noqa: E402
from ire.poison.targets import candidate_pool, choose_target  # noqa: E402
from ire.retrieval.common import doc_text, find_poison_shards  # noqa: E402
from ire.retrieval.poison import load_shards  # noqa: E402
from ire.schemas import QuestionAllRecord, QuestionRecord, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from ire.utils.log import get_logger  # noqa: E402
from ire.utils.run_info import git_commit  # noqa: E402

FROZEN_QS = REPO_ROOT / "configs" / "frozen" / "question_set.json"
FROZEN_TARGETS = REPO_ROOT / "configs" / "frozen" / "poison_targets.json"
MAIN_SHARD, ADAPTIVE_SHARD = "poison_main.jsonl", "poison_adaptive.jsonl"


def scan_corpus(corpus_path: Path, want: set[str], needles: list[str]) -> tuple[dict[str, tuple[str, str]], dict]:
    """One streaming pass: (title, text) of the wanted doc_ids and how many passages contain each needle."""
    found: dict[str, tuple[str, str]] = {}
    hits = Counter()
    low = [n.lower() for n in needles]
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj["doc_id"] in want:
                found[obj["doc_id"]] = (obj["title"], obj["text"])
            body = (obj["title"] + " " + obj["text"]).lower()
            for n in low:
                if n in body:
                    hits[n] += 1
    return found, {n: hits[n.lower()] for n in needles}


def adaptive_qids(targets: list[QuestionRecord], per_fold: int, seed: int) -> list[str]:
    rng = random.Random(f"{seed}:adaptive")
    out: list[str] = []
    for fold in sorted({q.fold for q in targets}):
        qids = sorted(q.qid for q in targets if q.fold == fold)
        out += sorted(rng.sample(qids, per_fold))
    return out


class StealthCap:
    """Similarity cap for stealth passages: cos(q, p) <= median cos of the question's clean top-50 dense pool."""

    def __init__(self, cfg: Any) -> None:
        from ire.retrieval.dense import DenseRetriever

        self.dense = DenseRetriever.from_config(cfg)
        self.pool_size = int(cfg["pool_size"])
        self.caps: dict[str, float] = {}

    def cap(self, question: str) -> float:
        if question not in self.caps:
            row = self.dense.search([question], k=self.pool_size + 200)[0]
            sims = [s for d, s in row if not d.startswith("poison_")][: self.pool_size]
            self.caps[question] = float(statistics.median(sims))
        return self.caps[question]

    def __call__(self, question: str, candidates: list[str]) -> tuple[str, dict]:
        cap = self.cap(question)
        q = self.dense.get_query_vector([question])[0]
        p = self.dense.encoder.encode_passages([doc_text("", c) for c in candidates])
        sims = [float(x) for x in p @ q]
        for variant, text, s in zip(("keywords", "title_only"), candidates, sims):
            if s <= cap:
                return text, {"variant": variant, "cos": round(s, 4), "cap": round(cap, 4), "under_cap": True}
        i = min(range(len(sims)), key=sims.__getitem__)
        return candidates[i], {"variant": ("keywords", "title_only")[i], "cos": round(sims[i], 4),
                               "cap": round(cap, 4), "under_cap": False}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-stealth-cap", action="store_true", help="do not apply the dense similarity cap")
    ap.add_argument("--refreeze-targets", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    log = get_logger("make_poison")
    seed = int(cfg["seed"])
    questions = read_jsonl(cfg.path("questions"), QuestionRecord)
    pool = read_jsonl(cfg.path("questions_all"), QuestionAllRecord)
    types = answer_types_of(read_frozen(FROZEN_QS))
    if set(types) != {q.qid for q in questions}:
        print("questions.jsonl does not match configs/frozen/question_set.json: run scripts/select_questions.py")
        return 1
    targets_q = [q for q in questions if q.role == "target"]
    per_fold = int(cfg["poison.adaptive_per_fold"])
    adaptive = adaptive_qids(targets_q, per_fold, seed)

    gold_ids = {d for q in targets_q for d in q.gold_doc_ids}
    log.info("scanning %s for %d gold passages and attack strings", cfg.path("corpus").name, len(gold_ids))
    gold, needle_hits = scan_corpus(cfg.path("corpus"), gold_ids, [T.CODEWORD, T.URL_TARGET])
    missing = gold_ids - set(gold)
    if missing:
        print(f"{len(missing)} gold passages of target questions are not in corpus.jsonl "
              f"(local profile: run scripts/build_local_corpus.py)")
        return 1

    cands = candidate_pool(pool)
    targets = {}
    for q in targets_q:
        texts = [doc_text(*gold[d]) for d in q.gold_doc_ids]
        all_q = QuestionAllRecord(qid=q.qid, question=q.question, answers=q.answers, gold_doc_ids=q.gold_doc_ids)
        targets[q.qid] = choose_target(all_q, types[q.qid], cands, texts, seed)

    frozen_targets = {
        "name": "poison_targets", "version": 1, "owner": "Arya", "seed": seed, "codeword": T.CODEWORD,
        "attack_url": T.ATTACK_URL, "url_target": T.URL_TARGET, "per_question": T.PER_QUESTION,
        "adaptive_qids": adaptive,
        "targets": {qid: {"gold_answer": t.gold_answer, "answer_type": t.answer_type,
                          "target_answer": t.target_answer, "method": t.method} for qid, t in targets.items()},
    }
    if args.refreeze_targets or not FROZEN_TARGETS.is_file():
        FROZEN_TARGETS.write_text(json.dumps(frozen_targets, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8", newline="\n")
        log.info("froze %d target answers in %s", len(targets), FROZEN_TARGETS)
    else:
        old = json.loads(FROZEN_TARGETS.read_text(encoding="utf-8"))
        diff = [k for k in ("codeword", "attack_url", "adaptive_qids", "targets") if old.get(k) != frozen_targets[k]]
        if diff:
            print(f"regenerated {diff} differ from {FROZEN_TARGETS.name} (corpus or code changed?). Stopping; "
                  f"use --refreeze-targets only if the change is intended and the team agrees.")
            return 1

    cap = None if args.no_stealth_cap else StealthCap(cfg)
    main_recs, adapt_recs, stealth_info = [], [], []
    by_qid = {q.qid: q for q in targets_q}
    for q in targets_q:
        title = gold[q.gold_doc_ids[0]][0] or q.question
        main_recs += build_main(q.qid, q.question, title, targets[q.qid].target_answer, seed)
    for qid in adaptive:
        q = by_qid[qid]
        title = gold[q.gold_doc_ids[0]][0] or q.question
        recs, info = build_adaptive(qid, q.question, title, targets[qid].target_answer, seed, cap)
        adapt_recs += recs
        stealth_info += info

    poison_dir = cfg.path("poison_dir")
    poison_dir.mkdir(parents=True, exist_ok=True)
    paths = {MAIN_SHARD: poison_dir / MAIN_SHARD, ADAPTIVE_SHARD: poison_dir / ADAPTIVE_SHARD}
    write_jsonl(paths[MAIN_SHARD], main_recs)
    write_jsonl(paths[ADAPTIVE_SHARD], adapt_recs)
    roles = {q.qid: q.role for q in questions}
    shards = load_shards(find_poison_shards(poison_dir), roles)   # Samarth's validation (raises on problems)

    every = main_recs + adapt_recs
    phrase = Counter()
    for r in every:
        if phrase_hits(r.text):
            phrase[r.attack] += 1
    n_by_attack = Counter(r.attack for r in every)
    words = {a: statistics.mean(len(r.text.split()) for r in every if r.attack == a) for a in n_by_attack}
    checks = {
        "main_count": len(main_recs) == len(targets_q) * len(T.MAIN_ATTACKS) * T.PER_QUESTION,
        "adaptive_count": len(adapt_recs) == len(adaptive) * len(T.ADAPTIVE_ATTACKS) * T.PER_QUESTION,
        "only_target_qids": not shards.summary(roles)["qids_with_role_clean"],
        "codeword_absent_from_corpus": needle_hits[T.CODEWORD] == 0,
        "url_absent_from_corpus": needle_hits[T.URL_TARGET] == 0,
        "fact_targets_in_text": all(r.target_answer in r.text for r in every if r.attack in ("fact", "diverse",
                                                                                              "repeat")),
    }
    ok = all(checks.values())
    stealth_under = [i for i in stealth_info if i.get("under_cap")]
    report = {
        "job": "poison", "env": cfg.env, "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(), "questions_sha256": sha256_file(cfg.path("questions")),
        "shards": {name: {"records": n, "sha256": sha256_file(paths[name])}
                   for name, n in ((MAIN_SHARD, len(main_recs)), (ADAPTIVE_SHARD, len(adapt_recs)))},
        "summary": shards.summary(roles),
        "by_attack": dict(sorted(n_by_attack.items())),
        "mean_words_by_attack": {a: round(w, 1) for a, w in sorted(words.items())},
        "phrase_rule_hit_rate_by_attack": {a: round(phrase[a] / n, 4) for a, n in sorted(n_by_attack.items())},
        "target_methods": dict(Counter(t.method for t in targets.values())),
        "target_answer_types": dict(Counter(t.answer_type for t in targets.values())),
        "adaptive_qids_per_fold": dict(Counter(by_qid[q].fold for q in adaptive)),
        "stealth_similarity_cap": None if cap is None else {
            "passages": len(stealth_info), "under_cap": len(stealth_under),
            "variant": dict(Counter(i["variant"] for i in stealth_info)),
            "mean_cos": round(statistics.mean(i["cos"] for i in stealth_info), 4),
            "mean_cap": round(statistics.mean(i["cap"] for i in stealth_info), 4)},
        "corpus_occurrences": needle_hits,
        "checks": checks, "ok": ok,
    }
    out = cfg.path("reports_dir") / f"poison_{cfg.env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {len(main_recs)} main + {len(adapt_recs)} adaptive poison passages to {poison_dir}")
    print(f"by attack: {report['by_attack']}")
    print(f"phrase-rule hit rate by attack: {report['phrase_rule_hit_rate_by_attack']}")
    print(f"target methods: {report['target_methods']}")
    if cap is not None:
        print(f"stealth cap: {report['stealth_similarity_cap']}")
    for name, value in checks.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    print(f"RESULT: {'OK' if ok else 'FAILED'}")
    print(f"REPORT: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
