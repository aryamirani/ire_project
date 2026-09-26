# Team Rules & Working Guide

How we work on this project: shared data, setup, storage, conventions and the weekly task lists. The research plan itself is in [`New_project_plan.md`](New_project_plan.md).

**Team:** Arya, Samarth, Vansh
**Roles:** P1 = search, embeddings, generation, provenance filter · P2 = data, poisoning, outlier filter, cascade · P3 = evaluation, calibration, plots

---

## 1. Ground Rules

1. **One source of truth for data.** Everyone works from the data in this repo (via Git LFS) or the one shared copy on Ada. Never from a private copy on your laptop or your own Ada folder.
2. **Pull before you run anything.** `git pull` (LFS files come with it) before every experiment.
3. **Data has owners.** Only the owner changes a data file, in a commit that says what changed:
   * P2: question sets, corpus samples / ID lists, poisoned passages, trust tiers, injection-phrase list.
   * P1: cached retrieval lists, generation cache.
   * P3: evaluation results, human labels (final agreed version).
4. **Fixed choices don't change mid-project:** random seed **42**, the generator model and prompt (once picked in Week 2), the injection-phrase list (once fixed in Week 1). If one really must change, tell the others first and re-run what depends on it.
5. **No reported number comes from the development sample.**
6. **Results freeze on Sun 25 Oct.** After that, no new experiments; the data and results in the repo at that point are what the report uses.
7. **Commits:** short, plain messages describing the change. No tool-generated trailers or co-author lines. Any use of AI tools is disclosed in `AI_USAGE.txt` only. Keep it updated as you go instead of reconstructing it at the end.

---

## 2. First-Time Setup (each person)

```bash
# 1. Install Git LFS (once per machine)
brew install git-lfs          # macOS; on Ada/Linux use the git-lfs binary or conda: conda install -c conda-forge git-lfs
git lfs install

# 2. Clone
git clone https://github.com/aryamirani/ire_project.git
cd ire_project

# 3. If any data file shows up as a tiny text pointer instead of real data
git lfs pull
```

**Install Git LFS before cloning.** Without it, data files arrive as small placeholder text files instead of the real data.

On Ada, also:
* Use the one shared conda environment (don't create your own copy).
* Set `HF_HOME` to a folder with enough space (see Section 4).

---

## 3. Shared Data in the Repo (Git LFS)

This is a 3-person project, so **everyone must work from exactly the same data**. The datasets and all data we generate are **pushed to this repo with Git LFS**, so anyone can clone and get identical, correct files.

The repo is already set up: `.gitattributes` sends these paths to LFS automatically:
```
data/**   labels/**   results/**/*.jsonl   *.parquet   *.npy
```
Just `git add` / `git commit` / `git push` files under those paths as normal.

**What goes into the repo via LFS:**

| File(s) | Owner | Why | Approx. size |
|---|---|---|---|
| Question sets (target + clean) with gold answers and attacker target answers | P2 | Everyone evaluates the same questions | < 1 MB |
| Passage ID lists: development sample, and the ~500k fallback corpus if used | P2 | Defines the exact corpus | ~5–30 MB |
| Development sample text (~200k passages) | P2 | Develop without downloading the full corpus | ~120 MB |
| All poisoned passages (incl. attack type, target string) | P2 | Attacks must be identical for everyone | ~2 MB |
| Trust tiers for every passage | P2 | Provenance results depend on them | ~30 MB |
| Injection-phrase list, hijack templates | P2 | Filter 2 and attacks must match | < 1 MB |
| Cached top-150 retrieval lists | P1 | Filters and sweeps use the same candidates | ~10–20 MB |
| Generation cache | P1 | No duplicate LLM calls; same answers for everyone | ~10–50 MB |
| `eval.jsonl` results | P3 | Everyone plots from the same results | ~10–50 MB |
| Human labels (calibration) | P3 | Calibration uses the final agreed labels | < 1 MB |

**What does *not* go into LFS, and how we keep it identical anyway:**
* The **full raw NQ corpus (~2 GB)** and the **indexes and dense embeddings (~10–20 GB, single files of several GB)** are too large for GitHub LFS:
  * GitHub rejects individual LFS files over 2 GB.
  * LFS storage and download quotas are limited (check the account's current quota), and every clone by each of us counts against the download quota.
* For these, the repo contains instead:
  * a **download script** with the exact source (Hugging Face dataset name and revision, or the BEIR zip URL),
  * **SHA-256 checksums** of the raw files, verified by the script after download,
  * **build scripts** for every index, with fixed model names, seeds and settings,
  * the path of the **one shared copy on Ada** that everyone reads from.
* These folders are in `.gitignore` (`data/raw/`, `indexes/`), so they can't be committed by accident.

---

## 4. Storage on Ada

We don't do special optimisation (no quantisation). We try the simple setup; if it doesn't fit, we shrink the corpus (decision Fri 2 Oct).

| Item | Full corpus (~2.7M) | Fallback (~500k) |
|---|---|---|
| NQ raw data | ~2 GB | ~2 GB (raw download, before sampling) |
| BM25 index | ~2–3 GB | < 1 GB |
| Dense embeddings + FAISS index | ~8 GB | ~1.5 GB |
| SPLADE index | ~2–3 GB (plus temporary files while indexing) | < 1 GB |
| Retriever models | ~1 GB | ~1 GB |
| Generator LLM (3–4B, unquantised) | ~7–8 GB | ~7–8 GB |
| Conda env | ~6–10 GB | ~6–10 GB |
| Cached results, generations, labels | < 1 GB | < 1 GB |
| **Total** | **~30–35 GB (tight for 32 GB)** | **~20–25 GB (fits)** |

A 7–8B unquantised model adds another ~8 GB, so it only makes sense with the fallback corpus or if we have more than 32 GB.

**Habits:**
* Point `HF_HOME` at a folder with space, so model downloads don't silently fill the home directory.
* Delete the downloaded zip after unzipping, and SPLADE's temporary files after indexing.
* Run `conda clean -a` and `pip cache purge` after installing packages.
* One shared environment and one copy of the indexes, not one per person.
* Use Ada's scratch space for temporary work inside jobs if available (it may be node-local and wiped after the job, so copy results back).
* Encode and index in chunks that save progress, so a job time limit or failure doesn't lose work.

---

## 5. Data Sources

| What | Where | Notes |
|---|---|---|
| BEIR NQ corpus, queries, relevance labels | Hugging Face `BeIR/nq` + `BeIR/nq-qrels`, or `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip` | ~2.7M passages, 3,452 test queries |
| PoisonedRAG NQ target questions | `github.com/sleeepeer/PoisonedRAG` (see its README for the file location) | Correct + attacker target answers for 100 NQ questions |
| Extra gold answers | Hugging Face `google-research-datasets/nq_open` | Match by question text; spot-check 20 matches |

**Licence / credit:** NQ is released under CC BY-SA 3.0. The README credits NQ, BEIR and PoisonedRAG as data sources.

---

## 6. Repo Layout

```
data/            # Git LFS: question sets, passage ID lists, dev sample, poison, trust tiers
  raw/           # gitignored: full NQ download (rebuilt by script, checked by checksum)
indexes/         # gitignored: BM25 / FAISS / SPLADE indexes, dense embeddings (build scripts in src/, shared copy on Ada)
src/
  data/          # download, samples, trust tiers, poison
  retrieval/     # bm25, dense, splade, rrf, get_doc_vector
  generation/    # LLM prompting + generation cache
  defenses/      # provenance, outlier, cascade
  eval/          # candidate pool, metrics, judge, calibration
configs/         # experiment configs
results/         # Git LFS: top-150 caches, generation cache, eval results
notebooks/       # exploration and plots
labels/          # Git LFS: human labels for judge calibration
submissions/     # submitted documents
RULES.md         # this file
New_project_plan.md
AI_USAGE.txt
```

---

## 7. Data Interfaces (JSON Records)

Everyone reads and writes these formats, so the three of us can work in parallel.

```jsonc
// passages.jsonl: one per passage
{"doc_id": "doc123" | "poison_q17_fact_2", "text": "...", "trust_tier": "verified|community|external",
 "trust": 1.0, "is_poison": false, "attack": null | "fact|hijack|url", "target_qid": null | "q17",
 "target_string": null | "wrong answer / codeword / URL"}

// retrieval.jsonl: one per (question, retriever), cached raw top-150 from the single index.
// Raw scores only, never ranks. RRF has no rows here: it is fused at runtime from bm25 + dense.
{"qid": "q17", "is_target": true, "retriever": "bm25|dense|splade",
 "top150": [{"doc_id": "doc123", "raw_score": 12.4, "trust_tier": "verified"},
            {"doc_id": "poison_q17_fact_1", "raw_score": 13.9, "trust_tier": "external"}, ...],
 "latency_ms": 8.1}

// generation.jsonl: deduplicated generation cache
{"gen_key": "sha1(qid + ordered context_ids)", "qid": "q17",
 "context_ids": ["...5 ids in order"], "answer": "..."}

// eval.jsonl: one per experiment configuration
{"qid": "q17", "is_target": true, "retriever": "dense", "poison_rate": 3, "attack": "fact",
 "filter": "none|provenance|outlier|outlier_nophrase|cascade", "setting": 0.4 | {"theta": 3, "lambda": 0.6},
 "gen_key": "...", "ndcg@5": 0.72, "ndcg@10": 0.66, "poison_in_top5": 0, "n_poison_top5": 0,
 "collateral_poison_top5": 0, "filter_ms": 3.2,
 "attack_success": 0, "em": 1, "f1": 0.85,
 "ragas_faithfulness": null | 0.95, "ragas_answer_relevance": null | 0.9, "ragas_correctness": null | 0.8}
```

---

## 8. Implementation Conventions

* **Poisoned passage IDs:** `poison_<qid>_<attack>_<n>`, `attack ∈ {fact, hijack, url}`, `n ∈ 1..5`.
* **Retrieval caches store raw scores only, never ranks.** Ranks are always recomputed after masking.
* **Candidate pool** (shared by every experiment; owner P3):
  ```python
  def is_active(d, attack, rate):
      return (not d.is_poison) or (d.attack == attack and d.n <= rate)

  def candidate_pool(raw_top150, attack, rate, K=50):
      kept = [d for d in raw_top150 if is_active(d, attack, rate)]
      kept = sorted(kept, key=lambda d: (-d.raw_score, d.doc_id))[:K]
      for r, d in enumerate(kept, start=1):
          d.rank = r
      return kept
  ```
* **RRF:** never cache the fused list. Mask and re-rank the BM25 and dense lists separately, then fuse with `Σ 1 / (60 + rank')`, then take the top-50.
* **Dense embeddings** for the outlier filter come from `get_doc_vector(doc_id)` (owner P1), for all retrievers.
* **E5** needs the `query: ` / `passage: ` prefixes.
* **Generation cache key:** `sha1(qid + ordered top-5 doc IDs)`. The question ID must be in the key.
* **Generator prompt:** the neutral template in the plan (Section 3), temperature 0, passages in ranked order. No safety instructions.
* **Injection-phrase list** is fixed (and committed) **before** any hijack template is written.
* **Seed 42** everywhere (sampling, trust-tier assignment, splits).

---

## 9. Labelling Logistics (Judge Calibration)

* 150 answers, split into three blocks of 50, each block labelled by a pair:

  | Block | Labellers |
  |---|---|
  | A (1–50) | Arya + Samarth |
  | B (51–100) | Samarth + Vansh |
  | C (101–150) | Vansh + Arya |

  So each person labels 100 (about 2–3 hours).
* Google Sheet: question, passages, answer, one label column per labeller. The judge's verdict is hidden.
* Label independently first, then settle disagreements together. The final agreed labels are committed to `labels/` by P3.
* Dev/test split (75/75) is fixed before any judge-prompt tuning.

---

## 10. Weekly Task Lists

### Week 0 (Sat 26 – Sun 27 Sep): Setup
- [ ] **All:** agree on Python version and `requirements.txt`; seed 42.
- [ ] **All:** check everyone has an **Ada** account; set up the one shared conda environment; find out which GPUs are available and how much storage each of us has (Section 4).
- [ ] **All:** install Git LFS and clone the repo (Section 2); check the account's LFS quota.
- [ ] **All:** agree on the JSON records (Section 7) and repo layout (Section 6).
- [ ] **All:** create `AI_USAGE.txt` and add to it as you go.
- [ ] **P2:** download BEIR NQ with a script that records the source and checksums; clone PoisonedRAG and locate its NQ targets.

### Week 1 (28 Sep – 4 Oct): Data, Retrievers, Poisoning
- [ ] **P2 (by Tue 29 Sep):** choose target + clean questions; gold answers (spot-check 20); build the ~200k development sample; push question sets, ID list and sample via LFS.
- [ ] **P2 (Wed):** fix and push the generic injection-phrase list **before** writing any hijack templates.
- [ ] **P2 (Wed–Sun):** poison generators (fact distortion; instruction hijack with codewords in 10+ phrasings), 5 per target question per attack, with target strings; trust tiers. Push via LFS.
- [ ] **P1:** BM25, dense, SPLADE, RRF. Develop on the sample Mon–Tue. **From Wed, try full-corpus indexing on Ada as-is**, in chunks that save progress.
- [ ] **All (Fri 2 Oct): corpus decision.** If full-corpus indexing worked and fits, keep it. Otherwise P2 builds the ~500k fallback (gold passages + BM25 hard negatives, + dense hard negatives if available, + random) and P1 indexes that.
- [ ] **P1:** once dense embeddings exist, provide `get_doc_vector(doc_id)`.
- [ ] **P3:** nDCG@5/@10, Recall@k, poison-in-top-k, string-match ASR, EM/F1; RAGAS set up and tested on dummy answers.
- [ ] **Milestone:** clean nDCG@10 for all 4 retrievers on the chosen corpus. Sanity check: BM25 near the published BEIR NQ number (~0.3), dense higher.

### Week 2 (5 – 11 Oct): End-to-End + Baseline
- [ ] **P1:** add all poisoned passages into each retriever's single index; cache the raw top-150 for every question × retriever (push via LFS); runtime RRF fusion after masking; record latency and index size.
- [ ] **P1:** pick the generator (3–4B by default; 7–8B only if it runs directly on Ada's GPUs); generation script with the neutral prompt and the generation cache.
- [ ] **P3:** `candidate_pool` (Section 8); one evaluation script with string-match ASR, collateral poisoning, EM/F1 and RAGAS; start the labelling guide.
- [ ] **Checkpoint Wed 8 Oct:** one command runs retrieve → generate → evaluate on 20 target questions for all 4 retrievers.
- [ ] **All (Thu–Sun):** full baseline, no filters: 4 retrievers × 3 rates × 2 attacks on target questions, plus clean questions. Read 10–20 outputs by hand to confirm the metrics make sense.

### Week 3 (12 – 18 Oct): Filters + Labelling
- [ ] **P1:** provenance filter with rank-based normalisation; λ sweep ready.
- [ ] **P2:** outlier filter (pool-wide scoring, z-score sum, phrase rule); θ sweep ready, with and without the phrase rule; centroid distance alone at each poisoning rate. Cascade (Filter 2 → Filter 1).
- [ ] **P1 + P2:** time each filter per query (filter overhead).
- [ ] **P3:** sample the 150 answers, fix the dev/test split, set up the labelling sheet.
- [ ] **All (Wed–Sun):** label 100 answers each (Section 9); settle disagreements.

### Week 4 (19 – 25 Oct): Sweeps, Calibration, Plots
- [ ] **P1 + P2 (Mon–Thu):** run the experiment grid (plan, Section 5): full sweeps at rate 3 for all 4 retrievers; anchors at rates 1 and 5; cascade at 9 anchor combinations; clean questions at every setting.
- [ ] **P3 (Mon–Wed):** human κ; improve the judge on dev, report on test; string-match ASR vs. human labels; RAGAS on the stratified subset.
- [ ] **P3 (Thu–Sun):** plots:
  - **Main figure:** ASR (x-axis) vs. quality (nDCG@5 / answer correctness, y-axis); one curve per retriever, one panel per filter, points = settings.
  - Cascade: 3 × 3 anchor grid vs. each filter alone.
  - Baseline bar chart: ASR per retriever per poisoning rate.
  - Collateral poisoning on clean questions per retriever and rate.
  - Filter 2 with vs. without the phrase rule.
  - Judge vs. human confusion matrix; latency table (search, filter overhead, generation per query); index sizes.
- [ ] **All (Sat–Sun):** findings and error examples (attacks that got through; clean passages wrongly filtered).
- [ ] **Freeze results Sun 25 Oct.** Push final results and labels via LFS. No new experiments after this.

### Week 5 (26 Oct – 1 Nov): Write-Up
- [ ] **Report (Mon–Thu, split by section; one person edits Fri):** introduction, related work, setup, filters and cascade, calibration, results, discussion and limitations (corpus size if we fell back, simulated trust tiers, one generator, one corpus, centroid-distance breakdown).
- [ ] **Repo:** README with setup and how to reproduce each figure; remove dead code; finalise `AI_USAGE.txt`.
- [ ] **Slides (Fri–Sat, if required):** ~10 slides using the architecture diagram and the trade-off curves.
- [ ] **Done Sun 1 Nov.**

### Buffer (2 – 5 Nov)
Fixes only. Submit before 6 Nov.

---

## 11. Operational Risks

| Risk | Mitigation |
|---|---|
| Team members end up with different data | Everything shared goes through the repo (LFS); large files rebuilt from scripts with checksums (Section 3). |
| Git LFS quota exceeded | Only small/medium files in LFS; raw corpus, indexes and embeddings stay out and are rebuilt from scripts or read from the shared Ada copy. |
| Data file shows up as a text pointer | Git LFS wasn't installed before cloning: install it, then `git lfs pull`. |
| Ada job time limits / failures / queue waits | Submit indexing jobs early; save progress in chunks; cache top-150 lists, embeddings and generations. |
| Ada storage fills up | Follow the habits in Section 4; fall back to the ~500k corpus. |
| Someone changes a fixed choice (seed, prompt, model, phrase list) | Ground rule 4: tell the others first and re-run what depends on it. |
