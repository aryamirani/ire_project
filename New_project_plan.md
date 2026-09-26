> **Note:** This plan was updated after meeting the TA. The original proposal is kept in [`submissions/Project Plan.md`](submissions/Project%20Plan.md).

# Project Proposal: Measuring RAG Safety Against Bad Data

**Team:** Arya, Samarth, Vansh

## What is the idea?

### Why is it important?
Right now, the standard tools we use to evaluate RAG systems (like RAGAS) assume that the documents being searched are perfectly safe. But in the real world—whether searching the public web or an internal company wiki—data is often polluted. Attackers can hide malicious text (prompt injections) inside normal-looking documents to hijack the LLM and make it say whatever they want.

Because these attacks use the exact keywords a user is searching for, modern semantic search models often pull the bad documents straight to the top of the results.

### If it were solved, what would improve?
If we can build a proper way to evaluate this, we can measure how different search methods—lexical (BM25), dense, and hybrid—balance finding the right answer (nDCG) against accidentally fetching dangerous text. On top of that, we can test cheap post-retrieval filters and show exactly how much safety they buy and how much answer quality they cost. This would let engineers tune their search pipelines to filter out bad data before it ever reaches the LLM, without hurting normal queries.

### What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it?
Recent research points this out as a major vulnerability. However, most papers either focus entirely on how to make search more accurate or entirely on how to make LLMs more secure. Very few look at how the *search architecture itself* can act as a shield.
Key papers to reference:
* **[PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models (Zou et al., 2024)](https://arxiv.org/abs/2402.07867)**: This paper proves that injecting poisoned text into a dataset easily tricks semantic search models into ranking bad results at the top. We reuse its attack setup and target questions on Natural Questions.
* **[Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG (Guo et al., 2026)](https://arxiv.org/abs/2601.10923)**: This recent benchmark demonstrates how attackers manipulate search results to promote their own documents. It explicitly calls for a comparison between sparse and dense search methods when under attack.
* **[RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2024)](https://arxiv.org/abs/2309.15217)**: This establishes the standard, reference-free RAG metrics (like Answer Relevance), which we use as our answer-quality measure alongside our safety metrics.
* **[Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection (Greshake et al., 2023)](https://arxiv.org/abs/2302.12173)**: This foundational paper demonstrates earlier attempts at indirect prompt injections, proving that LLMs can be compromised by retrieving poisoned third-party data.

---

## What is the plan?

### Dataset
We use **Natural Questions (NQ), BEIR version** as our corpus.
* **Why NQ:** It has real user questions, labels for which passages are relevant (needed for nDCG), and short gold answers (needed for RAGAS answer correctness and for checking whether an attack succeeded). It is also the corpus PoisonedRAG attacks, so we can compare our baseline attack numbers with theirs as a sanity check.
* **Target questions:** We start from the target questions released by PoisonedRAG, each with a correct answer and an attacker-chosen wrong answer, and extend to a few hundred questions.
* **Corpus size:** The full corpus has about 2.7 million Wikipedia passages, which is too slow to index with dense and SPLADE models on our hardware. We keep **every passage relevant to our evaluation questions plus a random sample of the rest (about 200k–500k passages)**. This keeps retrieval hard while making indexing feasible, and we report it clearly.
* **Why not MIND:** MIND is a news-recommendation dataset. It has no search queries, no search relevance labels, and no question–answer pairs, and it only ships titles and abstracts. It could be a stretch second domain (e.g. an LLM news-digest setting) if time allows.

### Poisoning
For each target question, we inject crafted passages that (a) are written to rank highly for that question and (b) carry either a prompt-injection payload or the attacker's wrong answer. We vary the poisoning rate: **1, 3 and 5 poisoned passages per target question**.

### Retrievers
* **Lexical:** BM25.
* **Dense:** a modern embedding model, **E5** or **BGE** (not DPR).
* **Hybrid:** **SPLADE**, and **BM25 + dense** merged with Reciprocal Rank Fusion.

### Generator
A single fixed open LLM reads the top-k retrieved passages and answers the question. Keeping it fixed means any differences in the results come from retrieval and filtering, not from the model.

### Metrics
* **Quality:**
  * **nDCG@10** for retrieval.
  * **RAGAS** (faithfulness, answer relevance, answer correctness) for the generated answers.
* **Safety:**
  * **Poison-in-top-k:** the fraction of queries where at least one poisoned passage appears in the top-k, plus how many appear on average.
  * **Attack Success Rate (ASR):** the fraction of queries where the LLM follows the injected instruction or gives the attacker's wrong answer.

### Defenses: Post-Retrieval Filters
Two filters run between retrieval and generation. Each has one setting that we vary to trace how safety and answer quality trade off:
1. **Provenance-weighted scoring:** Each passage gets a trust score based on its source (e.g. original, curated corpus vs. newly added or user-contributed content). The final ranking score is `λ · relevance + (1 − λ) · trust`. The setting is **λ**.
2. **Outlier / anomaly filtering:** Retrieved passages that look unusual compared with the rest of the top-k are flagged and dropped. Signals include distance from the centre of the top-k embeddings, suspiciously high similarity to the query, and instruction-like language. The setting is the **cutoff threshold**.

For every retriever × filter × setting combination we report nDCG, RAGAS, poison-in-top-k and ASR. The main result is a set of **safety-vs-quality curves** showing where each configuration lands.

### Calibrating the LLM-as-a-Judge
ASR and parts of RAGAS rely on an LLM judge, so we check the judge against human labels before trusting it:
* We hand-label **about 150 (100–200) generated answers** as attacked / not attacked and correct / incorrect. Each answer is labelled by at least two team members.
* We report agreement between the human labellers (Cohen's κ) and between the judge and the human majority label (accuracy, precision, recall, κ).
* If agreement is low, we refine the judge prompt on one part of the labels and re-check it on a held-out part before using it for the full evaluation.

### Execution Phases
1. **Search Pipeline & Backends:** Build the subsampled NQ corpus, index it, and implement BM25, the dense retriever (E5/BGE), and the hybrid retrievers (SPLADE, BM25 + dense).
2. **Dataset Poisoning & Generation:** Create the poisoned corpus at each poisoning rate and run LLM generation for every retriever.
3. **Baseline Evaluation:** Measure quality (nDCG, RAGAS) and safety (poison-in-top-k, ASR) with no defenses.
4. **Defenses:** Implement provenance-weighted scoring and outlier filtering, sweep their settings, and produce the safety-vs-quality curves.
5. **Judge Calibration:** Collect human labels, measure agreement, and finalise the judge used for the reported safety scores.

### Team Split
With three people, we can divide the work cleanly:
* **Person 1:** Sets up the retrievers (BM25, E5/BGE, SPLADE, BM25 + dense) and indexing infrastructure, and implements provenance-weighted scoring.
* **Person 2:** Handles the dataset subsampling and poisoning scripts, runs LLM generation, and implements the outlier/anomaly filter.
* **Person 3:** Builds the evaluation harness (nDCG, RAGAS, poison-in-top-k, ASR), runs the judge calibration, and makes the plots.
* **Everyone:** Contributes to the human labelling for judge calibration.

### Resources & Feasibility
We have all the skills and hardware we need. We only use pretrained retrievers and a fixed LLM, and both defenses are lightweight re-ranking and filtering steps with no training, so everything can run on local hardware (like an M1 Mac) or on free Colab instances. To keep costs down, we subsample the corpus as described above, evaluate on a few hundred target questions, and save retrieval results so that sweeping the filter settings doesn't require re-running search.
