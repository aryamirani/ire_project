---
title: 'IRE Project - Arya, Samarth, Vansh'

---

# Project Proposal: Measuring RAG Safety Against Bad Data

## What is the idea?

### Why is it important? 
Right now, the standard tools we use to evaluate RAG systems (like RAGAS) assume that the documents being searched are perfectly safe. But in the real world—whether searching the public web or an internal company wiki—data is often polluted. Attackers can hide malicious text (prompt injections) inside normal-looking documents to hijack the LLM and make it say whatever they want. 

Because these attacks use the exact keywords a user is searching for, modern semantic search models often pull the bad documents straight to the top of the results. 

### If it were solved, what would improve?

If we can build a proper way to evaluate this, we can measure how different search methods—like lexical (BM25), dense, and hybrid search—balance finding the right answer (nDCG) against accidentally fetching dangerous text. This would allow engineers to tune their search pipelines to naturally filter out bad data before it ever reaches the LLM, without slowing down the system or hurting normal queries.

### What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it?
Recent research points this out as a major vulnerability. However, most papers either focus entirely on how to make search more accurate or entirely on how to make LLMs more secure. Very few look at how the *search architecture itself* can act as a shield.
Key papers to reference:
* **[PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models (Zou et al., 2024)](https://arxiv.org/abs/2402.07867)**: This paper proves that injecting poisoned text into a dataset easily tricks semantic search models into ranking bad results at the top.
* **[Hidden-in-Plain-Text: A Benchmark for Social-Web Indirect Prompt Injection in RAG (Guo et al., 2026)](https://arxiv.org/abs/2601.10923)**: This recent benchmark demonstrates how attackers manipulate search results to promote their own documents. It explicitly calls for a comparison between sparse and dense search methods when under attack.
* **[RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2024)](https://arxiv.org/abs/2309.15217)**: This establishes the standard, reference-free RAG metrics (like Answer Relevance), which we will build on by adding a safety penalty.
* **[Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection (Greshake et al., 2023)](https://arxiv.org/abs/2302.12173)**: This foundational paper demonstrates earlier attempts at indirect prompt injections, proving that LLMs can be compromised by retrieving poisoned third-party data.

---

## What is the plan?

### Execution Phases
1. **Search Pipeline & Backends:** Setting up the vector databases and writing the code for three different search methods to compare: BM25 (lexical), DPR (dense), and SPLADE (hybrid).
2. **Dataset Poisoning & Generation:** Taking a standard dataset (like MIND), injecting prompt injection payloads into a subset of the documents, and running the LLM generation step to see exactly when it gets tricked.
3. **Evaluation & Metrics:** Building the offline evaluation harness. We will define a new "Retrieval Safety" metric, set up an LLM-as-a-judge, and plot the trade-offs to see which search strategy best ignores the bad text while keeping latency low.

### Team Split
With three people, we can divide the work cleanly:
* **Person 1:** Sets up the search backends and indexing infrastructure.
* **Person 2:** Handles the dataset poisoning scripts and runs the LLM generation tests.
* **Person 3:** Builds the evaluation harness and handles all the metrics, testing, and plotting.

### Resources & Feasibility
We have all the skills and hardware we need. Because we are comparing search methods and running evaluations rather than training massive models from scratch, everything can run smoothly on local hardware (like an M1 Mac) or on free Colab instances. The work relies entirely on systems integration, text processing, and standard evaluation metrics, which maps perfectly to the course modules.