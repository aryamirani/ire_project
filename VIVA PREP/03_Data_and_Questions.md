# 03 -- Data and questions

## 1. Sources
| Source | What we take from it | Why |
|---|---|---|
| **BEIR Natural Questions** (test split) | the corpus (2,681,468 Wikipedia passages, each `{_id, title, text}`), 3,452 test queries, qrels (which passages are relevant = gold) | standard retrieval benchmark; gold passages give retrieval metrics (nDCG, Recall) |
| **nq_open** (validation split) | short answer strings for the same questions | BEIR NQ has no answers, only relevance judgements; ASR and EM/F1 need answer strings |

Downloaded from the pinned Hugging Face copy (`BeIR/nq`, `BeIR/nq-qrels`) because the original UKP server ran at
0.06-0.4 MB/s from Samarth's machine; `--source ukp` reads the original zip in place. A test shows both sources give
identical outputs. The corpus is **streamed** (2.68M passages as Python objects would not fit comfortably in 16 GB).

## 2. Joining questions to answers (`src/ire/data/nq.py`)
BEIR and nq_open have different ids, so they are joined on the **question text**:
```python
def normalize_question(text):
    """Join key for BEIR queries vs nq_open: NFKC, lowercase, drop punctuation, collapse whitespace."""
```
Counts (data card, Ada): 3,452 BEIR test queries, all with gold passages in the corpus (4,201 gold ids, all found);
**2,255** match an nq_open question (answer coverage **65.3%**), all from nq_open's validation split. These 2,255
form `questions_all.jsonl` (`QuestionAllRecord {qid, question, answers, gold_doc_ids}`), the pool from which the
final questions are drawn.

Why not use all 3,452? Without an answer string we cannot compute ASR, EM or F1.

## 3. The final 400 questions (Arya; `src/ire/data/question_set.py`, frozen in `configs/frozen/question_set.json`)
**Eligibility rules** (each prevents a specific measurement artefact):
| Rule | Why |
|---|---|
| shortest gold answer 1-5 words, >= 2 characters | short answers make string-match metrics meaningful |
| no yes/no answers | a yes/no question has a 50% chance answer; ASR would be noise |
| **no answer contained in the question** | otherwise string-match ASR/EM fires just by echoing the question |
| question >= 3 words | very short queries are ambiguous |
| one question per normalised text | duplicates would be double-counted |

**Answer type** (`answer_type`): year | date | number | person | place | other, from the answer string (regexes for
years, months, numbers with units) and the question's wh-word (who -> person, where -> place, how many -> number).
Used to stratify the selection and to pick *type-matched* false target answers.

**Selection** (deterministic, seed 13): sample 400 eligible questions; order them by (answer type, seeded random
key); deal them round-robin into four cells **target/A, target/B, clean/A, clean/B** -> 100 per cell, each with the
same answer-type mix (+/- 1). Result (Ada `questions_ada.json`): per cell about 34 person, 32 other, 12 year,
11 place, 6-7 date, 5 number; 512 gold passage ids, 0 missing from the corpus.

**Roles**
- **target (200)**: poison is written for these questions; ASR is measured only on them.
- **clean (200)**: no poison is aimed at them; they measure *collateral damage* (does poison written for other
  questions reach them?) and *false positives* (does a detector remove good passages?).

**Folds A/B** (100 target + 100 clean each): for cross-fitting. Any data-dependent setting (query-phrase cutoffs,
the best sweep parameter) is chosen on one fold and reported on the other, then swapped and pooled (file 10).

**Frozen**: `configs/frozen/question_set.json` stores `{qid, role, fold, answer_type}` for the 400 plus the sha256 of
the `questions_all.jsonl` it was drawn from (`86ff2915...`). `questions.jsonl` is rebuilt from it, so every machine
gets byte-identical questions (sha256 `0365180a...` on both Ada and the laptop). Frozen files are never edited.

## 4. The corpus files and modes (`src/ire/data/prepare.py`)
| Mode | Content | Used on |
|---|---|---|
| `full` | every BEIR NQ passage, 2,681,468 (1.44 GiB jsonl) | Ada |
| `random` | all gold passages of the 400 questions + seeded uniform random fill, `n_passages` total | local: 5,000 = 512 gold + 4,488 random |
| `subset` | ~500k: gold + hard negatives from each retriever's top-100 | planned fallback; not needed (full fitted), raises NotImplementedError |
The plan allowed a 500k fallback "if the full corpus does not fit the cluster storage". It fitted (indexes ~19 GB on
gnode056's /scratch), so the **full corpus is used** -- the realistic setting (poison competes with 2.68M passages).

Indexed / encoded text = `title + " " + text` everywhere (BM25, dense with "passage: " prefix, SPLADE, the text
store the filters read), so every component sees the same string.

## 5. Local profile caveat
The local corpus is 5,000 passages built around the 400 questions' gold passages, so local recall is near 1.0
(BM25 R@10 0.9494, dense 0.9992, SPLADE 0.9979). That is **not** a result -- the task is trivially easy with 4,488
distractors. Local numbers check correctness only.

## 6. Likely viva questions
- *Why join on question text and not ids?* BEIR and nq_open do not share ids; normalised text is the only link.
  149 nq_open questions occur on several rows (answers merged).
- *Why 65% coverage?* nq_open only keeps questions with short answers (<= 5 tokens); BEIR keeps all NQ test
  queries with relevant passages, including long-answer-only ones.
- *Why stratify by answer type?* So target and clean questions (and both folds) have the same difficulty mix; a
  difference between target and clean is then due to poison, not question type.
- *Why must the answer not appear in the question?* "who sang X with Y" -- if Y is also the answer, any answer
  that repeats the question would score EM/ASR.
- *Why 200 + 200?* Budget: each setting costs 4 retrievers x 400 questions LLM requests before caching; 200 target
  questions give a binary ASR estimate with a Wilson half-width of about +/- 7 points at p = 0.5.
