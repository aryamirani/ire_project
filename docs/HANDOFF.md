# HANDOFF -- Samarth's lane (indexes, retrievers, cache, generation, cluster + perplexity filters)

Last updated: (Claude Code fills this in after every step)

## Status board
| Step | Title | Status | Date | Commit |
|---|---|---|---|---|
| 0 | Repo skeleton and environment | TODO | | |
| 1 | Config, paths, logging, data-contract schemas | TODO | | |
| 2 | Load NQ corpus and questions | TODO | | |
| 3 | BM25 index and retriever | TODO | | |
| 4 | Dense index and retriever (+ get_doc_vector) | TODO | | |
| 5 | SPLADE index and retriever | TODO | | |
| 6 | RRF hybrid retriever | TODO | | |
| 7 | Top-150 cache builder | TODO | | |
| 8 | Poison ingestion hook (index poison shards) | TODO | | |
| 9 | Generation harness and cache | TODO | | |
| 10 | Cluster filter | TODO | | |
| 11 | Perplexity filter and score cache | TODO | | |
| 12 | Held-out phrasing tool | TODO | | |
| 13 | End-to-end smoke test | TODO | | |

Status values: TODO, IN PROGRESS, DONE-LOCAL (passes on tiny local profile), DONE (also passed on Ada), REOPENED, BLOCKED.

## Ada requests (for Arya and Vansh)
| Job | After step | Owner | Command | Status (REQUESTED / RUNNING / PASSED / FAILED) | Report |
|---|---|---|---|---|---|
| A0 setup | 0 | | `bash ada/setup_env.sh` | not yet requested | |
| A1 data | 2 | Arya | | not yet requested | |
| A2 bm25 | 3 | Arya | | not yet requested | |
| A3 dense | 4 | Vansh | | not yet requested | |
| A4 splade | 5 | Vansh | | not yet requested | |
| A5 cache | 7 | Vansh | | not yet requested | |
| A6 poison | 8 | Arya | | not yet requested | |
| A7 generate | 9 | Vansh | | not yet requested | |
| A8 perplexity | 11 | Vansh | | not yet requested | |
| A9 smoke | 13 | Arya/Vansh | | not yet requested | |

Teammates: the full instructions for each job are in `ADA_RUNBOOK.md`.

## What teammates can use right now
(Claude Code: list the modules/commands that work, with one-line usage each.)

## Interface changes
(Anything that differs from the contracts in IMPLEMENTATION_PLAN.md section 2. Empty = none.)

## Step log
(Newest first. Per step: what was built, files, how to run, verify results with actual numbers, open issues.)

## Open issues / blockers
(None yet.)
