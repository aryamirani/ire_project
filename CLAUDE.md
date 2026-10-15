# CLAUDE.md -- instructions for Claude Code (Samarth's lane)

Project: IRE course project "Measuring RAG Safety Against Bad Data".
Read, in this order, before doing anything:
1. `docs/IRE_Final_Project_Plan.pdf` (the agreed project plan; source of truth for design)
2. `IMPLEMENTATION_PLAN.md` (your step-by-step task list)
3. `HANDOFF.md` (current progress, written for teammates Arya and Vansh)

## Environment (important)
Samarth has NO Ada access. You work on his machine with the **local profile** (`IRE_ENV=local`: tiny corpus, small stand-in models). Heavy jobs are run by Arya and Vansh on Ada (`IRE_ENV=ada`). Read IMPLEMENTATION_PLAN.md sections 0.5 and 5 before starting. Never try to reach Ada or run full-corpus jobs yourself. Write job scripts in `ada/` and requests in `ADA_RUNBOOK.md` and `HANDOFF.md` instead.

## Rules
- Work on ONE step of IMPLEMENTATION_PLAN.md at a time, in order. Start the next step when the previous one is DONE-LOCAL. Do not wait for Ada.
- You may set a step to DONE-LOCAL (tests + local checks pass). Only mark it DONE after a returned Ada report passes `scripts/check_ada_report.py`. Never invent or estimate Ada results.
- After each step: run that step's "Verify" checks, fix failures, then update HANDOFF.md (status, what was built, files, how to run, check results, open issues). Do not commit; leave changes uncommitted and tell the user which files changed and a suggested commit message.
- Never run `git add`, `git commit`, `git push`, `git commit --amend` or any command that changes git history or the index; the user commits and pushes. Read-only git commands (`git status`, `git diff`, `git log`) are fine. Never add Co-Authored-By lines or any Claude/Anthropic attribution anywhere in the repo.
- If a verify check fails and you cannot fix it in a reasonable attempt, mark the step BLOCKED in HANDOFF.md with the exact error and stop. Do not mark a step DONE on a failed or skipped check, and never fake output or numbers.
- My lane only: indexes, retrievers, top-150 cache, `get_doc_vector`, generation harness + cache, cluster filter, perplexity filter + score cache, held-out phrasings tool. Do NOT implement poisoning, query-and-phrase filter, candidate_pool masking, metrics, controls or stats (Arya and Vansh own them). Where you need them, use the stubs/contracts in IMPLEMENTATION_PLAN.md section 2.
- Never change the data formats in section 2 without recording it in HANDOFF.md under "Interface changes".
- Never edit the frozen files in `configs/frozen/` once they exist.
- Keep everything reproducible: fixed seeds, versions pinned in `requirements.txt`, all paths from `configs/paths.yaml`, no absolute paths in code.
- Large files (indexes, caches, vectors, model weights) go under `$IRE_DATA_DIR` (set in `.env`), never into git.
- Ask me before: downloading anything over 2 GB locally, running anything over 30 minutes, deleting files.
- Every Ada job must be one command, resumable, and write a JSON report. Write teammate instructions so a non-expert can follow them.
