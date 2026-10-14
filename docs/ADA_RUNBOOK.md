# ADA_RUNBOOK.md -- for teammates running Samarth's jobs on Ada

Samarth does not have Ada access, so the heavy jobs for his part (indexes, caches, perplexity scores, LLM benchmark) are run by Arya and Vansh. You do **not** need to understand the code. Each job is one command, and you send back one JSON report.

Claude Code appends a section below for every job as its step is finished. Check `HANDOFF.md` ("Ada requests") to see which jobs are waiting.

## One-time setup (first person to run anything)
1. `git clone <repo> && cd <repo>` (or `git pull` if already cloned).
2. `cp .env.example .env` and set `IRE_DATA_DIR` to a folder with plenty of space (full corpus + indexes + caches + models need hundreds of GB; ask Samarth for the current estimate in `HANDOFF.md`). Set `IRE_ENV=ada`.
3. `bash ada/setup_env.sh` (creates the environment, downloads models, runs `check_env`). If compute nodes have no internet, run the download part on the login node; the script says which part.
4. Send Samarth the output of `python scripts/check_env.py` (GPU type and free disk).

## How to run any job
- Short jobs: run the command in an interactive GPU session (e.g. `srun` with a GPU) or as a batch job. Use whichever your Ada account normally uses; the scripts read `ada/common.sh`, where the SLURM header (partition, GPU count, time limit) is kept in one place. Fill in the real values once and commit them.
- Jobs are **resumable**: if one dies or times out, run the same command again.
- When it finishes, it prints `REPORT: <path>`. Send Samarth that JSON file (paste in the team chat or commit under `reports/`).
- If it fails: send the last 50 lines of the log plus the report file if one exists. Do not try to fix the code yourself; Samarth's Claude Code will fix it and ask you to re-run.

## Job list
(Claude Code: add one block per job using the request template in IMPLEMENTATION_PLAN.md section 5. Keep the table in sync with HANDOFF.md.)

## Division of labour (suggested; adjust freely)
- Arya: data (A1), BM25 (A2), poison ingestion (A6). She already owns the poison shards.
- Vansh: dense (A3), SPLADE (A4), caches (A5), LLM benchmark (A7), perplexity scores (A8). He already runs the sweeps and needs these outputs.
- Either: A0 setup and A9 smoke test.
