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

| Job | Script | Status | Owner |
|---|---|---|---|
| A0 setup | `ada/setup_env.sh` | REQUESTED | whoever starts first (Arya or Vansh) |

### A0 setup
```
ADA REQUEST A0
Owner: whoever starts first (Arya or Vansh)          Needs: nothing
Run:   git pull && bash ada/setup_env.sh                    (on the LOGIN node: it needs internet)
       then, inside a GPU session:  bash ada/setup_env.sh --check-only
Time:  10-30 min (estimate)      Disk: ~15 GB under $IRE_DATA_DIR (estimate: environment ~6 GB + models ~8 GB)
Success looks like: the check_env output ends with a RESULT line saying OK and lists a GPU, and the script prints "SUMMARY A0: OK".
Send back: $IRE_DATA_DIR/reports/A0.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A0_setup.log and the report file if one exists.
```
Details for a non-expert:
1. Before the first run: `cp .env.example .env`, then edit `.env`: `IRE_ENV=ada`, `IRE_DATA_DIR=<your big folder>`,
   `HF_HOME=<your big folder>/hf_cache`. `HF_TOKEN` can stay empty (the models are not gated); a token only avoids rate limits.
   Never commit `.env`.
2. If the script stops with "need conda or python3.13", load a module first (e.g. `module load conda`) and run it again.
3. What it does: creates the environment at `$IRE_DATA_DIR/envs/ire` (conda if available, else a `python3.13` venv);
   installs torch 2.6.0 (CUDA 12.4 build) and the pinned `requirements.txt`; downloads the models listed in
   `configs/env_ada.yaml` (`intfloat/e5-base-v2`, `naver/splade-cocondenser-ensembledistil`, `Qwen/Qwen2.5-3B-Instruct`,
   `openai-community/gpt2`) into `HF_HOME`; runs `python scripts/check_env.py --report A0`.
4. The login node has no GPU, so the first run's report shows "GPU: NONE AVAILABLE". That is expected. Run the
   `--check-only` command inside a GPU session (srun/sbatch, settings at the top of `ada/common.sh`) to get the report
   with the GPU, and send that one.
5. It is safe to run again: the environment is reused and downloads resume.
6. Please also fill in the SLURM values at the top of `ada/common.sh` (partition, account, time) and commit them,
   so later jobs can use them.

## Division of labour (suggested; adjust freely)
- Arya: data (A1), BM25 (A2), poison ingestion (A6). She already owns the poison shards.
- Vansh: dense (A3), SPLADE (A4), caches (A5), LLM benchmark (A7), perplexity scores (A8). He already runs the sweeps and needs these outputs.
- Either: A0 setup and A9 smoke test.
