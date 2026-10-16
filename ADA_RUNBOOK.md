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
| A1 data | `ada/run_prepare_data.sh` | REQUESTED | Arya |
| A2 bm25 | `ada/run_index.sh bm25` | REQUESTED | Arya |

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

### A1 data
```
ADA REQUEST A1
Owner: Arya          Needs: A0 done (environment exists)
Run:   git pull && bash ada/run_prepare_data.sh               (on the LOGIN node: it downloads ~735 MB)
Time:  10-30 min (estimate: download at a few MB/s + two passes over 2.68M passages)
Disk:  ~3 GB under $IRE_DATA_DIR (estimate: raw parquet 0.75 GB + corpus.jsonl ~1.5-2 GB)
Success looks like: the output ends with "RESULT: OK" and "SUMMARY A1: OK", with corpus passages ~2.68M,
       200 target + 200 clean (100 per role and fold) and every check PASS.
Send back: $IRE_DATA_DIR/reports/A1.json (paste it in the team chat or commit it to reports/)
           and commit reports/data_card_ada.json (counts, sizes, sources; no absolute paths in it).
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A1_prepare_data.log and the report file if one exists.
```
Details for a non-expert:
1. It downloads BEIR NQ (the pinned Hugging Face copy `BeIR/nq` + `BeIR/nq-qrels`, ~729 MiB) and the `nq_open`
   answers (~4.5 MiB) into `$IRE_DATA_DIR/raw/`,
   then writes `$IRE_DATA_DIR/data/corpus.jsonl` (all ~2.68M passages), `questions_all.jsonl` (every test question
   with a gold passage and an nq_open answer) and `questions.jsonl` (the placeholder 200 target + 200 clean set).
2. The corpus mode comes from `configs/env_ada.yaml` (`corpus.mode: full`). Do not use `--corpus-mode subset`
   yet: the ~500k subset needs the retrievers and is not implemented until after step 5.
3. Before downloading it checks free disk: it stops if the download is over 6 GB or free space would drop
   below 5 GB (change with `--max-download-gb` / `--min-free-gb` if your quota is different).
4. Safe to re-run: the download resumes, and if the files are already built for the same settings the build is
   skipped and only the checks and the report run again (`--force` rebuilds).
   If Hugging Face is blocked on Ada, add `--source ukp` to download the original BEIR `nq.zip` (~475 MiB, but
   that server was very slow from Samarth's machine).
5. Optional, only if asked: `bash ada/run_prepare_data.sh --dev-subset` builds the 1,000-question /
   50,000-passage dev subset in `$IRE_DATA_DIR/data/dev/` and writes `reports/A1_dev.json`.
6. Arya: when your final question set is ready, it replaces `questions.jsonl` (same format, see HANDOFF.md).

### A2 bm25
```
ADA REQUEST A2
Owner: Arya          Needs: A1 done (data/corpus.jsonl and questions exist)
Run:   git pull && bash ada/run_index.sh bm25        (in a CPU job or interactive session; no GPU, no internet)
Time:  15-45 min (estimate: tokenising + indexing 2.68M passages, then ~1,400 test searches)
Disk:  ~1.5-3 GB under $IRE_DATA_DIR/indexes/ada/bm25 (estimate)   RAM: ask for 48 GB (estimate: peak 16-32 GB)
Success looks like: "RESULT: OK" and "SUMMARY A2: OK"; the report shows ~2.68M docs, Recall@10/@100 on
       1,000 questions, and the unique-token passage at rank 1.
Send back: $IRE_DATA_DIR/reports/A2.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A2_bm25.log and the report file if one exists.
```
Details for a non-expert:
1. Do not run it on the login node (it needs several CPU cores and lots of memory for ~15-45 min). Either open an
   interactive session (`srun -p <partition> -c 10 --mem=48G -t 02:00:00 --pty bash`, then the Run command), or
   submit it: `sbatch -p <partition> -c 10 --mem=48G -t 02:00:00 -o $IRE_DATA_DIR/logs/%x-%j.log --wrap "bash ada/run_index.sh bm25"`.
   Use the partition from the top of `ada/common.sh`. A GPU is not needed.
2. It indexes `$IRE_DATA_DIR/data/corpus.jsonl` plus every `poison_*.jsonl` in `$IRE_DATA_DIR/data/poison/`
   (none yet, which is fine). Arya: once your poison shards are in that folder, the same command rebuilds the
   index with them (it notices the new files); job A6 covers that later.
3. Then it measures Recall@10/@100 of gold passages (1,000 seeded questions from `questions_all.jsonl` plus the
   400 in `questions.jsonl`), search latency, determinism, and a unique-token test on a throw-away 50,000-passage
   index (the real index is never modified by the test).
4. Safe to re-run: if the index is already built from the same corpus and settings, the build is skipped and only
   the checks run again. `bash ada/run_index.sh bm25 --force` rebuilds.
5. If it runs out of memory, re-submit with `--mem=96G` and tell Samarth the peak memory from the job accounting.

## Division of labour (suggested; adjust freely)
- Arya: data (A1), BM25 (A2), poison ingestion (A6). She already owns the poison shards.
- Vansh: dense (A3), SPLADE (A4), caches (A5), LLM benchmark (A7), perplexity scores (A8). He already runs the sweeps and needs these outputs.
- Either: A0 setup and A9 smoke test.
