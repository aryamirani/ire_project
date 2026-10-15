# ada/ -- job scripts for the Ada cluster

These scripts are run by Arya and Vansh (Samarth has no Ada access).
**Read `../ADA_RUNBOOK.md` first**: it has the one-time setup, the job list, the exact commands,
expected runtimes and what to send back.

- `common.sh` -- sourced by every job: loads `.env`, activates the environment, prints GPU info.
  The SLURM settings (partition, GPUs, CPUs, memory, time) are at the top of this file; fill them in once.
- `setup_env.sh` -- job A0: environment, pinned packages, model downloads, `reports/A0.json`.

Every job is one command, resumable (re-run the same command after a failure), writes its outputs under
`$IRE_DATA_DIR`, and ends by printing `REPORT: <path to JSON>`.
