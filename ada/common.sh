#!/usr/bin/env bash
# ada/common.sh -- sourced by every ada/*.sh job. Do not run directly.
#   * finds the repo root and cds into it
#   * loads .env (IRE_ENV, IRE_DATA_DIR, HF_HOME, HF_TOKEN) without printing secrets
#   * activates the project environment created by ada/setup_env.sh
#   * prints GPU info
# The SLURM settings for the whole project live in this one place (fill in once, then commit).

set -euo pipefail

# ---- SLURM settings (filled in by Arya on 2026-10-26 from `sinfo` / `sacctmgr show assoc`) ----
ADA_PARTITION="${ADA_PARTITION:-u22}"               # default GPU partition (u22-cpu = CPU-only nodes)
ADA_ACCOUNT="${ADA_ACCOUNT:-}"                      # empty: u22 accepts the default account (-A research is
                                                    # rejected on u22-cpu)
# Storage on Ada (checked 2026-10-26): /share1/<user> has a 25 GB / 3,000-FILE quota, so the Python environment
# (tens of thousands of files) must live in home (set IRE_ENV_DIR in .env); GPU nodes have node-local /scratch
# (~1.7 TB) and /ssd_scratch (~480 GB) that are not shared between nodes.
ADA_GPUS="${ADA_GPUS:-1}"
ADA_CPUS="${ADA_CPUS:-10}"
ADA_MEM="${ADA_MEM:-48G}"
ADA_TIME="${ADA_TIME:-06:00:00}"

# ---- repo root and .env ----
IRE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$IRE_REPO"
if [[ ! -f .env ]]; then
  echo "ERROR: $IRE_REPO/.env not found. Run: cp .env.example .env  and edit IRE_DATA_DIR, IRE_ENV=ada" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source <(tr -d '\r' < .env)   # tolerate Windows line endings
set +a
: "${IRE_DATA_DIR:?IRE_DATA_DIR must be set in .env}"
export IRE_ENV="${IRE_ENV:-ada}"
export HF_HOME="${HF_HOME:-$IRE_DATA_DIR/hf_cache}"
export PYTHONUNBUFFERED=1
mkdir -p "$IRE_DATA_DIR/logs" "$IRE_DATA_DIR/reports" "$HF_HOME"

# Environment location (created by ada/setup_env.sh)
IRE_ENV_DIR="${IRE_ENV_DIR:-$IRE_DATA_DIR/envs/ire}"

ire_activate() {
  # Activate the project environment (conda prefix or venv, whichever setup_env.sh made).
  if [[ -d "$IRE_ENV_DIR/conda-meta" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$IRE_ENV_DIR"
  elif [[ -f "$IRE_ENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$IRE_ENV_DIR/bin/activate"
  else
    echo "ERROR: no environment at $IRE_ENV_DIR. Run: bash ada/setup_env.sh" >&2
    return 1
  fi
}

ire_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
  else
    echo "nvidia-smi not found: no GPU visible on this node (fine for CPU-only steps)."
  fi
}

ire_sbatch_hint() {
  # Print the sbatch command to run a job script in batch mode, e.g.  ire_sbatch_hint ada/run_cache.sh dense
  local acct=""
  [[ -n "$ADA_ACCOUNT" ]] && acct="-A $ADA_ACCOUNT "
  echo "sbatch ${acct}-p $ADA_PARTITION --gres=gpu:$ADA_GPUS -c $ADA_CPUS --mem=$ADA_MEM -t $ADA_TIME" \
       "-o $IRE_DATA_DIR/logs/%x-%j.log --wrap \"bash $*\""
}

echo "== IRE job  repo=$IRE_REPO  env=$IRE_ENV  data=$IRE_DATA_DIR  host=$(hostname)  $(date -u +%FT%TZ)"
echo "== GPU:"; ire_gpu_info
