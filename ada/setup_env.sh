#!/usr/bin/env bash
# ada/setup_env.sh -- job A0: create the environment, install pinned packages, download models,
# and write the environment report $IRE_DATA_DIR/reports/A0.json.
#
#   bash ada/setup_env.sh               # full setup (needs internet: run on the LOGIN node)
#   bash ada/setup_env.sh --check-only  # only re-run the check (run inside a GPU session so the GPU is reported)
#
# Idempotent and resumable: an existing environment is reused, pip skips installed packages,
# cached models are skipped and partial downloads resume. Safe to run again after any failure.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LOG="$IRE_DATA_DIR/logs/A0_setup.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1

if [[ $CHECK_ONLY -eq 0 ]]; then
  # ---- 1. environment (internet needed) ----
  if [[ -d "$IRE_ENV_DIR/conda-meta" || -f "$IRE_ENV_DIR/bin/activate" ]]; then
    echo "== [1/4] environment exists: $IRE_ENV_DIR (reusing)"
  elif command -v conda >/dev/null 2>&1; then
    echo "== [1/4] creating conda env (python=$PYTHON_VERSION) at $IRE_ENV_DIR"
    conda create -y -p "$IRE_ENV_DIR" "python=$PYTHON_VERSION"
  elif command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
    echo "== [1/4] creating venv with python$PYTHON_VERSION at $IRE_ENV_DIR"
    mkdir -p "$(dirname "$IRE_ENV_DIR")"
    "python$PYTHON_VERSION" -m venv "$IRE_ENV_DIR"
  else
    echo "ERROR: need conda or python$PYTHON_VERSION. Try 'module load conda' (or a python $PYTHON_VERSION module)," >&2
    echo "       or install Miniconda into \$IRE_DATA_DIR, then re-run this script." >&2
    exit 1
  fi
  ire_activate

  # ---- 2. pinned packages (internet needed) ----
  echo "== [2/4] installing torch==$TORCH_VERSION from $TORCH_INDEX_URL and requirements.txt"
  python -m pip install --upgrade pip
  python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
  python -m pip install -r requirements.txt
  python -m pip install -e . --no-deps

  # ---- 3. models into HF_HOME (internet needed) ----
  echo "== [3/4] downloading models for IRE_ENV=$IRE_ENV into $HF_HOME"
  python scripts/download_models.py
  echo "== Java check: not required (BM25 uses bm25s, not pyserini)."
else
  echo "== --check-only: skipping environment, install and download"
  ire_activate
fi

# ---- 4. environment report ----
echo "== [4/4] python scripts/check_env.py --report A0"
set +e
python scripts/check_env.py --report A0
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A0: OK. Report: $IRE_DATA_DIR/reports/A0.json"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "NOTE: no GPU on this node. To report the GPU, re-run inside a GPU session:  bash ada/setup_env.sh --check-only"
  fi
else
  echo "SUMMARY A0: FAILED (see 'RESULT' above). Send the last 50 lines of $LOG and the report file."
fi
echo "Re-run: bash ada/setup_env.sh   (or: bash ada/setup_env.sh --check-only)"
exit $STATUS
