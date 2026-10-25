#!/usr/bin/env bash
# ada/run_generate_benchmark.sh -- job A7: benchmark the real generator (Qwen/Qwen2.5-3B-Instruct, fp16).
#
#   bash ada/run_generate_benchmark.sh                          # batch sizes 1, 8, 16, 32
#   bash ada/run_generate_benchmark.sh --batch-sizes 1,8,16     # other batch sizes
#
# Needs a GPU, A0 (environment + model downloaded) and A1 (data). Uses a TEMPORARY generation cache (the real one
# is untouched). Builds the document text store $IRE_DATA_DIR/data/text_store.sqlite on first use (~2.68M rows).
# Writes $IRE_DATA_DIR/reports/A7.json and prints "REPORT: <path>". Safe to re-run.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/A7_generate_benchmark.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

ARGS=("$@")
[[ " $* " == *" --batch-sizes "* ]] || ARGS=(--batch-sizes 1,8,16,32 "${ARGS[@]}")

set +e
python scripts/generate_benchmark.py --report A7 "${ARGS[@]}"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A7: OK. Report: $IRE_DATA_DIR/reports/A7.json"
else
  echo "SUMMARY A7: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_generate_benchmark.sh $*"
exit $STATUS
