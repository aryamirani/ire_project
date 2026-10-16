#!/usr/bin/env bash
# ada/run_prepare_data.sh -- job A1: download BEIR NQ + nq_open and build the corpus and question files.
#
#   bash ada/run_prepare_data.sh                 # full corpus (corpus.mode in configs/env_ada.yaml)
#   bash ada/run_prepare_data.sh --dev-subset    # optional extra: 1,000 questions + 50,000 passages in data/dev/
#
# Needs internet for the download (~735 MB): run it on the LOGIN node (no GPU needed; ~10-30 min).
# Add --source ukp to use the original BEIR nq.zip if Hugging Face is blocked.
# Resumable: the download resumes, and a finished build is skipped on re-run (only the checks and the report
# run again). Writes $IRE_DATA_DIR/reports/A1.json (A1_dev.json for --dev-subset) and prints "REPORT: <path>".

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/A1_prepare_data.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

REPORT_NAME="A1"
for arg in "$@"; do [[ "$arg" == "--dev-subset" ]] && REPORT_NAME="A1_dev"; done

set +e
python scripts/prepare_data.py --report "$REPORT_NAME" "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A1: OK. Report: $IRE_DATA_DIR/reports/$REPORT_NAME.json"
else
  echo "SUMMARY A1: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_prepare_data.sh $*"
exit $STATUS
