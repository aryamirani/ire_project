#!/usr/bin/env bash
# ada/run_ppl.sh -- job A8 (step 11): GPT-2 perplexity thresholds, held-out check and the score cache for every
# passage that can enter a pool (so the sweeps do no GPT-2 work).
#
#   bash ada/run_ppl.sh            # GPU job; after A5 (needs the current top-150 caches) on the pinned node
#   bash ada/run_ppl.sh --no-pools # thresholds + held-out check only
#
# Writes configs/perplexity_thresholds.json (commit it), $IRE_DATA_DIR/caches/perplexity.sqlite and
# $IRE_DATA_DIR/reports/A8.json. Minutes on a GPU (GPT-2 small).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/A8_perplexity.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/calibrate_perplexity.py --report A8 "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A8: OK. Report: $IRE_DATA_DIR/reports/A8.json; commit configs/perplexity_thresholds.json"
else
  echo "SUMMARY A8: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_ppl.sh $*"
exit $STATUS
