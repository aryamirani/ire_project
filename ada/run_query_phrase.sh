#!/usr/bin/env bash
# ada/run_query_phrase.sh -- Arya's job P2: calibrate the query-and-phrase filter on Ada (plan 4.1).
#
#   bash ada/run_query_phrase.sh            # GPU job; needs current A5 caches (built after A6) and the dense index
#
# Writes configs/query_phrase_thresholds.json in the repo (commit it: the sweeps read it) and
# $IRE_DATA_DIR/reports/query_phrase_ada.json (cutoffs, cross-fold flag rates, phrase-rule FPR on 20,000 random
# corpus passages, detection sanity check at rate 5). Minutes.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/P2_query_phrase.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/calibrate_query_phrase.py --detection --corpus-sample 20000 "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY P2: OK. Report: $IRE_DATA_DIR/reports/query_phrase_ada.json; commit configs/query_phrase_thresholds.json"
else
  echo "SUMMARY P2: FAILED (exit $STATUS). See $LOG."
fi
echo "Re-run: bash ada/run_query_phrase.sh $*"
exit $STATUS
