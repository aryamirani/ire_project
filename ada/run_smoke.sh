#!/usr/bin/env bash
# ada/run_smoke.sh -- job A9 (step 13): end-to-end smoke test on 20 questions x 4 retrievers with the real models.
#
#   bash ada/run_smoke.sh          # GPU job on the pinned node, after A5 (caches), A8 (perplexity) and P2 (query-phrase)
#
# cached top-150 -> candidate_pool STUB (first 50, no masking) -> query-phrase, cluster and perplexity filters ->
# first 5 survivors -> Qwen2.5-3B-Instruct (generation cache). Writes $IRE_DATA_DIR/reports/A9.json. ~10-30 min.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/A9_smoke.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/smoke_test.py --report A9 "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A9: OK. Report: $IRE_DATA_DIR/reports/A9.json"
else
  echo "SUMMARY A9: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_smoke.sh $*"
exit $STATUS
