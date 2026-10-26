#!/usr/bin/env bash
# ada/run_make_poison.sh -- Arya's job P1: final question set + poison shards.
#
#   bash ada/run_make_poison.sh                    # after A1 (data) and A3 (dense index, for the stealth cap)
#   bash ada/run_make_poison.sh --no-stealth-cap   # if the dense index does not exist yet (stealth uncapped)
#
# 1. scripts/select_questions.py  writes data/questions.jsonl from configs/frozen/question_set.json
#                                  (replaces Samarth's placeholder; every later cache keys on this file)
# 2. scripts/make_poison.py       writes data/poison/poison_main.jsonl + poison_adaptive.jsonl and checks the
#                                  regenerated targets against configs/frozen/poison_targets.json
# Run it BEFORE A6 (poison ingestion) and A5 (caches). Seconds for step 1; a few minutes for step 2 (one pass
# over corpus.jsonl + dense encoding of 500 stealth candidates). GPU optional.
# Reports: $IRE_DATA_DIR/reports/questions_ada.json and poison_ada.json.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/P1_make_poison.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/select_questions.py
STATUS=$?
if [[ $STATUS -eq 0 ]]; then
  python scripts/make_poison.py "$@"
  STATUS=$?
fi
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY P1: OK. Reports: $IRE_DATA_DIR/reports/questions_ada.json, $IRE_DATA_DIR/reports/poison_ada.json"
else
  echo "SUMMARY P1: FAILED (exit $STATUS). See $LOG."
fi
echo "Re-run: bash ada/run_make_poison.sh $*"
exit $STATUS
