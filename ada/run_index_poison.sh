#!/usr/bin/env bash
# ada/run_index_poison.sh -- job A6: ingest Arya's poison shards into the BM25, dense and SPLADE indexes.
#
#   bash ada/run_index_poison.sh            # shards: every poison_*.jsonl in $IRE_DATA_DIR/data/poison/
#   bash ada/run_index_poison.sh --force    # re-ingest even if the indexes already hold exactly these shards
#
# Needs A2, A3, A4 (the three indexes) and Arya's real shards (including the adaptive shard) in data/poison/.
# Needs a GPU (dense and SPLADE encode the poison passages and the questions).
# BM25 is rebuilt with the shards; dense and SPLADE keep their clean rows and only append the poison passages.
# Before the first ingestion it saves each clean index's top-150 for every question as the masking reference.
# Idempotent: re-running with the same shards does nothing; changed shards are re-ingested.
# AFTER THIS JOB EVERY TOP-150 CACHE IS STALE: run A5 (bash ada/run_cache.sh all) afterwards.
# Writes $IRE_DATA_DIR/reports/A6.json and prints "REPORT: <path>".

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/A6_index_poison.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/index_poison.py --retriever all --check-masking --report A6 "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A6: OK. Report: $IRE_DATA_DIR/reports/A6.json"
  echo "NEXT: run A5 now (bash ada/run_cache.sh all): the old caches no longer match the indexes."
else
  echo "SUMMARY A6: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_index_poison.sh $*"
exit $STATUS
