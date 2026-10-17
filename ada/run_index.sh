#!/usr/bin/env bash
# ada/run_index.sh -- build one retriever index and write its report.
#
#   bash ada/run_index.sh bm25      # job A2 (CPU only; no GPU needed)
#   bash ada/run_index.sh dense     # job A3 (needs a GPU; E5 encoding + FAISS)
#   bash ada/run_index.sh splade    # job A4 (added in step 5)
#
# Extra arguments are passed on, e.g. `bash ada/run_index.sh bm25 --force` to rebuild.
# Resumable: an index built from the same corpus, poison shards and settings is reused (only the checks rerun).
# Writes $IRE_DATA_DIR/reports/<job>.json and prints "REPORT: <path>".

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

RETRIEVER="${1:-}"
case "$RETRIEVER" in
  bm25)   JOB="A2" ;;
  dense)  JOB="A3" ;;
  splade) JOB="A4" ;;
  *) echo "usage: bash ada/run_index.sh bm25|dense|splade [--force]" >&2; exit 2 ;;
esac
shift

LOG="$IRE_DATA_DIR/logs/${JOB}_${RETRIEVER}.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/build_index.py --retriever "$RETRIEVER" --report "$JOB" "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY $JOB: OK. Report: $IRE_DATA_DIR/reports/$JOB.json"
elif [[ $STATUS -eq 2 ]]; then
  echo "SUMMARY $JOB: '$RETRIEVER' is not implemented yet. Nothing was run."
else
  echo "SUMMARY $JOB: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_index.sh $RETRIEVER $*"
exit $STATUS
