#!/usr/bin/env bash
# ada/run_cache.sh -- job A5: build the top-150 retrieval caches for the final question set.
#
#   bash ada/run_cache.sh all          # bm25, dense, splade, then rrf (fused from the bm25 + dense caches)
#   bash ada/run_cache.sh dense        # one retriever
#   bash ada/run_cache.sh all --force  # rebuild even if current
#
# Needs a GPU (dense and SPLADE encode the questions). Needs A2, A3, A4 (indexes), A6 (poison ingested) and Arya's
# final data/questions.jsonl: a cache built before those is only a timing test and must be rebuilt (the loader
# refuses a cache whose index fingerprint or questions file changed).
# Resumable: an interrupted build continues after the last written question; current caches are skipped.
# Writes $IRE_DATA_DIR/reports/A5.json and prints "REPORT: <path>".

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

RETRIEVER="${1:-all}"
case "$RETRIEVER" in
  bm25|dense|splade|rrf|all) ;;
  *) echo "usage: bash ada/run_cache.sh bm25|dense|splade|rrf|all [--force]" >&2; exit 2 ;;
esac
shift || true

LOG="$IRE_DATA_DIR/logs/A5_cache.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"

set +e
python scripts/build_cache.py --retriever "$RETRIEVER" --report A5 "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY A5: OK. Report: $IRE_DATA_DIR/reports/A5.json"
else
  echo "SUMMARY A5: FAILED (exit $STATUS). Send the last 50 lines of $LOG and the report file if one exists."
fi
echo "Re-run: bash ada/run_cache.sh $RETRIEVER $*"
exit $STATUS
