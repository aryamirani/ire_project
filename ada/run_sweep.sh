#!/usr/bin/env bash
# ada/run_sweep.sh -- the pilot and the sweeps (scripts/run_sweep.py). One command, resumable.
#
#   bash ada/run_sweep.sh --pilot                       # 500-request pilot (plan 6.1), writes runs/pilot
#   bash ada/run_sweep.sh --tier headline               # core grid at rate 3
#   bash ada/run_sweep.sh --tier rates sweeps adaptive  # more tiers, in the plan's cut order
#   bash ada/run_sweep.sh --list --tier headline        # only print the settings (no GPU work)
#
# Needs the final caches (A5 after the held-out re-run), indexes, thresholds (P2, A8) and the LLM, so submit it
# through ada/pinned_node.sh on gnode056 with a GPU:
#   sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 06:00:00 -o ~/ire_data_dir/logs/sweep-%j.log \
#     --wrap "bash \$HOME/ire_project/ada/pinned_node.sh ada/run_sweep.sh --tier headline"
# Run records go to $IRE_DATA_DIR/reports/runs/<out-name>/ (copied back to home by pinned_node.sh), the report to
# $IRE_DATA_DIR/reports/SWEEP_<out-name>.json. If the job is killed, submit the same command again.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ire_activate

LOG="$IRE_DATA_DIR/logs/SWEEP.log"
exec > >(tee -a "$LOG") 2>&1
echo "== log: $LOG"
ire_gpu_info

set +e
python scripts/run_sweep.py "$@"
STATUS=$?
set -e

echo
if [[ $STATUS -eq 0 ]]; then
  echo "SUMMARY SWEEP: OK. Run records: $IRE_DATA_DIR/reports/runs/"
else
  echo "SUMMARY SWEEP: FAILED or partly failed (exit $STATUS). Send the last 50 lines of $LOG and the SWEEP_*.json report."
fi
echo "Re-run: bash ada/run_sweep.sh $*"
exit $STATUS
