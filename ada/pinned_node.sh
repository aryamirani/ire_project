#!/usr/bin/env bash
# ada/pinned_node.sh -- run an ada/*.sh job with IRE_DATA_DIR on the compute node's local /scratch.
#
# Why: /share1 is not mounted on compute nodes and home is too small for every index at once (see
# docs/HANDOFF_ARYA.md). The indexes live in /scratch/$USER/ire_data_dir on ONE node, so every index job must be
# submitted with the same `sbatch -w <node>`. Home keeps the persistent copy of the small files.
#
#   sbatch -w gnode056 ... --wrap "bash ~/ire_project/ada/pinned_node.sh ada/run_index.sh dense"
#
# 1. stage in:  $HOME_DATA/{data,hf_cache,caches} -> $NODE_DATA (rsync, only what changed)
# 2. run:       the job from the node checkout $NODE_REPO (code synced from $HOME_REPO, .env -> $NODE_DATA)
# 3. stage out: $NODE_DATA/{reports,logs,caches,data/poison} -> $HOME_DATA (indexes stay on the node)

set -uo pipefail
HOME_REPO="${HOME_REPO:-$HOME/ire_project}"
NODE_REPO="${NODE_REPO:-$HOME/ire_node}"
HOME_DATA="${HOME_DATA:-$(grep -E '^IRE_DATA_DIR=' "$HOME_REPO/.env" | cut -d= -f2-)}"
NODE_DATA="${NODE_DATA:-/scratch/$USER/ire_data_dir}"
export PATH="$HOME/miniforge3/bin:$PATH"

echo "== pinned_node: host=$(hostname) home_data=$HOME_DATA node_data=$NODE_DATA job=$*"
mkdir -p "$NODE_DATA" && chmod 700 "/scratch/$USER"
# indexes built under the earlier symlink layout move into the node data dir once
if [[ -d "/scratch/$USER/ire_indexes" && ! -e "$NODE_DATA/indexes" ]]; then
  mv "/scratch/$USER/ire_indexes" "$NODE_DATA/indexes"
fi

# node checkout: same code and git metadata as the home checkout, its own .env
mkdir -p "$NODE_REPO"
rsync -a --delete --exclude .env "$HOME_REPO/" "$NODE_REPO/"
sed -e "s#^IRE_DATA_DIR=.*#IRE_DATA_DIR=$NODE_DATA#" -e "s#^HF_HOME=.*#HF_HOME=$NODE_DATA/hf_cache#" \
    "$HOME_REPO/.env" > "$NODE_REPO/.env"
chmod 600 "$NODE_REPO/.env"

for d in data hf_cache caches; do
  [[ -d "$HOME_DATA/$d" ]] && rsync -a "$HOME_DATA/$d/" "$NODE_DATA/$d/"
done
# IRE_STAGE_INDEXES=1: also copy $HOME_DATA/indexes to the node. Used to run the sweep on another node when gnode056
# is full: home holds a copy of the dense index (vectors for the cluster filter) and the bm25 / splade manifests (the
# cache loader checks their fingerprints); the sweep reads the top-150 caches, not the bm25 / splade indexes.
if [[ "${IRE_STAGE_INDEXES:-0}" == 1 && -d "$HOME_DATA/indexes" ]]; then
  echo "== pinned_node: staging indexes from $HOME_DATA/indexes"
  rsync -a "$HOME_DATA/indexes/" "$NODE_DATA/indexes/"
fi

cd "$NODE_REPO"
bash "$@"
STATUS=$?

mkdir -p "$HOME_DATA/reports" "$HOME_DATA/logs" "$HOME_DATA/caches"
rsync -a "$NODE_DATA/reports/" "$HOME_DATA/reports/" 2>/dev/null
rsync -a "$NODE_DATA/logs/" "$HOME_DATA/logs/node_$(hostname)/" 2>/dev/null
[[ -d "$NODE_DATA/caches" ]] && rsync -a "$NODE_DATA/caches/" "$HOME_DATA/caches/"
[[ -d "$NODE_DATA/data/poison" ]] && rsync -a "$NODE_DATA/data/poison/" "$HOME_DATA/data/poison/"
# threshold files written into the node checkout's configs/ (P2, A8): keep a copy next to the reports (commit them)
mkdir -p "$HOME_DATA/reports/thresholds"
for f in query_phrase_thresholds.json perplexity_thresholds.json; do
  [[ -f "$NODE_REPO/configs/$f" ]] && cp -p "$NODE_REPO/configs/$f" "$HOME_DATA/reports/thresholds/$f"
done
# a newly built model snapshot (downloaded on the node) is kept in home too, so the next stage-in is cheap
rsync -a "$NODE_DATA/hf_cache/" "$HOME_DATA/hf_cache/" 2>/dev/null
echo "== pinned_node: job exit $STATUS; reports/logs/caches/poison copied back to $HOME_DATA"
exit $STATUS
