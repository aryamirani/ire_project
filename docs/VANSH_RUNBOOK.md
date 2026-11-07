# VANSH_RUNBOOK -- commands for the rest of the project, in order (written 2026-11-04)

Everything here is code that exists in the repo but was **written without being run**. Expect small fixes on the
first run; the local steps exist to find them before any Ada time is spent. Prefix every local command on macOS with
`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` and use the project venv.

## 0. One-time local setup (laptop)
Python 3.11+ is required (macOS `python3` is 3.9: do not use it).
```
git pull
uv venv --python 3.13 .venv && source .venv/bin/activate && python --version      # must print 3.13.x
uv pip install torch==2.6.0 && uv pip install -r requirements.txt matplotlib && uv pip install -e . --no-deps
cp .env.example .env     # IRE_ENV=local; IRE_DATA_DIR="<repo>/ire_data_dir" (git-ignored; allowed inside the repo); HF_HOME="<that>/hf_cache"
pip freeze | grep -i matplotlib       # then pin that version in requirements.txt
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
python scripts/download_models.py
python scripts/prepare_data.py          # then the local chain, only needed for the local dry run (section 2):
python scripts/build_index.py --retriever bm25 && python scripts/build_index.py --retriever dense && python scripts/build_index.py --retriever splade
python scripts/make_poison.py && python scripts/index_poison.py && python scripts/build_cache.py --retriever all
```
Do NOT run `calibrate_query_phrase.py` locally: it overwrites `configs/query_phrase_thresholds.json` (the Ada values).

## 1. Tests (no models, seconds)
```
pytest tests/test_pool.py tests/test_metrics.py tests/test_stats.py tests/test_controls.py tests/test_sweep.py tests/test_analysis.py -q
pytest -q                                # whole suite (the 41 tests of this lane + config passed on 2026-11-08)
python scripts/run_sweep.py --list --tier headline rates sweeps adaptive optional    # prints 384 settings, loads nothing
```

## 2. Local dry run on the tiny corpus (finds bugs before Ada)
```
python scripts/run_sweep.py --pilot                       # 10 target + 10 clean questions, fact only, local 1.5B stand-in
python scripts/analyze.py --run-name pilot
python scripts/make_plots.py --run-name pilot
```
Local numbers are for correctness only. Look at one record: `head -c 1500 $IRE_DATA_DIR/reports/runs/pilot/bm25__fact__r3__none.jsonl`.

## 3. Ada (all through Arya's account on gnode056; see VANSH_START.md section 2 for the sbatch pattern)
Get the code there (`git pull` in `~/ire_project` on Ada, or ask Arya), then:

**A7 -- LLM benchmark (decide the batch size).** Does not depend on any new code:
```
sbatch -p u22 -w gnode056 --gres=gpu:1 -c 4 --mem=11G -t 01:00:00 -o ~/ire_data_dir/logs/A7-%j.log \
  --wrap "bash \$HOME/ire_project/ada/pinned_node.sh ada/run_generate_benchmark.sh"
python scripts/check_ada_report.py A7 --file ~/ire_data_dir/reports/A7.json
```
Then set `compute.generate_batch_size` in `configs/env_ada.yaml` (or pass `--batch-size` to the sweep). Rule of thumb from
HANDOFF: generate in batches and treat the generation cache as the record.

**Pilot (500 requests, plan 6.1):**
```
sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 01:00:00 -o ~/ire_data_dir/logs/pilot-%j.log \
  --wrap "bash \$HOME/ire_project/ada/pinned_node.sh ada/run_sweep.sh --pilot"
```
Read `~/ire_data_dir/reports/SWEEP_pilot.json`: failures, `generation_requests`, `generation_cache_hit_rate`, runtime.
Run `python scripts/analyze.py --run-name pilot` on the copied-back records and look at `headline.csv` for sanity
(oracle ASR near 0, no-filter ASR clearly above the no-poison floor).

**Headline sweep (after the decision rule is committed with everyone's agreement):**
```
sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 06:00:00 -o ~/ire_data_dir/logs/sweep-%j.log \
  --wrap "bash \$HOME/ire_project/ada/pinned_node.sh ada/run_sweep.sh --tier headline"
```
Resubmit the same line after a time limit: finished settings are skipped, answers come from the generation cache.
More tiers, one after another (each is its own job; Ada allows 5 queued + running):
```
... ada/run_sweep.sh --tier rates
... ada/run_sweep.sh --tier sweeps
... ada/run_sweep.sh --tier adaptive
... ada/run_sweep.sh --tier optional        # url; the first thing to cut
```
Do not start two sweep jobs at the same time: they write the same generation cache.

## 4. After the sweeps (laptop or Ada login node; CPU only)
```
python scripts/analyze.py                      # runs/main -> analysis/main/*.csv   (summary, headline, crossfit, mechanism, complementarity)
python scripts/make_plots.py                   # analysis/main/fig_*.png
```
`reports/runs/` and `reports/analysis/` are copied back to `~/ire_data_dir/reports/` by `pinned_node.sh`; `scp` them to the laptop.

## 5. Human calibration (plan 7), in parallel with the sweeps once the headline records exist
```
python scripts/calibration_sample.py           # needs run records + text store (run it on the node, or locally on pilot data to test)
# send each person ONLY their own answers_<name>.csv and passages_<name>.csv; keep key.json private
# everyone fills the `label` column (answers: 1 = attack succeeded, 0 = not; passages: poison / clean / ambiguous)
python scripts/calibration_agreement.py        # kappa per pair, judge accuracy on dev and test; writes disagreements.csv
# resolve disagreements in disagreements.csv (`final` column), re-run the script
```

## 6. Report numbers come from
`analysis/main/headline.csv` (Holm-adjusted, `helps`, `beats_random`, `oracle_captured`), `complementarity.csv` (RQ3),
`summary.csv` (rates, sweeps, TPR/FPR/AUC, backfill, filter ms, collateral), `crossfit.csv`, `mechanism.csv`,
`agreement.json`, the figures, and `SWEEP_main.json` (requests, cache hit rate, runtime).
