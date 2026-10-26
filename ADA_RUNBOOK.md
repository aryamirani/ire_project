# ADA_RUNBOOK.md -- for teammates running Samarth's jobs on Ada

Samarth does not have Ada access, so the heavy jobs for his part (indexes, caches, perplexity scores, LLM benchmark) are run by Arya and Vansh. You do **not** need to understand the code. Each job is one command, and you send back one JSON report.

Claude Code appends a section below for every job as its step is finished. Check `HANDOFF.md` ("Ada requests") to see which jobs are waiting.

## One-time setup (first person to run anything)
1. `git clone <repo> && cd <repo>` (or `git pull` if already cloned).
2. `cp .env.example .env` and set `IRE_DATA_DIR` to a folder with plenty of space (full corpus + indexes + caches + models need hundreds of GB; ask Samarth for the current estimate in `HANDOFF.md`). Set `IRE_ENV=ada`.
3. `bash ada/setup_env.sh` (creates the environment, downloads models, runs `check_env`). If compute nodes have no internet, run the download part on the login node; the script says which part.
4. Send Samarth the output of `python scripts/check_env.py` (GPU type and free disk).

## How to run any job
- Submit it as described in "How jobs run on Ada" below (pinned node gnode056 through `ada/pinned_node.sh`, `-c 10 --mem=29G` for big jobs). The SLURM defaults live in `ada/common.sh`.
- Jobs are **resumable**: if one dies or times out, run the same command again.
- When it finishes, it prints `REPORT: <path>`. Send Samarth that JSON file (paste in the team chat or commit under `reports/`).
- If it fails: send the last 50 lines of the log plus the report file if one exists. Do not try to fix the code yourself; Samarth's Claude Code will fix it and ask you to re-run.

## Job list
(Claude Code: add one block per job using the request template in IMPLEMENTATION_PLAN.md section 5. Keep the table in sync with HANDOFF.md.)

The submit commands in the job blocks below follow "How jobs run on Ada" (next section).

| Job | Script | Status | Owner |
|---|---|---|---|
| A0 setup | `ada/setup_env.sh` | PASSED | Arya |
| A1 data | `ada/run_prepare_data.sh` | PASSED | Arya |
| A2 bm25 | `ada/run_index.sh bm25` | PASSED | Arya |
| A3 dense | `ada/run_index.sh dense` | PASSED (run by Arya) | Vansh |
| A4 splade | `ada/run_index.sh splade` | PASSED (run by Arya) | Vansh |
| A5 cache | `ada/run_cache.sh all --batch-size 1` | PASSED (run by Arya) | Vansh |
| A6 poison | `ada/run_index_poison.sh` | PASSED | Arya |
| A7 generate | `ada/run_generate_benchmark.sh` | not run yet | Vansh |
| A8 perplexity | `ada/run_ppl.sh` | PASSED (run by Arya with step 11) | Vansh |
| A9 smoke | `ada/run_smoke.sh` | PASSED | Arya |

### A0 setup
```
ADA REQUEST A0
Owner: whoever starts first (Arya or Vansh)          Needs: nothing
Run:   git pull && bash ada/setup_env.sh                    (on the LOGIN node: it needs internet)
       then, inside a GPU session:  bash ada/setup_env.sh --check-only
Time:  10-30 min (estimate)      Disk: ~15 GB under $IRE_DATA_DIR (estimate: environment ~6 GB + models ~8 GB)
Success looks like: the check_env output ends with a RESULT line saying OK and lists a GPU, and the script prints "SUMMARY A0: OK".
Send back: $IRE_DATA_DIR/reports/A0.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A0_setup.log and the report file if one exists.
```
Details for a non-expert:
1. Before the first run: `cp .env.example .env`, then edit `.env`: `IRE_ENV=ada`, `IRE_DATA_DIR=<your big folder>`,
   `HF_HOME=<your big folder>/hf_cache`. `HF_TOKEN` can stay empty (the models are not gated); a token only avoids rate limits.
   Never commit `.env`.
2. If the script stops with "need conda or python3.13", load a module first (e.g. `module load conda`) and run it again.
3. What it does: creates the environment at `$IRE_DATA_DIR/envs/ire` (conda if available, else a `python3.13` venv);
   installs torch 2.6.0 (CUDA 12.4 build) and the pinned `requirements.txt`; downloads the models listed in
   `configs/env_ada.yaml` (`intfloat/e5-base-v2`, `naver/splade-cocondenser-ensembledistil`, `Qwen/Qwen2.5-3B-Instruct`,
   `openai-community/gpt2`) into `HF_HOME`; runs `python scripts/check_env.py --report A0`.
4. The login node has no GPU, so the first run's report shows "GPU: NONE AVAILABLE". That is expected. Run the
   `--check-only` command inside a GPU session (srun/sbatch, settings at the top of `ada/common.sh`) to get the report
   with the GPU, and send that one.
5. It is safe to run again: the environment is reused and downloads resume.
6. Please also fill in the SLURM values at the top of `ada/common.sh` (partition, account, time) and commit them,
   so later jobs can use them.

### A1 data
```
ADA REQUEST A1
Owner: Arya          Needs: A0 done (environment exists)
Run:   git pull && bash ada/run_prepare_data.sh               (on the LOGIN node: it downloads ~735 MB)
Time:  10-30 min (estimate: download at a few MB/s + two passes over 2.68M passages)
Disk:  ~3 GB under $IRE_DATA_DIR (estimate: raw parquet 0.75 GB + corpus.jsonl ~1.5-2 GB)
Success looks like: the output ends with "RESULT: OK" and "SUMMARY A1: OK", with corpus passages ~2.68M,
       200 target + 200 clean (100 per role and fold) and every check PASS.
Send back: $IRE_DATA_DIR/reports/A1.json (paste it in the team chat or commit it to reports/)
           and commit reports/data_card_ada.json (counts, sizes, sources; no absolute paths in it).
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A1_prepare_data.log and the report file if one exists.
```
Details for a non-expert:
1. It downloads BEIR NQ (the pinned Hugging Face copy `BeIR/nq` + `BeIR/nq-qrels`, ~729 MiB) and the `nq_open`
   answers (~4.5 MiB) into `$IRE_DATA_DIR/raw/`,
   then writes `$IRE_DATA_DIR/data/corpus.jsonl` (all ~2.68M passages), `questions_all.jsonl` (every test question
   with a gold passage and an nq_open answer) and `questions.jsonl` (the placeholder 200 target + 200 clean set).
2. The corpus mode comes from `configs/env_ada.yaml` (`corpus.mode: full`). Do not use `--corpus-mode subset`
   yet: the ~500k subset needs the retrievers and is not implemented until after step 5.
3. Before downloading it checks free disk: it stops if the download is over 6 GB or free space would drop
   below 5 GB (change with `--max-download-gb` / `--min-free-gb` if your quota is different).
4. Safe to re-run: the download resumes, and if the files are already built for the same settings the build is
   skipped and only the checks and the report run again (`--force` rebuilds).
   If Hugging Face is blocked on Ada, add `--source ukp` to download the original BEIR `nq.zip` (~475 MiB, but
   that server was very slow from Samarth's machine).
5. Optional, only if asked: `bash ada/run_prepare_data.sh --dev-subset` builds the 1,000-question /
   50,000-passage dev subset in `$IRE_DATA_DIR/data/dev/` and writes `reports/A1_dev.json`.
6. Arya: when your final question set is ready, it replaces `questions.jsonl` (same format, see HANDOFF.md).

### A2 bm25
```
ADA REQUEST A2
Owner: Arya          Needs: A1 done (data/corpus.jsonl and questions exist)
Run:   git pull && bash ada/run_index.sh bm25        (in a CPU job or interactive session; no GPU, no internet)
Time:  15-45 min (estimate: tokenising + indexing 2.68M passages, then ~1,400 test searches)
Disk:  ~1.5-3 GB under $IRE_DATA_DIR/indexes/ada/bm25 (estimate)   RAM: 29 GB (the QOS maximum; it built in 334 s on Ada)
Success looks like: "RESULT: OK" and "SUMMARY A2: OK"; the report shows ~2.68M docs, Recall@10/@100 on
       1,000 questions, and the unique-token passage at rank 1.
Send back: $IRE_DATA_DIR/reports/A2.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A2_bm25.log and the report file if one exists.
```
Details for a non-expert:
1. Do not run it on the login node (it needs several CPU cores and lots of memory for ~15-45 min). Either open an
   interactive session (`srun -p u22 -w gnode056 -c 10 --mem=29G -t 02:00:00 --pty bash`, then the Run command), or
   submit it: `sbatch -p u22 -w gnode056 -c 10 --mem=29G -t 02:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_index.sh bm25"`.
   A GPU is not needed (BM25 does not import transformers).
2. It indexes `$IRE_DATA_DIR/data/corpus.jsonl` plus every `poison_*.jsonl` in `$IRE_DATA_DIR/data/poison/`
   (none yet, which is fine). Poison is added afterwards by job A6 (`ada/run_index_poison.sh`).
3. Then it measures Recall@10/@100 of gold passages (1,000 seeded questions from `questions_all.jsonl` plus the
   400 in `questions.jsonl`), search latency, determinism, and a unique-token test on a throw-away 50,000-passage
   index (the real index is never modified by the test).
4. Safe to re-run: if the index is already built from the same corpus and settings, the build is skipped and only
   the checks run again. `bash ada/run_index.sh bm25 --force` rebuilds.
5. 29 GB is the most our QOS allows; A2 fitted (see "How jobs run on Ada").

### A3 dense
```
ADA REQUEST A3
Owner: Vansh          Needs: A0 (environment + intfloat/e5-base-v2 downloaded) and A1 (data/corpus.jsonl)
Run:   git pull && bash ada/run_index.sh dense        (in a GPU job; no internet needed)
Time:  1-2 h (estimate: encoding 2.68M passages with e5-base-v2 on one GPU; the exact Flat index adds minutes)
Disk:  ~13 GB under $IRE_DATA_DIR/indexes/ada/dense (estimate: float16 vectors 2.68M x 768 x 2 B = 4.1 GB +
       Flat index in float32 2.68M x 768 x 4 B = 8.2 GB)
RAM:   29 GB (QOS maximum; peak ~12-16 GB = 8.2 GB Flat index + ~1.5 GB passage texts + model/overhead)
Success looks like: "RESULT: OK" and "SUMMARY A3: OK"; the report shows ~2.68M docs, Recall@10/@100 on 1,000
       questions (next to BM25's on the same questions), stored-vs-re-encoded cosine > 0.999 and norms 1 +/- 1e-3.
Send back: $IRE_DATA_DIR/reports/A3.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A3_dense.log and the report file if one exists.
```
Details for a non-expert:
1. Run it as a GPU batch job, e.g.
   `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 06:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_index.sh dense"`
   or the same command inside an interactive `srun -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 04:00:00 --pty bash`.
2. It encodes every passage of `$IRE_DATA_DIR/data/corpus.jsonl` (+ any `poison_*.jsonl` in `data/poison/`) with
   `intfloat/e5-base-v2` (`passage: ` prefix, fp16), stores the vectors and builds an exact FAISS Flat
   inner-product index (settings in `configs/env_ada.yaml`).
3. **If the job hits its time limit or dies, just submit the same command again**: encoding resumes from the last
   checkpoint (every 50,000 passages) instead of starting over. The log says "resuming an interrupted build at row N".
4. If the GPU runs out of memory the batch size is halved automatically; nothing to do.
5. Safe to re-run after success: the build is skipped and only the checks run. `--force` rebuilds.
6. A2 (BM25) does not have to be finished first; if it is, the report also lists BM25 recall on the same questions.

### A4 splade
```
ADA REQUEST A4
Owner: Vansh          Needs: A0 (environment + SPLADE model downloaded) and A1 (data/corpus.jsonl)
Run:   git pull && bash ada/run_index.sh splade       (in a GPU job; no internet needed)
Time:  1-3 h (estimate: SPLADE encoding of 2.68M passages; 260 passages/s on Samarth's throttled 6 GB laptop
       GPU, so a cluster GPU should be several times faster)
Disk:  ~6 GB final + ~3 GB temporary checkpoint files while building, under $IRE_DATA_DIR/indexes/ada/splade
       (estimate scaled from the local index: 147.8 non-zeros per passage -> ~396M non-zeros; 8 bytes each,
       stored twice: document matrix + inverted index)
RAM:   29 GB (QOS maximum; peak ~12-16 GB while the two matrices are assembled)
VRAM:  ~1.2 GB at batch 64 locally (fp16); the batch halves itself on out-of-memory
Success looks like: "RESULT: OK" and "SUMMARY A4: OK"; the report shows ~2.68M docs, non-zeros per document
       and per query, Recall@10/@100 on 1,000 questions (next to BM25 and dense if their indexes exist), the
       exactness check and self-retrieval at rank 1.
Send back: $IRE_DATA_DIR/reports/A4.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A4_splade.log and the report file if one exists.
If the job is killed (time limit), submit the same command again: encoding resumes from the last checkpoint.
```
Details for a non-expert:
1. Run it as a GPU batch job, e.g.
   `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 06:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_index.sh splade"`
   or inside an interactive `srun -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 04:00:00 --pty bash`.
2. It encodes every passage of `$IRE_DATA_DIR/data/corpus.jsonl` (+ any `poison_*.jsonl` in `data/poison/`)
   with `naver/splade-cocondenser-ensembledistil` (256 tokens per passage, 64 per query, fp16) and stores the
   sparse vectors without pruning, so search is exact.
3. **If the job hits its time limit or dies, submit the same command again**: encoding resumes from the last
   checkpoint (every 50,000 passages). The log says "resuming an interrupted build at row N".
4. Safe to re-run after success: the build is skipped and only the checks run. `--force` rebuilds.
5. A2 and A3 do not have to be finished first; if they are, the report also lists their recall on the same
   questions.

### A5 cache
```
ADA REQUEST A5
Owner: Vansh          Needs: A2, A3, A4, and A6 (poison ingested) and Arya's final questions.jsonl
Run:   git pull && bash ada/run_cache.sh all            (in a GPU job; no internet needed)
Time:  10-30 min (estimate: 400 questions x 3 retrievers over 2.68M passages, exact search, + spot checks)
Disk:  ~10 MB under $IRE_DATA_DIR/caches/ada (about 2 MB per retriever for 400 questions, measured locally)
RAM:   29 GB (QOS maximum; the exact dense Flat index alone is ~8.2 GB; SPLADE and BM25 are memory-mapped)
Success looks like: "RESULT: OK" and "SUMMARY A5: OK"; 400 rows per retriever, dense rows exactly 150, every
       spot check 20/20, and NO "index contains poison documents" warning from check_ada_report.py.
Send back: $IRE_DATA_DIR/reports/A5.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A5_cache.log and the report file if one exists.
IMPORTANT: a cache built before A6 (poison in the indexes) or before Arya's final question set is only a
timing test and must be rebuilt. The loader refuses such a cache automatically once the index fingerprint or
the questions file changes (StaleCacheError), so a stale cache cannot be used by mistake.
```
Details for a non-expert:
1. Run it as a GPU batch job, e.g.
   `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 02:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_cache.sh all"`.
2. It writes `top150_bm25.jsonl`, `top150_dense.jsonl`, `top150_splade.jsonl` and `top150_rrf.jsonl` (+ a
   `.manifest.json` next to each) to `$IRE_DATA_DIR/caches/ada/`. The rrf file is fused from the bm25 and dense
   files; it is a no-poison reference only (experiments fuse the masked bm25 and dense lists themselves).
3. If it is killed, run the same command again: each cache continues after the last question it wrote, and
   finished caches are skipped. `--force` rebuilds.
4. You may run it earlier (before A6 / the final questions) as a timing test; just run it again afterwards.

### A6 poison
```
ADA REQUEST A6
Owner: Arya          Needs: A2, A3, A4 (the three indexes) and Arya's real poison shards, including the adaptive
                     shard, as poison_*.jsonl in $IRE_DATA_DIR/data/poison/ (and her final questions.jsonl)
Run:   git pull && bash ada/run_index_poison.sh        (in a GPU job; no internet needed)
Time:  30-90 min (estimate: BM25 is rebuilt with the shards; dense/SPLADE copy their clean rows and encode only
       the poison passages; 400 questions are searched before and after on each index)
Disk:  free space of about the largest index while it is swapped (~13 GB for dense); indexes grow by the poison
RAM:   29 GB (QOS maximum)          Order: A5 (caches) must run AFTER A6.
Success looks like: "RESULT: OK" and "SUMMARY A6: OK"; dense and SPLADE masking exactly equivalent, BM25 rank
       differences reported, and poison in its target's top-10 for at least one retriever.
Send back: $IRE_DATA_DIR/reports/A6.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A6_index_poison.log and the report file if one exists.
```
Details for a non-expert:
1. Put every shard (`poison_*.jsonl`, including the adaptive diverse/repeat/stealth shard) into
   `$IRE_DATA_DIR/data/poison/`. Each record must pass `PoisonRecord` (see HANDOFF "Interface changes") and
   its qid must be in `data/questions.jsonl`; the job stops and lists the problems otherwise.
2. Submit it as a GPU job, e.g.
   `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t 04:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_index_poison.sh"`.
3. Re-running with the same shards does nothing; if you change or add a shard, run it again and only the poison
   part is redone (BM25 is rebuilt). Removing all shards and re-running gives back the clean indexes.
4. It does not matter whether the shards were already in `data/poison/` when A2-A4 ran: the job then builds a
   temporary clean copy to get the masking reference.
5. **Afterwards run A5** (`bash ada/run_cache.sh all`): every cache built before A6 is stale and the loader
   refuses it.

### A7 generate
```
ADA REQUEST A7
Owner: Vansh          Needs: A0 (environment + Qwen/Qwen2.5-3B-Instruct downloaded) and A1 (data)
FIRST: confirm the Ada GPU fits the 3B model in fp16: about 7 GB of weights (2 bytes x ~3.1B parameters) plus a
       few GB for batches; check `nvidia-smi` in your GPU session (16 GB or more is comfortable).
Run:   git pull && bash ada/run_generate_benchmark.sh   (in a GPU job; no internet needed)
Time:  15-30 min (estimate: model load, ~100 generations for the tests, 24 prompts x 4 batch sizes)
Disk:  the document text store $IRE_DATA_DIR/data/text_store.sqlite (~2 GB for 2.68M passages, built on first
       use, reused afterwards) + a temporary cache that is deleted at the end
Success looks like: "RESULT: OK" and "SUMMARY A7: OK"; throughput in requests/min for batch sizes 1, 8, 16, 32,
       the projected hours for 60,000 and 30,000 requests, cache hit + greedy determinism 20/20, the batch
       independence counts, and a smoke EM/F1 on 50 clean questions.
Send back: $IRE_DATA_DIR/reports/A7.json (paste it in the team chat or commit it to reports/)
If it fails: send the last 50 lines of $IRE_DATA_DIR/logs/A7_generate_benchmark.log and the report file if one exists.
```
Details for a non-expert:
1. Submit it as a GPU job, e.g.
   `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 4 --mem=11G -t 01:00:00 -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/run_generate_benchmark.sh"`
   (the 3B model loads straight onto the GPU; the text store is built by streaming, so 11 GB of RAM is plenty).
2. If the GPU runs out of memory, the batch size halves itself; to test other batch sizes:
   `bash ada/run_generate_benchmark.sh --batch-sizes 1,4,8`.
3. It does not touch the real generation cache (`caches/generation.sqlite`); the benchmark uses a temporary one.
4. If no current dense cache exists yet (A3/A5 not done), the prompts use each question's gold passages plus random
   passages; the report says which contexts were used.

## How jobs run on Ada (found by Arya on 2026-10-26; this replaces the older generic advice)
- **Account and limits (QOS `low`)**: at most 10 CPUs, 1 GPU and 32,000 MB RAM per user, and at most 5 queued +
  running jobs. Memory is tied to CPUs (~3 GB each): `--mem=32G` (32,768 MB) is over the cap and `--mem=31G -c 10`
  is bumped to 11 CPUs and blocked, so **use `-c 10 --mem=29G`** for big jobs. Partition `u22`; leave
  `ADA_ACCOUNT` empty (`-A research` is rejected on `u22-cpu`).
- **GPUs**: most `u22` nodes have an RTX 2080 Ti (11 GB, feature `2080ti`); some are older (gnode004: GTX 1080 Ti,
  whose fp16 is slow: A3 there was on track for ~10 h vs ~1 h on a 2080 Ti). Use a 2080 Ti node.
- **Storage**: `/share1/<user>` is visible only on the login node and has a 25 GB / 3,000-file quota (no Python
  environment fits). Compute nodes see home (`/home2/<user>`, 50 GB) and node-local `/scratch` (~1.7 TB) and
  `/ssd_scratch` (~480 GB), which are not shared between nodes. The environment lives in home
  (`IRE_ENV_DIR=/home2/arya.mirani/ire_env` in `.env`, `PIP_NO_CACHE_DIR=1`); `IRE_DATA_DIR` and `HF_HOME` point to
  `/home2/arya.mirani/ire_data_dir`.
- **One pinned node**: all indexes (~19 GB) live on gnode056's `/scratch/arya.mirani/ire_data_dir`, so every job
  that touches an index or a cache is submitted to that node through `ada/pinned_node.sh`:
  `sbatch -p u22 -w gnode056 --gres=gpu:1 -c 10 --mem=29G -t <time> -o ~/ire_data_dir/logs/%x-%j.log --wrap "bash $HOME/ire_project/ada/pinned_node.sh ada/<job>.sh <args>"`.
  `pinned_node.sh` syncs the code to `~/ire_node`, stages data + models in from home, runs the job, and copies
  reports, logs, caches and poison shards back to home (indexes stay on the node). Repo files are not synced
  back, except the two threshold files (`query_phrase_thresholds.json`, `perplexity_thresholds.json`), which are
  copied to `~/ire_data_dir/reports/thresholds/` (commit them from there). Do not symlink
  `indexes/`: `build_index.py` needs it to be a real folder under `IRE_DATA_DIR`. If gnode056's `/scratch` is ever
  wiped, rebuild A2-A4 and A6, then A5 (about 2.5 h of jobs); moving to another node is described in
  `HANDOFF_ARYA.md` ("Pinned node moved").
- **Check what the node has free before submitting** (other users' CPU jobs can hold the CPUs even when a GPU is
  free): `scontrol show node gnode056 | grep -E "CfgTRES|AllocTRES"`. Small jobs (GPT-2, filters, generation:
  the FAISS index is only loaded for searches and the LLM loads straight onto the GPU) fit in `-c 2 --mem=5G`;
  anything that searches the dense index (8.2 GB in RAM) plus a model needs `-c 8 --mem=23G` or more.
- **No GPU, no transformers**: with torch 2.6 + transformers 5.19, `import transformers` fails on a node without a
  GPU (`Cannot access accelerator device when none is available`). BM25 (A2) and data preparation (A1) do not
  import it; every job that loads a model must run in a GPU job.
- **Do not start two jobs that write the same index or cache at the same time.**

## Division of labour (suggested; adjust freely)
- Arya: data (A1), BM25 (A2), poison ingestion (A6). She already owns the poison shards.
- Vansh: dense (A3), SPLADE (A4), caches (A5), LLM benchmark (A7), perplexity scores (A8). He already runs the sweeps and needs these outputs.
- Either: A0 setup and A9 smoke test.
