# Training SOTA

Everything for training lives in `training/` and is separate from the `sota` package, so the package never needs it. One command runs the whole recipe for one model size:

```bash
pip install -e ".[train]"
cp training/local.env.example training/local.env      # then point it at your data
training/run.sh base                                   # omniASR CTC-300M -> sota-base (73M)
training/run.sh large                                  # omniASR CTC-1B   -> sota-large (204M)
```

Each step skips itself once its output exists, and each training stage resumes from its last checkpoint. To continue after an interruption, run the same command again.

## 1. What `run.sh` does

| Step | Command | Output |
| --- | --- | --- |
| 1. Assets | `scripts/00_get_assets.sh 1b\|300m` | stock checkpoint, char tokenizer, LibriSpeech (100 + 360 h, dev, test) |
| 2. Manifests | `scripts/01_prepare_data.py` and `--only fleurs` | `data/*.jsonl`, `data/*_units.json`; downloads FLEURS en_us from the Hub |
| 3. Convert | `python -m sota_train.convert --size 1b\|300m` | `<SOTA_MODELS_DIR>/<name>/<name>.sota.pt`, after a logit-parity check against fairseq2 |
| 4. Chain | `scripts/run_chain.sh large\|base` | `runs/<size>/…`, then `runs/eval/<size>.json` and `runs/eval/<size>_teacher.json` |
| 5. Release | `scripts/06_export_pruned.py`, `scripts/05_export_teacher.py` | `runs/<size>/release/` in the Hugging Face layout |

`scripts/run_chain.sh` runs the stages in this order:

| Stage | Config (`configs/<size>/`) | What it does |
| --- | --- | --- |
| 1 | `stage1_finetune.yaml` | CTC fine-tune of the full model. Its weights are exported to `teacher.pt`. |
| 2 | `stage2_prune_s80.yaml` | Hard Concrete gates on heads and FFN units, plus distillation to the teacher, at 80% target sparsity. |
| Cut | `scripts/04_apply_pruning.py` | Folds the gates into the weights and removes pruned structures. Checks that the cut model reproduces the masked one. |
| 2b | `stage2b_distill.yaml` | Distillation with the mask frozen. |
| 3 | `stage3_recover.yaml` | CTC plus frame-level KL to the teacher's outputs, for 16k steps. |

The method and the reasons behind it are in [DESIGN.md](DESIGN.md), and every number is in [RESULTS.md](RESULTS.md).

## 2. Data and paths

All locations are environment variables. `training/local.env` (git-ignored) sets them once for a machine, and an unset variable defaults to a folder inside `training/`:

| Variable | Default | Holds |
| --- | --- | --- |
| `SOTA_MODELS_DIR` | `training/pretrained` | `omniASR-CTC-1B/`, `omniASR-CTC-300M/` (stock `.pt`, converted `.sota.pt`) and `omniASR_tokenizer.model` |
| `ASR_ROOT` | `training/corpora/khmer` | the Khmer corpus (below) |
| `LIBRI_DIR` | `training/corpora/librispeech` | `LibriSpeech/<subset>/`, downloaded by step 1 |
| `FLEURS_EN_DIR` | `training/corpora/fleurs-en` | FLEURS en_us wavs, written by step 2 |
| `SOTA_CACHE` | `training/cache` | decoded eval audio and the generated fairseq2 asset cards |

**The Khmer corpus can't be downloaded by the scripts.** `ASR_ROOT` must already hold:

```
data/dataset/train.tsv                 DDD training index: shard_path, row_index, transcript, duration, speaker_id, source
data/dataset/train/*.parquet           DDD shards: columns audio.bytes, transcript
data/fleurs_train/*.parquet            FLEURS-km train shards, same schema
manifests/val_indomain.jsonl           DDD test set (held out, reporting only)
manifests/train_combined.json          {"exclude_keys": [...]} rows never trained on
manifests/test_fleurs_test.jsonl       FLEURS-km test
manifests/dev_fleurs.jsonl             FLEURS-km validation
```

`scripts/01_prepare_data.py` documents each output file at its top. DDD rows recorded at 48 kHz are resampled to 16 kHz when they are read.

## 3. Running one stage

```bash
cd training && source scripts/env.sh
scripts/run_stage.sh configs/base/stage2_prune_s80.yaml [--set optim.max_steps=50 ...]
"$PY" scripts/03_evaluate.py runs/base/stage3_recover/latest.pt --name my_eval [--sets ddd_test libri_test_clean]
"$PY" scripts/07_time_inference.py            # inference time of each teacher and SOTA model -> runs/eval/inference_time.json
```

`scripts/env.sh` reads `local.env` and picks the interpreter: the repo's `.venv` if present, else `python3`. It also puts `training/` and the repo root on `PYTHONPATH`.

**Ablations.** The configs that reproduce the ablations in [RESULTS.md](RESULTS.md) are in `configs/experiments/`:
- `a_prune_from_stock`: no Stage 1;
- `large_b1_ctc`: Stage 3 with CTC only;
- `large_b2_kd8k`: Stage 3 with KD for 8k steps.

## 4. Checkpoints and resuming

Rerun the same command to continue from `<run_dir>/latest.pt`.

- **When it saves.** `latest.pt` is written:
  - every `save.every_steps` steps or `save.every_minutes` minutes, whichever comes first;
  - on SIGINT/SIGTERM, after the current step;
  - after a crash outside the optimizer step;
  - at the end of the stage.
- **What it holds.**
  - fp32 master weights and AdamW moments, which can be stored in bf16 with `save.moments_dtype`;
  - gates and Lagrange multipliers;
  - the step counter, from which the LR and sparsity schedules follow;
  - data progress (the epoch and the finished data units);
  - Python, NumPy, torch and CUDA RNG states;
  - the config and the git commit.
- **How it writes.** Each save goes to a temp file that is then renamed, so a crash mid-save never loses the previous checkpoint. If the disk can't hold the temp file, the save is skipped with a warning.
- **One checkpoint per run.** The configs set `save.best: false`, so each run keeps only `latest.pt`. Evals still run every `eval.every_steps` and are logged to `metrics.jsonl`.
- **Supervision.** `scripts/run_stage.sh` relaunches the trainer after a crash, up to 5 times in a row.
  - To stop it cleanly, run `kill -TERM <its pid>`.
  - Exit codes: 0 = done, 3 = stopped, anything else = crash.
- **Refusing a bad resume.** A resume is refused if the stage, init, teacher, seed, pruning units, precision, target sparsity, distillation layers, language mix or KD weight changed.
- **Data position.** Progress is tracked per *unit*: one DDD parquet shard, or one chunk of 100 LibriSpeech utterances. A resumed epoch skips finished units. A crash can repeat at most the tail of a few units, and never skips one.

## 5. Outputs

```
training/runs/
├── large/ base/
│   ├── stage1_finetune/      DONE, metrics.jsonl, teacher.pt
│   ├── stage2_prune_s80/     latest.pt (gated), pruned.pt + pruned.json (per-layer architecture)
│   ├── stage2b_distill/      latest.pt
│   ├── stage3_recover/       latest.pt  <- the released model
│   └── release/              sota-<size>/, teacher/
├── experiments/              the ablation runs
├── eval/<name>.json          test-set scores, parameters, GMAC/s, real-time factor
└── logs/                     <size>_<stage>.log, chain_<size>.log
```

Training logs go to `runs/logs/<size>_<stage>.log` and metrics to `<run_dir>/metrics.jsonl`.

## 6. Precision modes

`model.precision` in a config is one of:

| Mode | Use |
| --- | --- |
| `amp` | fp32 weights with bf16 autocast. The 300M stages and every pruned-model stage use it. |
| `offload` | bf16 weights and gradients on the GPU; fp32 master weights and AdamW moments in host memory, updated there each step (`sota_train/offload.py`). The 1B stages use it. |

Stage 2 freezes the CNN feature extractor, since its channels aren't pruned, and uses gradient checkpointing. All gates are sampled once per forward pass, outside the checkpointed region, so the recompute during backward sees the same masks (`tests/test_model.py`).
