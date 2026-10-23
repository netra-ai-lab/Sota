# training/

The pipeline that produced the SOTA models. It is kept apart from the `sota` package, which never imports anything from here.

```bash
pip install -e ".[train]"                 # from the repo root
cp training/local.env.example training/local.env
training/run.sh base                      # or: large
```

| Path | |
| --- | --- |
| `run.sh` | One command, end to end: assets, manifests, conversion, the chain, evaluation and export. Rerun it to resume. |
| `configs/large`, `configs/base` | The four stages of each released model. |
| `configs/experiments/` | The ablations reported in `docs/RESULTS.md`. |
| `scripts/` | `00`–`06`, numbered in run order (step 02, the conversion, is `python -m sota_train.convert`), plus `run_chain.sh` and `run_stage.sh` (crash-resuming supervisor). |
| `sota_train/` | Trainer, data stream, conversion, evaluation and CPU-offloaded AdamW. |
| `release/cards/` | The Hugging Face model cards as published. |
| `tests/` | Data, KD, offload and resume tests (`pytest` from the repo root runs them). |
| `runs/`, `data/`, `cache/` | Outputs and generated manifests (git-ignored). |

Full guide: [docs/TRAINING.md](../docs/TRAINING.md).
