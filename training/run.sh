#!/usr/bin/env bash
# Train a SOTA model from scratch, end to end:
#
#   training/run.sh base     # omniASR CTC-300M -> 73M  (sota-base)
#   training/run.sh large    # omniASR CTC-1B   -> 204M (sota-large)
#
# 1. download the stock checkpoint, tokenizer and LibriSpeech     scripts/00_get_assets.sh
# 2. build the training/eval manifests (+ FLEURS en/km)           scripts/01_prepare_data.py
# 3. convert the fairseq2 checkpoint to the training format      python -m sota_train.convert
# 4. fine-tune -> prune -> cut -> distil -> recover -> evaluate   scripts/run_chain.sh
# 5. export the release folders (Hugging Face layout)            scripts/05/06_export_*.py
#
# Every step skips itself when its output exists and every stage resumes from its last
# checkpoint, so rerunning the same command continues where it stopped. Paths come from
# training/local.env (see local.env.example). The Khmer corpus (DDD + FLEURS-km, $ASR_ROOT)
# is not downloadable and must be in place first; see docs/TRAINING.md.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
SIZE=${1:?usage: training/run.sh base|large}
case $SIZE in large) STOCK=1b; NAME=omniASR-CTC-1B ;; base) STOCK=300m; NAME=omniASR-CTC-300M ;; *) echo "size must be base or large"; exit 2 ;; esac
eval "$("$PY" -c 'import sota_train as t; print(f"ASR_ROOT={t.ASR_ROOT} MODELS={t.MODELS_DIR}")')"
say() { echo "=== $(date '+%F %T') run.sh $SIZE: $*"; }

if [ ! -f "$ASR_ROOT/data/dataset/train.tsv" ]; then
    echo "Khmer corpus not found at ASR_ROOT=$ASR_ROOT (expects data/dataset/train.tsv, manifests/, data/fleurs_train/)."
    echo "Set ASR_ROOT in training/local.env; the expected layout is in docs/TRAINING.md."
    exit 1
fi

say "1/5 assets"
scripts/00_get_assets.sh "$STOCK"

say "2/5 manifests"
[ -f data/ddd_units.json ] && [ -f data/libri_train.jsonl ] || "$PY" scripts/01_prepare_data.py
[ -f data/fleurs_en_train.jsonl ] && [ -f data/fleurs_km_units.json ] || "$PY" scripts/01_prepare_data.py --only fleurs

say "3/5 convert $NAME"
[ -f "$MODELS/$NAME/$NAME.sota.pt" ] || "$PY" -m sota_train.convert --size "$STOCK"

say "4/5 training chain"
scripts/run_chain.sh "$SIZE"

say "5/5 release export"
OUT=runs/$SIZE/release
[ -f "$OUT/sota-$SIZE/model.safetensors" ] || "$PY" scripts/06_export_pruned.py "runs/$SIZE/stage3_recover/latest.pt" "$OUT/sota-$SIZE"
[ -f "$OUT/teacher/model.safetensors" ] || "$PY" scripts/05_export_teacher.py "runs/$SIZE/stage1_finetune/teacher.pt" "$OUT/teacher" --size "$STOCK"
say "done: scores in runs/eval/$SIZE.json, models in $OUT/"
