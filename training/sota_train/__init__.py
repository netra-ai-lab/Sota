"""sota_train: the training pipeline behind the SOTA models (fine-tune, prune, distil, recover).

Every external location is an environment variable with a default inside ``training/``.
``training/local.env`` (git-ignored, ``KEY=VALUE`` lines) is read first, so a machine
keeps its paths there instead of exporting them in every shell:

    SOTA_MODELS_DIR  stock Omnilingual checkpoints + tokenizer, and their converted copies
    ASR_ROOT         Khmer corpus root: DDD parquet shards, FLEURS-km shards and manifests
    LIBRI_DIR        LibriSpeech download dir (holds LibriSpeech/<subset>/)
    FLEURS_EN_DIR    FLEURS en_us wavs, written by scripts/01_prepare_data.py --only fleurs
    SOTA_CACHE       decoded eval audio, one .pt per eval set

The fairseq2 asset cards for the stock checkpoints are generated from SOTA_MODELS_DIR and
registered before fairseq2 is imported, which is why it happens at package import time.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # training/


def _read_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), os.path.expandvars(value.strip().strip('"')))


_read_env_file(PROJECT_ROOT / "local.env")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.setdefault(name, str(PROJECT_ROOT / default)))


MODELS_DIR = _env_path("SOTA_MODELS_DIR", "pretrained")
ASR_ROOT = _env_path("ASR_ROOT", "corpora/khmer")
LIBRI_DIR = _env_path("LIBRI_DIR", "corpora/librispeech")
LIBRI_ROOT = LIBRI_DIR / "LibriSpeech"
FLEURS_EN_DIR = _env_path("FLEURS_EN_DIR", "corpora/fleurs-en")
#: decoded eval audio, one .pt per eval set (~7 GB once every final-eval set is cached)
CACHE_DIR = _env_path("SOTA_CACHE", "cache")

#: stock checkpoint name per model size, as published by Omnilingual ASR
SIZES = {"1b": "omniASR-CTC-1B", "300m": "omniASR-CTC-300M"}
TOKENIZER = MODELS_DIR / "omniASR_tokenizer.model"


def stock_checkpoint(size: str) -> Path:
    """The original fairseq2 checkpoint, e.g. ``<models>/omniASR-CTC-1B/omniASR-CTC-1B.pt``."""
    name = SIZES[size]
    return MODELS_DIR / name / f"{name}.pt"


def converted_checkpoint(size: str) -> Path:
    """The same weights in this project's ``{"arch", "model"}`` format (``python -m sota_train.convert``)."""
    name = SIZES[size]
    return MODELS_DIR / name / f"{name}.sota.pt"


def fairseq2_card(size: str) -> str:
    return f"{SIZES[size].replace('-', '_')}_local"


def _write_asset_cards() -> Path:
    cards = [
        f"name: {fairseq2_card(size)}\nmodel_family: wav2vec2_asr\nmodel_arch: {size}\n"
        f'checkpoint: "file://{stock_checkpoint(size)}"\ntokenizer_ref: omniASR_tokenizer_local\n'
        for size in SIZES
    ]
    cards.append(f'name: omniASR_tokenizer_local\ntokenizer_family: char_tokenizer\ntokenizer: "file://{TOKENIZER}"\n')
    out = CACHE_DIR / "fairseq2_assets"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "sota_local.yaml").write_text("\n---\n\n".join(cards))
    except OSError:  # read-only checkout: conversion is then unavailable, training still works
        pass
    return out


os.environ.setdefault("FAIRSEQ2_USER_ASSET_DIR", str(_write_asset_cards()))
