"""Where model weights come from: a short name, a Hugging Face repo id, a local folder or a .pt file."""

from __future__ import annotations

import json
from pathlib import Path

import torch

#: short names for the released models
MODELS = {
    "sota-base": "Darayut/sota-base",  # 73M, pruned from omniASR CTC-300M
    "sota-large": "Darayut/sota-large",  # 204M, pruned from omniASR CTC-1B
    "teacher-300m": "Darayut/omniASR-CTC-300M-km-en",  # 325M, fine-tuned, unpruned
    "teacher-1b": "Darayut/omniASR-CTC-1B-km-en",  # 975M, fine-tuned, unpruned
}
DEFAULT_MODEL = "sota-base"
_FILES = ["config.json", "model.safetensors", "vocab.json"]


def available_models() -> dict[str, str]:
    """Short name -> Hugging Face repo id."""
    return dict(MODELS)


def resolve(name_or_path: str | Path) -> Path:
    """A local folder or file for ``name_or_path``; Hub repos are downloaded (and cached) first."""
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path
    repo = MODELS.get(str(name_or_path), str(name_or_path))
    if "/" not in repo:
        raise ValueError(f"unknown model {name_or_path!r}: use one of {sorted(MODELS)}, a Hugging Face id or a path")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=_FILES))


def load_weights(path: Path) -> tuple[dict, dict[str, torch.Tensor], Path | None]:
    """(architecture dict, state dict, vocab.json or None) from a release folder or a training checkpoint."""
    if path.is_dir():
        from safetensors.torch import load_file

        arch = json.loads((path / "config.json").read_text())
        vocab = path / "vocab.json"
        return arch, load_file(str(path / "model.safetensors")), vocab if vocab.exists() else None
    ck = torch.load(path, map_location="cpu", weights_only=False)  # {"arch", "model", ...} from training
    return ck["arch"], ck["model"], None
