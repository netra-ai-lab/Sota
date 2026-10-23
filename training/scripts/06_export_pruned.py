"""Export a pruned checkpoint as a release folder that ``sota.load_model`` reads.

    python scripts/06_export_pruned.py runs/base/stage3_recover/latest.pt runs/base/release/sota-base

Writes model.safetensors, config.json (per-layer heads and FFN widths) and vocab.json, then
reloads the folder with ``sota.load_model`` and checks it reproduces the checkpoint's logits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sota  # noqa: E402
from sota.model import ArchConfig, PrunableCTC  # noqa: E402
from sota.text import VOCAB_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="{arch, model} or trainer checkpoint of a pruned model")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    save_file({k: v.contiguous() for k, v in ck["model"].items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(ck["arch"], indent=2) + "\n")
    shutil.copy(VOCAB_PATH, out / "vocab.json")

    ref = PrunableCTC(ArchConfig.from_dict(ck["arch"]))
    ref.load_state_dict(ck["model"])
    ref.eval()
    asr = sota.load_model(out, device="cpu")
    g = torch.Generator().manual_seed(0)
    audio = torch.randn(1, 16000 * 3, generator=g) * 0.1
    lens = torch.tensor([audio.shape[1]])
    with torch.no_grad():
        diff = (asr.model(audio, lens)["logits"] - ref(audio, lens)["logits"]).abs().max().item()
    n = sum(p.numel() for p in asr.model.parameters())
    print(f"{out}: {n / 1e6:.2f}M params, max |logit diff| vs checkpoint = {diff:.2e}")
    if diff != 0:
        raise SystemExit("export does not reproduce the checkpoint")


if __name__ == "__main__":
    main()
