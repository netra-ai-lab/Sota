"""Export a full-size (unpruned) checkpoint, e.g. the Stage 1 teacher, as a release folder.

    python scripts/05_export_teacher.py runs/base/stage1_finetune/teacher.pt runs/base/release/teacher --size 300m

Writes:
    model.safetensors + config.json   the weights for ``sota.load_model``
    <name>.pt                         a fairseq2 checkpoint, a drop-in for the original omniASR one
    omniASR_tokenizer.model           the unchanged char tokenizer
    asset_card.yaml                   fairseq2 card for the two files above (paths to fill in)

and checks that the fairseq2 checkpoint, reloaded through fairseq2, gives the same logits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sota.model import ArchConfig, PrunableCTC  # noqa: E402
from sota_train import TOKENIZER, SIZES  # noqa: E402
from sota_train.convert import parity  # noqa: E402

#: inverse of sota_train.convert.map_state_dict
TO_FAIRSEQ2 = {
    "encoder_frontend.pos_conv.parametrizations.weight.original0": "encoder_frontend.pos_encoder.conv.weight_g",
    "encoder_frontend.pos_conv.parametrizations.weight.original1": "encoder_frontend.pos_encoder.conv.weight_v",
    "encoder_frontend.pos_conv.bias": "encoder_frontend.pos_encoder.conv.bias",
}

CARD = """# fairseq2 asset card. Replace /path/to/{name} with the folder holding these files,
# then: export FAIRSEQ2_USER_ASSET_DIR=<that folder>
name: {card}
model_family: wav2vec2_asr
model_arch: {arch}
checkpoint: "file:///path/to/{name}/{name}.pt"
tokenizer_ref: {card}_tokenizer

---

name: {card}_tokenizer
tokenizer_family: char_tokenizer
tokenizer: "file:///path/to/{name}/omniASR_tokenizer.model"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="{arch, model} checkpoint of a full-size model")
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", choices=sorted(SIZES), required=True, help="fairseq2 model_arch of the stock model")
    ap.add_argument("--name", help="file stem of the fairseq2 checkpoint (default: <stock name>-km-en)")
    args = ap.parse_args()
    name = args.name or f"{SIZES[args.size]}-km-en"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if not torch.all(sd["encoder_frontend.feature_extractor.out_scale"] == 1):
        raise SystemExit("conv channels were pruned; fairseq2 cannot represent this model (use 06_export_pruned.py)")
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(ck["arch"], indent=2) + "\n")

    fs2_sd = {TO_FAIRSEQ2.get(k, k): v for k, v in sd.items() if k != "encoder_frontend.feature_extractor.out_scale"}
    torch.save({"model": fs2_sd, "fs2": True}, out / f"{name}.pt")
    shutil.copy(TOKENIZER, out / "omniASR_tokenizer.model")
    card = name.replace("-", "_")
    (out / "asset_card.yaml").write_text(CARD.format(name=name, card=card, arch=args.size))

    # reload through fairseq2 from a card with the real paths, compare with the training model
    check_dir = out / ".card_check"
    check_dir.mkdir(exist_ok=True)
    (check_dir / "card.yaml").write_text(CARD.format(name=name, card=card, arch=args.size).replace(f"/path/to/{name}", str(out)))
    os.environ["FAIRSEQ2_USER_ASSET_DIR"] = str(check_dir)
    import omnilingual_asr  # noqa: F401
    from fairseq2.models import load_model

    fs2 = load_model(card, dtype=torch.float32, device=torch.device("cpu")).eval()
    model = PrunableCTC(ArchConfig.from_dict(ck["arch"]))
    model.load_state_dict(sd)
    model.eval()
    g = torch.Generator().manual_seed(0)
    lens = torch.tensor([16000 * 4, 16000 * 2 + 7])
    audio = torch.randn(2, int(lens.max()), generator=g) * 0.1
    audio[1, lens[1]:] = 0
    diff = parity(model, fs2, audio, lens)
    shutil.rmtree(check_dir)
    print(f"{out}: fairseq2 reload max |logit diff| = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()
