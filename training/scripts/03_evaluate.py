"""Evaluate a checkpoint ({"arch", "model"}: converted, best.pt, pruned.pt) on the eval sets.

    .venv/bin/python scripts/03_evaluate.py <ckpt> --name 1b_zeroshot [--sets ddd_dev ddd_test ...]

Writes runs/eval/<name>.json with CER/WER, real-time factor on this GPU, parameter
count and MACs per second of audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sota_train import ASR_ROOT, CACHE_DIR, PROJECT_ROOT  # noqa: E402
from sota_train.data import load_eval_set  # noqa: E402
from sota_train.evaluate import evaluate  # noqa: E402
from sota.model import PrunableCTC, load_model  # noqa: E402
from sota.text import CharVocab  # noqa: E402

SETS = {
    "ddd_dev": "data/ddd_dev.jsonl",
    "ddd_test": "data/ddd_test.jsonl",
    "fleurs_km_dev": "data/fleurs_km_dev.jsonl",
    "fleurs_km_test": "data/fleurs_km_test.jsonl",
    "fleurs_en_dev": "data/fleurs_en_dev.jsonl",
    "fleurs_en_test": "data/fleurs_en_test.jsonl",
    "libri_dev_clean": "data/libri_dev-clean.jsonl",
    "libri_test_clean": "data/libri_test-clean.jsonl",
    "libri_test_other": "data/libri_test-other.jsonl",
}


def macs_per_second(model: PrunableCTC) -> float:
    """Multiply-accumulates for one second of 16 kHz audio (matmuls/convs only)."""
    cfg = model.cfg
    t, in_ch, macs = 16000, 1, 0.0
    for out_ch, k, s in cfg.conv_layers:
        t = (t - k) // s + 1
        macs += t * k * in_ch * out_ch
        in_ch = out_ch
    d = cfg.model_dim
    macs += t * in_ch * d  # projection
    macs += t * cfg.pos_conv_kernel * d * d / cfg.pos_conv_groups
    for h, f in zip(cfg.num_heads, cfg.ffn_dims):
        hd = h * cfg.head_dim
        macs += 4 * t * d * hd + 2 * t * t * hd  # q, k, v, out + scores and weighted sum
        macs += 2 * t * d * f
    macs += t * d * cfg.vocab_size
    return macs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-batch-seconds", type=float, default=200.0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    device = torch.device("cuda", 0)
    model = load_model(str(args.ckpt))
    model.to(device=device, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32).eval()
    for hc in model.hard_concretes():  # a gated (mid-pruning) checkpoint: deterministic masks
        hc.log_alpha.data = hc.log_alpha.data.float()
    vocab = CharVocab.default()
    sets = {}
    for name in args.sets:
        path = PROJECT_ROOT / SETS[name]
        if not path.exists():
            print(f"skipping {name}: {path} missing")
            continue
        sets[name] = load_eval_set(path, vocab, ASR_ROOT, limit=args.limit, cache_dir=CACHE_DIR)
    res = evaluate(model, sets, vocab, device=device, max_batch_seconds=args.max_batch_seconds, examples=3)
    out = {
        "ckpt": str(args.ckpt),
        "params_total": sum(p.numel() for n, p in model.named_parameters() if not n.endswith("log_alpha")),
        "params_encoder": float(model.num_params("deterministic")),
        "gmacs_per_audio_second": macs_per_second(model) / 1e9,
        "dtype": args.dtype,
        "results": res,
    }
    dest = PROJECT_ROOT / "runs/eval" / f"{args.name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{args.name}: {out['params_total'] / 1e6:.1f}M params, {out['gmacs_per_audio_second']:.1f} GMAC/s audio")
    for name, r in res.items():
        extra = f" WER {100 * r['wer']:.2f}%" if "wer" in r else ""
        print(f"  {name:18s} n={r['n']:5d}  CER {100 * r['cer']:.2f}%{extra}  RTF {r['rtf']:.4f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
