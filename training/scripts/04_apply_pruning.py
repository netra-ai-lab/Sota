"""Turn a Stage 2 checkpoint (gated 1B) into the physically smaller model.

    .venv/bin/python scripts/04_apply_pruning.py runs/stage2_prune_s80/latest.pt \
        --out runs/stage2_prune_s80/pruned.pt [--target 0.8]

Folds the deterministic gates into the weights, removes zeroed heads and FFN units
(``PrunableCTC.prune``), checks on real dev audio that the pruned model reproduces the
masked one, and reports the per-layer architecture.

DiariZen averages the best post-peak checkpoints before this step. Each 1B checkpoint
is ~4 GB here, and runs keep a single checkpoint, so the final ``latest.pt`` is pruned.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sota_train import ASR_ROOT, CACHE_DIR, PROJECT_ROOT  # noqa: E402
from sota_train.data import eval_batches, load_eval_set  # noqa: E402
from sota.model import ArchConfig, PrunableCTC, normalize_audio_  # noqa: E402
from sota.text import CharVocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=float, default=None, help="expected sparsity; warn if off by > 2 points")
    ap.add_argument("--check-utts", type=int, default=8)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = PrunableCTC(ArchConfig.from_dict(ck["arch"]))
    model.load_state_dict(ck["model"])
    model.eval()
    full = float(model.num_params("full"))
    kept = float(model.num_params("deterministic"))

    device = torch.device("cuda", 0)
    vocab = CharVocab.default()
    items = load_eval_set(PROJECT_ROOT / "data/ddd_dev.jsonl", vocab, ASR_ROOT, limit=args.check_utts,
                          cache_dir=CACHE_DIR)
    _, audio, lens = next(eval_batches(items, 1e9))
    normalize_audio_(audio, lens)
    with torch.no_grad():
        ref = model.to(device)(audio.to(device), lens.to(device))

    pruned = copy.deepcopy(model)
    new_cfg = pruned.prune()
    small = PrunableCTC(new_cfg)
    small.load_state_dict(pruned.state_dict())
    small.to(device).eval()
    with torch.no_grad():
        out = small(audio.to(device), lens.to(device))
    worst = max((out["logits"][i, :n] - ref["logits"][i, :n]).abs().max().item()
                for i, n in enumerate(ref["frame_lens"].tolist()))

    n_total = sum(p.numel() for p in small.parameters())
    sparsity = 1 - float(small.num_params("full")) / full
    report = {
        "source": str(args.ckpt), "source_step": ck.get("step"),
        "encoder_params_before": full, "encoder_params_after": float(small.num_params("full")),
        "params_total_after": n_total, "sparsity": sparsity,
        "max_abs_logit_diff_vs_masked": worst,
        "num_heads": new_cfg.num_heads, "ffn_dims": new_cfg.ffn_dims, "conv_layers": new_cfg.conv_layers,
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("num_heads", "ffn_dims", "conv_layers")}, indent=2))
    print("layer  heads  ffn")
    for i, (h, f) in enumerate(zip(new_cfg.num_heads, new_cfg.ffn_dims), 1):
        print(f"{i:5d}  {h:5d}  {f:5d}")
    if abs(kept - small.num_params("full")) > 1:
        raise SystemExit("pruned parameter count differs from the deterministic estimate")
    if worst > 0.05:
        raise SystemExit(f"pruned model does not reproduce the masked model (max diff {worst:.3g})")
    if args.target is not None and abs(sparsity - args.target) > 0.02:
        print(f"WARNING: sparsity {sparsity:.3f} is more than 2 points from the target {args.target}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": new_cfg.to_dict(), "model": small.cpu().state_dict(), "pruning_report": report}, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
