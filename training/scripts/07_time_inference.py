"""Time inference of each fine-tuned teacher (0% sparsity) and SOTA model (80% sparsity).

    python scripts/07_time_inference.py [--seconds 60] [--runs 20] [--out runs/eval/components.json]

Runs one utterance of ``--seconds`` audio through each model as ``sota.transcribe`` does
(bf16 autocast on CUDA, batch 1) and records the median forward time over ``--runs`` runs,
after warm-up. Compute does not depend on the audio content, so the input is noise.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

import sota
from sota.model import normalize_audio_

# The training checkpoints the released models were exported from (logit diff 0); a hub name works too.
MODELS = {"teacher-300m": "runs/base/stage1_finetune/teacher.pt", "sota-base": "runs/base/stage3_recover/latest.pt",
          "teacher-1b": "runs/large/stage1_finetune/teacher.pt", "sota-large": "runs/large/stage3_recover/latest.pt"}


class PartTimer:
    """CUDA events around a module span: start before ``first`` runs, stop after ``last`` returns."""

    def __init__(self, first: torch.nn.Module, last: torch.nn.Module) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        first.register_forward_pre_hook(lambda *_: self.start.record())
        last.register_forward_hook(lambda *_: self.stop.record())

    def ms(self) -> float:
        return self.start.elapsed_time(self.stop)


@torch.inference_mode()
def time_model(source: str, seconds: float, runs: int) -> dict:
    asr = sota.load_model(source, device="cuda")
    model = asr.model
    timer = PartTimer(model, model)
    wav = torch.randn(1, int(16000 * seconds), device="cuda")
    lens = torch.tensor([wav.shape[1]], device="cuda")
    normalize_audio_(wav, lens)
    times = []
    for i in range(runs + 5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(wav, lens)
        torch.cuda.synchronize()
        if i >= 5:
            times.append(timer.ms())
    out = {"ms": statistics.median(times), "params": asr.num_parameters}
    del asr, model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--out", default="runs/eval/inference_time.json")
    args = ap.parse_args()
    result = {"audio_seconds": args.seconds, "runs": args.runs, "models": {}}
    for name, source in MODELS.items():
        result["models"][name] = time_model(source if Path(source).exists() else name, args.seconds, args.runs)
        print(name, {k: round(v, 1) for k, v in result["models"][name].items()})
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
