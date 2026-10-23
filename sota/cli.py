"""Command line: ``sota transcribe audio.wav [...]`` and ``sota models``."""

from __future__ import annotations

import argparse
import sys

from .hub import DEFAULT_MODEL, MODELS


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sota", description="Khmer and English speech recognition.")
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transcribe", help="print one line per file: <path><TAB><text>")
    t.add_argument("audio", nargs="+", help="audio files (wav, flac, ogg, mp3, ...)")
    t.add_argument("--model", default=DEFAULT_MODEL, help=f"short name, Hugging Face id or path (default {DEFAULT_MODEL})")
    t.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    t.add_argument("--batch-size", type=int, default=8)
    sub.add_parser("models", help="list the released models")
    args = ap.parse_args(argv)

    if args.command == "models":
        for name, repo in MODELS.items():
            print(f"{name}\thttps://huggingface.co/{repo}")
        return

    from .asr import load_model

    asr = load_model(args.model, args.device)
    for path, text in zip(args.audio, asr.transcribe(list(args.audio), batch_size=args.batch_size)):
        print(f"{path}\t{text}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
