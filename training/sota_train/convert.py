"""Convert a fairseq2 Omnilingual CTC checkpoint into this project's format.

fairseq2 does the heavy lifting of reading the original ``.pt`` (``load_model`` on a
local asset card); this module reads the architecture off the loaded module tree,
maps the state dict onto ``PrunableCTC`` (an identity map apart from the weight-normed
position conv), and checks that both models produce the same logits.

    python -m sota_train.convert --size 1b      # -> $SOTA_MODELS_DIR/omniASR-CTC-1B/omniASR-CTC-1B.sota.pt
    python -m sota_train.convert --vocab        # re-export sota/vocab.json from the tokenizer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sota.model import ArchConfig, PrunableCTC, normalize_audio_
from sota.text import SPACE_PIECE, SPECIAL, VOCAB_PATH, CharVocab

from . import SIZES, converted_checkpoint, fairseq2_card  # importing the package registers the asset cards


def load_fairseq2(card: str):
    import omnilingual_asr  # noqa: F401  (registers the omniASR model family + cards)
    from fairseq2.models import load_model

    model = load_model(card, dtype=torch.float32, device=torch.device("cpu"))
    return model.eval()


def arch_from_fairseq2(fs2) -> ArchConfig:
    fe = fs2.encoder_frontend.feature_extractor.layers
    conv = [(l.conv.out_channels, l.conv.kernel_size[0], l.conv.stride[0]) for l in fe]
    layers = fs2.encoder.layers
    attn0 = layers[0].self_attn
    pos = fs2.encoder_frontend.pos_encoder.conv
    return ArchConfig(
        conv_layers=conv,
        num_heads=[l.self_attn.num_heads for l in layers],
        ffn_dims=[l.ffn.inner_proj.output_dim for l in layers],
        model_dim=fs2.model_dim,
        head_dim=attn0.head_dim,
        pos_conv_kernel=pos.kernel_size[0],
        pos_conv_groups=pos.groups,
        vocab_size=fs2.final_proj.output_dim,
        feature_grad_scale=float(fs2.encoder_frontend.feature_extractor.grad_scale),
    )


def map_state_dict(fs2_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in fs2_sd.items():
        if k.startswith("encoder_frontend.pos_encoder.conv."):
            leaf = k.rsplit(".", 1)[1]
            k = {
                "weight_g": "encoder_frontend.pos_conv.parametrizations.weight.original0",
                "weight_v": "encoder_frontend.pos_conv.parametrizations.weight.original1",
                "bias": "encoder_frontend.pos_conv.bias",
            }[leaf]
        out[k] = v
    return out


def convert(card: str) -> tuple[PrunableCTC, object]:
    fs2 = load_fairseq2(card)
    cfg = arch_from_fairseq2(fs2)
    model = PrunableCTC(cfg)
    sd = map_state_dict(fs2.state_dict())
    sd["encoder_frontend.feature_extractor.out_scale"] = torch.ones(cfg.conv_layers[-1][0])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    return model.eval(), fs2


def vocab_from_fairseq2(card: str) -> CharVocab:
    """The char tokenizer's pieces in id order (how ``sota/vocab.json`` was made)."""
    from fairseq2.data.tokenizers import load_tokenizer

    tok = load_tokenizer(card)
    dec = tok.create_decoder()
    pieces = []
    for i in range(tok.vocab_info.size):
        s = dec(torch.tensor([i]))
        # the decoder renders the space piece as "" (a bare word boundary)
        pieces.append(SPACE_PIECE if s == "" and i not in SPECIAL else s)
    return CharVocab(pieces)


@torch.no_grad()
def fairseq2_logits(fs2, audio: torch.Tensor, audio_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from fairseq2.nn import BatchLayout

    layout = BatchLayout.of(audio, seq_lens=audio_lens.tolist())
    seqs, seqs_layout, _ = fs2.encoder_frontend.extract_features(audio, layout)
    seqs, _ = fs2.encoder_frontend.process_features(seqs, seqs_layout, None)
    seqs = fs2.encoder(seqs, seqs_layout)
    lens = torch.tensor(seqs_layout.seq_lens, dtype=torch.long)
    return fs2.final_proj(seqs), lens


@torch.no_grad()
def parity(model: PrunableCTC, fs2, audio: torch.Tensor, audio_lens: torch.Tensor) -> float:
    """Max |logit difference| over the valid frames of a padded batch."""
    audio = audio.clone()
    normalize_audio_(audio, audio_lens)
    ref, ref_lens = fairseq2_logits(fs2, audio, audio_lens)
    out = model(audio, audio_lens)
    if not torch.equal(out["frame_lens"].cpu(), ref_lens):
        raise AssertionError(f"frame lengths differ: {out['frame_lens'].tolist()} vs {ref_lens.tolist()}")
    worst = 0.0
    for i, n in enumerate(ref_lens.tolist()):
        worst = max(worst, (out["logits"][i, :n] - ref[i, :n]).abs().max().item())
    return worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), default="1b")
    ap.add_argument("--out", type=Path, help="default: $SOTA_MODELS_DIR/<name>/<name>.sota.pt")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--vocab", action="store_true", help="only re-export sota/vocab.json from the tokenizer")
    args = ap.parse_args()
    if args.vocab:
        vocab_from_fairseq2("omniASR_tokenizer_local").save(VOCAB_PATH)
        print(f"wrote {VOCAB_PATH}")
        return
    args.card = fairseq2_card(args.size)
    args.out = args.out or converted_checkpoint(args.size)

    model, fs2 = convert(args.card)
    g = torch.Generator().manual_seed(0)
    lens = torch.tensor([16000 * 6, 16000 * 4 + 123, 16000 * 2 + 7])
    audio = torch.randn(3, int(lens.max()), generator=g) * 0.1
    for i, n in enumerate(lens.tolist()):
        audio[i, n:] = 0
    diff = parity(model, fs2, audio, lens)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.card}: {n_params / 1e6:.1f}M params, prunable {model.num_params() / 1e6:.1f}M, "
          f"max |logit diff| vs fairseq2 = {diff:.2e}")
    if diff > args.tol:
        raise SystemExit(f"parity check failed (> {args.tol})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": model.cfg.to_dict(), "model": model.state_dict(), "source": args.card}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
