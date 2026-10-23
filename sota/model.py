"""Prunable wav2vec2 CTC model with the layout of fairseq2's Omnilingual CTC models.

The module tree and parameter names mirror ``fairseq2``'s ``Wav2Vec2AsrModel``
(``encoder_frontend.feature_extractor.layers.N.conv``, ``encoder.layers.N.self_attn.q_proj``,
``final_proj``, ...), so an ``omniASR_CTC_*`` checkpoint loads with a near-identity key
map (``sota_train.convert``) and the forward pass matches fairseq2's frame for frame:
pre-LN transformer layers, a final encoder LayerNorm, a weight-normed convolutional
position encoder, and no frontend LayerNorm.

Structured pruning follows DPHuBERT / DiariZen (Peng et al. 2023; Han et al. 2025):
Hard Concrete gates on
  * conv feature-extractor output channels   (``prune_conv``)
  * attention heads                          (``prune_heads``)
  * FFN intermediate units                   (``prune_interm``)
  * whole attention / FFN sub-layers         (``prune_attn_layer`` / ``prune_ffn_layer``, off by default)
``num_params()`` is the expected L0 parameter count, differentiable in the gates, which
the trainer pushes toward the target sparsity with an augmented Lagrangian. ``prune()``
folds the deterministic gates into the weights, removes the zeroed structures, and
returns the smaller architecture, which this same class builds and runs.

The model dimension (1280 for the 1B) is never pruned, so the CTC head keeps its shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .hardconcrete import HardConcrete

SAMPLE_RATE = 16_000


@dataclass
class ArchConfig:
    """Everything needed to build the model. Pruning only changes the per-layer sizes."""

    conv_layers: list[tuple[int, int, int]]  # (out_channels, kernel, stride)
    num_heads: list[int]  # per encoder layer; 0 = attention removed
    ffn_dims: list[int]  # per encoder layer; 0 = FFN removed
    model_dim: int = 1280
    head_dim: int = 80
    pos_conv_kernel: int = 128
    pos_conv_groups: int = 16
    vocab_size: int = 9812
    # Training-time behaviour (not architecture): kept here so a checkpoint rebuilds identically.
    ffn_dropout: float = 0.0
    attn_dropout: float = 0.0
    layerdrop: float = 0.0
    feature_grad_scale: float = 0.1  # fairseq2's wav2vec2 default
    # Which gates exist.
    prune_conv: bool = False
    prune_heads: bool = False
    prune_interm: bool = False
    prune_attn_layer: bool = False
    prune_ffn_layer: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.num_heads)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ArchConfig:
        d = dict(d)
        d["conv_layers"] = [tuple(x) for x in d["conv_layers"]]
        return cls(**d)

    def with_pruning(self, units: str) -> ArchConfig:
        """Copy with gates for ``units`` (comma list of conv, head, interm, attlayer, ffnlayer)."""
        names = {u.strip() for u in units.split(",") if u.strip()}
        unknown = names - {"conv", "head", "interm", "attlayer", "ffnlayer"}
        if unknown:
            raise ValueError(f"unknown pruning units: {sorted(unknown)}")
        d = self.to_dict()
        d.update(
            prune_conv="conv" in names,
            prune_heads="head" in names,
            prune_interm="interm" in names,
            prune_attn_layer="attlayer" in names,
            prune_ffn_layer="ffnlayer" in names,
        )
        return ArchConfig.from_dict(d)


def omni_ctc_1b_config(vocab_size: int = 9812) -> ArchConfig:
    """omniASR_CTC_1B: xlsr-style 7-layer conv frontend, 48 x (1280, 16 heads, 5120 FFN)."""
    return ArchConfig(
        conv_layers=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        num_heads=[16] * 48,
        ffn_dims=[5120] * 48,
        model_dim=1280,
        head_dim=80,
        vocab_size=vocab_size,
    )


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):  # noqa: ANN001
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # noqa: ANN001
        return grad * ctx.scale, None


# --------------------------------------------------------------------------- frontend


class ConvLayer(nn.Module):
    """conv -> LayerNorm over channels -> GELU (fairseq2 ``Wav2Vec2FeatureExtractionLayer``)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, gate: bool) -> None:
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=True)
        self.layer_norm = nn.LayerNorm(out_ch)
        self.hard_concrete = HardConcrete(out_ch, init_mean=0.01) if gate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        if self.hard_concrete is not None:
            x = x * self.hard_concrete().to(x.dtype).unsqueeze(-1)
        return x

    def out_channels(self, mode: str = "expected") -> torch.Tensor | int:
        return self.hard_concrete.count(mode) if self.hard_concrete is not None else self.conv.out_channels


class FeatureExtractor(nn.Module):
    def __init__(self, conv_layers: list[tuple[int, int, int]], gate: bool) -> None:
        super().__init__()
        layers, in_ch = [], 1
        for out_ch, k, s in conv_layers:
            layers.append(ConvLayer(in_ch, out_ch, k, s, gate))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        # Soft gate values of the last conv layer, folded in by prune(): the next op is a
        # LayerNorm over channels, which a per-channel scale does not commute with.
        self.register_buffer("out_scale", torch.ones(in_ch), persistent=True)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2) * self.out_scale.to(x.dtype)  # (N, T, C)

    def frame_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            lengths = torch.div(lengths - layer.kernel, layer.stride, rounding_mode="floor") + 1
            lengths = lengths.clamp(min=0)
        return lengths


class Frontend(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        c_last = cfg.conv_layers[-1][0]
        self.feature_extractor = FeatureExtractor(cfg.conv_layers, cfg.prune_conv)
        self.post_extract_layer_norm = nn.LayerNorm(c_last)
        self.model_dim_proj = nn.Linear(c_last, cfg.model_dim)
        conv = nn.Conv1d(
            cfg.model_dim, cfg.model_dim, cfg.pos_conv_kernel,
            padding=cfg.pos_conv_kernel // 2, groups=cfg.pos_conv_groups,
        )
        self.pos_conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        self.remove_pad = cfg.pos_conv_kernel % 2 == 0
        self.grad_scale = cfg.feature_grad_scale
        # Run by PrunableCTC.forward in two halves (conv stack, then the rest), since the
        # padding mask is only known once the conv stack has fixed the frame count.


# --------------------------------------------------------------------------- encoder


class SelfAttention(nn.Module):
    def __init__(self, cfg: ArchConfig, num_heads: int) -> None:
        super().__init__()
        d, hd = cfg.model_dim, cfg.head_dim
        self.num_heads, self.head_dim, self.model_dim = num_heads, hd, d
        self.q_proj = nn.Linear(d, num_heads * hd)
        self.k_proj = nn.Linear(d, num_heads * hd)
        self.v_proj = nn.Linear(d, num_heads * hd)
        self.output_proj = nn.Linear(num_heads * hd, d)
        self.dropout_p = cfg.attn_dropout
        self.hard_concrete_for_heads = HardConcrete(num_heads, init_mean=0.01) if cfg.prune_heads else None
        self.hard_concrete_for_layer = HardConcrete(1, init_mean=0.01) if cfg.prune_attn_layer else None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        n, t, _ = x.shape
        shape = (n, t, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )  # (N, H, T, hd)
        if self.hard_concrete_for_heads is not None:
            out = out * self.hard_concrete_for_heads().to(out.dtype).view(1, -1, 1, 1)
        out = self.output_proj(out.transpose(1, 2).reshape(n, t, -1))
        if self.hard_concrete_for_layer is not None:
            out = out * self.hard_concrete_for_layer().to(out.dtype)
        return out

    def num_params(self, mode: str = "expected") -> torch.Tensor | float:
        h = self.hard_concrete_for_heads.count(mode) if self.hard_concrete_for_heads is not None else self.num_heads
        p = (self.model_dim + 1) * h * self.head_dim * 3 + h * self.head_dim * self.model_dim
        if self.hard_concrete_for_layer is not None:
            p = p * self.hard_concrete_for_layer.count(mode)
        return p + self.model_dim  # output bias: kept even when the block is pruned away


class FeedForward(nn.Module):
    def __init__(self, cfg: ArchConfig, dim: int) -> None:
        super().__init__()
        self.model_dim = cfg.model_dim
        self.inner_proj = nn.Linear(cfg.model_dim, dim)
        self.output_proj = nn.Linear(dim, cfg.model_dim)
        self.dropout = nn.Dropout(cfg.ffn_dropout)
        self.hard_concrete_for_intermediate = HardConcrete(dim, init_mean=0.5) if cfg.prune_interm else None
        self.hard_concrete_for_layer = HardConcrete(1, init_mean=0.01) if cfg.prune_ffn_layer else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.inner_proj(x)))
        if self.hard_concrete_for_intermediate is not None:
            x = x * self.hard_concrete_for_intermediate().to(x.dtype)
        x = self.output_proj(x)
        if self.hard_concrete_for_layer is not None:
            x = x * self.hard_concrete_for_layer().to(x.dtype)
        return x

    def num_params(self, mode: str = "expected") -> torch.Tensor | float:
        f = (
            self.hard_concrete_for_intermediate.count(mode)
            if self.hard_concrete_for_intermediate is not None
            else self.inner_proj.out_features
        )
        p = (self.model_dim + 1) * f + f * self.model_dim
        if self.hard_concrete_for_layer is not None:
            p = p * self.hard_concrete_for_layer.count(mode)
        return p + self.model_dim  # output bias: kept even when the block is pruned away


class EncoderLayer(nn.Module):
    """Pre-LN: x + attn(LN(x)), then x + ffn(LN(x)). Either half may be pruned away."""

    def __init__(self, cfg: ArchConfig, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        d = cfg.model_dim
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.self_attn = SelfAttention(cfg, num_heads) if num_heads > 0 else None
        self.ffn_layer_norm = nn.LayerNorm(d)
        self.ffn = FeedForward(cfg, ffn_dim) if ffn_dim > 0 else None
        # A half pruned to zero units still adds its output projection's bias to the residual
        # stream (output_proj(0) = bias), so prune() keeps that bias here to stay exact.
        self.attn_bias = nn.Parameter(torch.zeros(d)) if num_heads == 0 else None
        self.ffn_bias = nn.Parameter(torch.zeros(d)) if ffn_dim == 0 else None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        if self.self_attn is not None:
            x = x + self.self_attn(self.self_attn_layer_norm(x), attn_mask)
        elif self.attn_bias is not None:
            x = x + self.attn_bias.to(x.dtype)
        if self.ffn is not None:
            x = x + self.ffn(self.ffn_layer_norm(x))
        elif self.ffn_bias is not None:
            x = x + self.ffn_bias.to(x.dtype)
        return x

    def num_params(self, mode: str = "expected") -> torch.Tensor | float:
        d = self.self_attn_layer_norm.normalized_shape[0]
        p = 4 * d  # two LayerNorms
        if self.self_attn is not None:
            p = p + self.self_attn.num_params(mode)
        elif self.attn_bias is not None:
            p = p + d
        if self.ffn is not None:
            p = p + self.ffn.num_params(mode)
        elif self.ffn_bias is not None:
            p = p + d
        return p


class Encoder(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(cfg, h, f) for h, f in zip(cfg.num_heads, cfg.ffn_dims)
        )
        self.layer_norm = nn.LayerNorm(cfg.model_dim)


# --------------------------------------------------------------------------- model


class PrunableCTC(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_frontend = Frontend(cfg)
        self.encoder = Encoder(cfg)
        self.final_proj = nn.Linear(cfg.model_dim, cfg.vocab_size)
        self.gradient_checkpointing = False

    # ---- gates

    def hard_concretes(self) -> Iterator[HardConcrete]:
        for m in self.modules():
            if isinstance(m, HardConcrete):
                yield m

    def gate_parameters(self) -> list[nn.Parameter]:
        return [hc.log_alpha for hc in self.hard_concretes()]

    def weight_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if not n.endswith("log_alpha")]

    def _pin_masks(self) -> None:
        for hc in self.hard_concretes():
            hc.pinned = None
            hc.pinned = hc.sample()

    # ---- size

    def num_params(self, mode: str = "expected") -> torch.Tensor | float:
        """Prunable-encoder parameter count (frontend + transformer; not the CTC head).

        Matches DiariZen's accounting: conv layers (weights, bias, LayerNorm), the
        projection into the model dimension, the position conv, every encoder layer and
        the final LayerNorm. ``mode``: ``expected`` (E[L0], differentiable in the gates'
        ``log_alpha``), ``deterministic`` (what ``prune()`` would keep) or ``full`` (no pruning).
        """
        fe = self.encoder_frontend.feature_extractor
        total: torch.Tensor | float = 0
        in_ch: torch.Tensor | float = 1
        for layer in fe.layers:
            out_ch = layer.out_channels(mode)
            total = total + in_ch * out_ch * layer.kernel + out_ch * 3  # weights, bias, LN
            in_ch = out_ch
        fr = self.encoder_frontend
        total = total + in_ch * 2 + (in_ch + 1) * fr.model_dim_proj.out_features
        total = total + sum(p.numel() for p in fr.pos_conv.parameters())
        for layer in self.encoder.layers:
            total = total + layer.num_params(mode)
        return total + 2 * self.cfg.model_dim

    # ---- forward

    def frame_lengths(self, audio_lens: torch.Tensor) -> torch.Tensor:
        return self.encoder_frontend.feature_extractor.frame_lengths(audio_lens)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        hidden_layers: list[int] | None = None,
        return_logits: bool = True,
    ) -> dict:
        """``audio`` (N, S) already normalised per utterance; returns logits and hidden states.

        ``hidden_layers`` index the residual stream: 0 is the input to the first
        encoder layer, i the output of layer i, and ``num_layers`` the output of the
        final LayerNorm (what the CTC head reads).
        """
        # Gates stay pinned after forward returns: checkpointed layers recompute during
        # backward and must see the same masks. The next forward re-samples them.
        self._pin_masks()
        frame_lens = self.frame_lengths(audio_lens)
        x = self.encoder_frontend.feature_extractor(audio)
        t = x.shape[1]
        frame_lens = frame_lens.clamp(max=t)
        pad = torch.arange(t, device=x.device)[None, :] >= frame_lens[:, None]
        x = self._frontend_rest(x, pad)
        attn_mask = (~pad)[:, None, None, :] if bool(pad.any()) else None
        want = set(hidden_layers or [])
        hiddens: dict[int, torch.Tensor] = {}
        if 0 in want:
            hiddens[0] = x
        n_layers = len(self.encoder.layers)
        for i, layer in enumerate(self.encoder.layers, start=1):
            if self.training and self.cfg.layerdrop > 0 and torch.rand(()) < self.cfg.layerdrop:
                pass
            elif self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask)
            if i in want and i != n_layers:
                hiddens[i] = x
        x = self.encoder.layer_norm(x)
        if n_layers in want:
            hiddens[n_layers] = x
        out = {"hiddens": hiddens, "frame_lens": frame_lens}
        if return_logits:
            out["logits"] = self.final_proj(x)
        return out

    def _frontend_rest(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        fr = self.encoder_frontend
        if self.training and fr.grad_scale != 1.0:
            x = _GradScale.apply(x, fr.grad_scale)
        x = fr.post_extract_layer_norm(x)
        x = fr.model_dim_proj(x)
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        enc = fr.pos_conv(x.transpose(1, 2))
        if fr.remove_pad:
            enc = enc[:, :, :-1]
        return x + F.gelu(enc).transpose(1, 2)

    # ---- pruning

    @torch.no_grad()
    def prune(self) -> ArchConfig:
        """Fold the deterministic gates into the weights and drop zeroed structures, in place.

        Returns the new (gate-free) architecture. Rebuild with ``PrunableCTC(cfg)`` and
        load this model's ``state_dict()`` to get a clean module tree.
        """
        self.eval()
        fe = self.encoder_frontend.feature_extractor
        conv_cfg: list[tuple[int, int, int]] = []
        last_keep = None  # channels kept by the last conv layer, if it was gated
        for i, layer in enumerate(fe.layers):
            if layer.hard_concrete is not None:
                mask = layer.hard_concrete.sample()
                keep = mask.nonzero().squeeze(-1)
                if len(keep) == 0:
                    raise RuntimeError(f"conv layer {i} pruned to zero channels")
                _select(layer.conv, keep, out=True)
                _select_norm(layer.layer_norm, keep)
                if i + 1 < len(fe.layers):
                    nxt = fe.layers[i + 1].conv
                    nxt.weight.mul_(mask.view(1, -1, 1).to(nxt.weight.dtype))
                    _select(nxt, keep, out=False)
                else:
                    fe.out_scale = (fe.out_scale * mask.to(fe.out_scale.dtype))[keep].clone()
                    last_keep = keep
                layer.hard_concrete = None
            conv_cfg.append((layer.conv.out_channels, layer.kernel, layer.stride))
        if last_keep is not None:
            _select_norm(self.encoder_frontend.post_extract_layer_norm, last_keep)
            _select(self.encoder_frontend.model_dim_proj, last_keep, out=False)

        heads, ffns = [], []
        for layer in self.encoder.layers:
            att = layer.self_attn
            if att is not None:
                if att.hard_concrete_for_layer is not None:
                    g = att.hard_concrete_for_layer.sample()
                    att.output_proj.weight.mul_(g)
                    att.output_proj.bias.mul_(g)
                    att.hard_concrete_for_layer = None
                    if float(g) == 0.0:  # bias was scaled by g = 0 too: kept as zeros
                        layer.attn_bias = nn.Parameter(att.output_proj.bias.detach().clone())
                        layer.self_attn = att = None
            if att is not None and att.hard_concrete_for_heads is not None:
                hm = att.hard_concrete_for_heads.sample()
                full = hm.repeat_interleave(att.head_dim)
                idx = full.nonzero().squeeze(-1)
                att.hard_concrete_for_heads = None
                if len(idx) == 0:
                    layer.attn_bias = nn.Parameter(att.output_proj.bias.detach().clone())
                    layer.self_attn = att = None
                else:
                    for proj in (att.q_proj, att.k_proj, att.v_proj):
                        _select(proj, idx, out=True)
                    att.output_proj.weight.mul_(full.to(att.output_proj.weight.dtype))
                    _select(att.output_proj, idx, out=False)
                    att.num_heads = len(idx) // att.head_dim
            heads.append(att.num_heads if att is not None else 0)

            ffn = layer.ffn
            if ffn is not None:
                if ffn.hard_concrete_for_layer is not None:
                    g = ffn.hard_concrete_for_layer.sample()
                    ffn.output_proj.weight.mul_(g)
                    ffn.output_proj.bias.mul_(g)
                    ffn.hard_concrete_for_layer = None
                    if float(g) == 0.0:  # bias was scaled by g = 0 too: kept as zeros
                        layer.ffn_bias = nn.Parameter(ffn.output_proj.bias.detach().clone())
                        layer.ffn = ffn = None
            if ffn is not None and ffn.hard_concrete_for_intermediate is not None:
                m = ffn.hard_concrete_for_intermediate.sample()
                idx = m.nonzero().squeeze(-1)
                ffn.hard_concrete_for_intermediate = None
                if len(idx) == 0:
                    layer.ffn_bias = nn.Parameter(ffn.output_proj.bias.detach().clone())
                    layer.ffn = ffn = None
                else:
                    _select(ffn.inner_proj, idx, out=True)
                    ffn.output_proj.weight.mul_(m.to(ffn.output_proj.weight.dtype))
                    _select(ffn.output_proj, idx, out=False)
            ffns.append(ffn.inner_proj.out_features if ffn is not None else 0)

        d = self.cfg.to_dict()
        d.update(
            conv_layers=conv_cfg, num_heads=heads, ffn_dims=ffns,
            prune_conv=False, prune_heads=False, prune_interm=False,
            prune_attn_layer=False, prune_ffn_layer=False,
        )
        self.cfg = ArchConfig.from_dict(d)
        return self.cfg


def _select(mod: nn.Conv1d | nn.Linear, idx: torch.Tensor, *, out: bool) -> None:
    dim = 0 if out else 1
    idx = idx.to(mod.weight.device)
    mod.weight = nn.Parameter(mod.weight.index_select(dim, idx).clone())
    if out and mod.bias is not None:
        mod.bias = nn.Parameter(mod.bias.index_select(0, idx).clone())
    if isinstance(mod, nn.Linear):
        if out:
            mod.out_features = len(idx)
        else:
            mod.in_features = len(idx)
    else:
        if out:
            mod.out_channels = len(idx)
        else:
            mod.in_channels = len(idx)


def _select_norm(norm: nn.LayerNorm, idx: torch.Tensor) -> None:
    idx = idx.to(norm.weight.device)
    norm.weight = nn.Parameter(norm.weight.index_select(0, idx).clone())
    norm.bias = nn.Parameter(norm.bias.index_select(0, idx).clone())
    norm.normalized_shape = (len(idx),)


def normalize_audio_(audio: torch.Tensor, audio_lens: torch.Tensor) -> None:
    """Per-utterance zero-mean / unit-variance over the valid span, in place (Omnilingual's
    ``apply_audio_normalization``). Ported from streaming_asr's pipeline."""
    for i, n in enumerate(audio_lens.tolist()):
        seg = audio[i, :n]
        std = seg.std(unbiased=False).clamp_min(1e-7)
        seg.copy_((seg - seg.mean()) / std)


def load_model(path: str, map_location: str = "cpu") -> PrunableCTC:
    """Build a model from a ``{"arch": ..., "model": state_dict}`` checkpoint (any stage)."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = PrunableCTC(ArchConfig.from_dict(ck["arch"]))
    model.load_state_dict(ck["model"])
    return model
