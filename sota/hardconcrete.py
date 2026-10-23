"""Hard Concrete gates for L0 structured pruning.

Ported from DiariZen (``diarizen/models/module/wav2vec2/hardconcrete.py``, MIT,
Jiangyu Han / BUT), itself from DPHuBERT and FLOP (asappresearch/flop). One change:
a sampled mask can be *pinned* for the duration of a forward pass. With
per-layer gradient checkpointing a layer's forward runs twice (once for real,
once to recompute activations during backward); sampling inside the layer would
draw a different mask the second time and silently corrupt the gradients. The
model therefore samples every gate once, before the forward pass, outside any
checkpointed region, and each layer reads the pinned tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class HardConcrete(nn.Module):
    """A vector of ``n_in`` stochastic gates in [0, 1] with a closed-form expected L0 norm.

    Training: gates are sampled with the reparameterisation trick, so exact zeros
    and ones have non-zero probability and ``log_alpha`` gets gradients.
    Eval: a deterministic mask keeps the ``round(E[L0])`` largest soft values.
    """

    def __init__(
        self,
        n_in: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        temperature: float = 2 / 3,
        limit_l: float = -0.1,
        limit_r: float = 1.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.limit_l = limit_l
        self.limit_r = limit_r
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.eps = eps
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.compiled_mask: torch.Tensor | None = None
        self.pinned: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.compiled_mask = None
        # init_mean is the initial *drop* rate: 0.01 starts with nearly every unit kept.
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Expected number of non-zero gates (differentiable in ``log_alpha``)."""
        return (self.log_alpha + self.bias).sigmoid().sum()

    def count(self, mode: str = "expected") -> torch.Tensor | int:
        """Number of kept units: ``expected`` (E[L0], differentiable), ``deterministic``
        (non-zeros of the eval-time mask) or ``full`` (all of them)."""
        if mode == "expected":
            return self.l0_norm()
        if mode == "full":
            return self.n_in
        if mode == "deterministic":
            return self.n_in - round(self.n_in - self.l0_norm().item())
        raise ValueError(mode)

    def sample(self) -> torch.Tensor:
        if self.training:
            self.compiled_mask = None
            u = self.log_alpha.new_empty(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            return s.clamp(min=0.0, max=1.0)
        if self.compiled_mask is None:
            num_zeros = round(self.n_in - self.l0_norm().item())
            # Expected value of each gate, with DPHuBERT's empirically chosen 0.8.
            soft = torch.sigmoid(self.log_alpha.detach() / self.beta * 0.8)
            if num_zeros > 0:
                _, idx = torch.topk(soft, k=num_zeros, largest=False)
                soft[idx] = 0.0
            self.compiled_mask = soft
        return self.compiled_mask

    def forward(self) -> torch.Tensor:
        return self.pinned if self.pinned is not None else self.sample()

    def extra_repr(self) -> str:
        return f"n_in={self.n_in}"
