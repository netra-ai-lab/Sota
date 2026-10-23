"""Optimisers for the student's weights.

``CPUOffloadAdamW`` is a minimal ZeRO-Offload: the GPU holds the model in bf16 (weights
and gradients, 2 + 2 bytes per parameter); the fp32 master weights and both Adam
moments (12 bytes per parameter) live in CPU RAM. Each step copies the bf16 grads
down in chunks, runs PyTorch's fused CPU AdamW kernel on the fp32 master, and copies
the updated weights back up as bf16. For the 1B student that moves ~11.6 GB of
optimizer state off a 12 GB card.

``GPUAdamW`` is the ordinary path (fp32 weights on the GPU, bf16 autocast) for models
small enough to fit, i.e. the pruned ~190M student. Both expose the same interface,
so the trainer and the checkpoint code do not care which one is in use.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _Base:
    def __init__(self, named_params: list[tuple[str, nn.Parameter]], groups: list[dict]) -> None:
        """``groups``: [{"names": set|None, "lr": float, "weight_decay": float}], first match wins;
        ``names=None`` matches everything."""
        self.named = named_params
        self.group_of = []
        for name, _ in named_params:
            for gi, g in enumerate(groups):
                if g["names"] is None or name in g["names"] or any(name.startswith(p) for p in g.get("prefixes", ())):
                    self.group_of.append(gi)
                    break
            else:
                raise ValueError(f"parameter {name} matches no optimizer group")
        self.groups = groups

    def set_lr(self, lrs: list[float]) -> None:
        raise NotImplementedError

    def grad_norm(self) -> float:
        sq = None
        for _, p in self.named:
            if p.grad is not None:
                s = p.grad.detach().float().pow(2).sum()
                sq = s if sq is None else sq + s
        return math.sqrt(float(sq)) if sq is not None else 0.0

    def zero_grad(self) -> None:
        for _, p in self.named:
            p.grad = None


class CPUOffloadAdamW(_Base):
    def __init__(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        groups: list[dict],
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        chunk_numel: int = 64 * 1024 * 1024,
    ) -> None:
        super().__init__(named_params, groups)
        self.betas, self.eps = betas, eps
        self.chunk_numel = chunk_numel
        self.master = [p.detach().to("cpu", torch.float32).pin_memory() for _, p in named_params]
        self.exp_avg = [torch.zeros_like(m) for m in self.master]
        self.exp_avg_sq = [torch.zeros_like(m) for m in self.master]
        self.lrs = [g["lr"] for g in groups]
        self.step_count = 0
        # bf16 staging buffers for the device<->host copies, sized to the largest chunk
        biggest = max(m.numel() for m in self.master)
        n = max(chunk_numel, biggest)
        self._down = torch.empty(n, dtype=torch.bfloat16).pin_memory()
        self._up = torch.empty(n, dtype=torch.bfloat16).pin_memory()
        self._grad32 = torch.empty(n, dtype=torch.float32)

    def set_lr(self, lrs: list[float]) -> None:
        self.lrs = list(lrs)

    @torch.no_grad()
    def step(self, grad_scale: float = 1.0) -> None:
        """One AdamW update; ``grad_scale`` multiplies the gradients (clipping, averaging)."""
        self.step_count += 1
        b1, b2 = self.betas
        bc1 = 1 - b1**self.step_count
        bc2 = 1 - b2**self.step_count
        for i, (_, p) in enumerate(self.named):
            if p.grad is None:
                continue
            g = self.groups[self.group_of[i]]
            lr, wd = self.lrs[self.group_of[i]], g["weight_decay"]
            n = p.numel()
            down = self._down[:n].view_as(p)
            down.copy_(p.grad, non_blocking=False)
            grad = self._grad32[:n].view_as(p)
            grad.copy_(down)
            if grad_scale != 1.0:
                grad.mul_(grad_scale)
            m, v, w = self.exp_avg[i], self.exp_avg_sq[i], self.master[i]
            # AdamW (decoupled weight decay), same maths as torch.optim.AdamW
            w.mul_(1 - lr * wd)
            m.lerp_(grad, 1 - b1)
            v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
            w.addcdiv_(m, denom, value=-lr / bc1)
            up = self._up[:n].view_as(p)
            up.copy_(w)
            p.data.copy_(up, non_blocking=False)

    def master_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: m for (name, _), m in zip(self.named, self.master)}

    def state_dict(self, moments_dtype: torch.dtype = torch.float32) -> dict:
        """``moments_dtype=torch.bfloat16`` halves the moments' share of the checkpoint (8 of the
        12 bytes per parameter) at the cost of rounding them once on resume; the fp32 master
        weights are always saved exactly."""
        return {
            "kind": "offload",
            "step": self.step_count,
            "names": [n for n, _ in self.named],
            "exp_avg": [m.to(moments_dtype) for m in self.exp_avg],
            "exp_avg_sq": [v.to(moments_dtype) for v in self.exp_avg_sq],
        }

    def load_state_dict(self, sd: dict, master: dict[str, torch.Tensor]) -> None:
        names = [n for n, _ in self.named]
        if sd["kind"] != "offload" or sd["names"] != names:
            raise ValueError("optimizer state does not match this model's parameters")
        self.step_count = sd["step"]
        for i, (name, p) in enumerate(self.named):
            self.exp_avg[i].copy_(sd["exp_avg"][i])
            self.exp_avg_sq[i].copy_(sd["exp_avg_sq"][i])
            self.master[i].copy_(master[name])
            p.data.copy_(self.master[i].to(p.dtype))


class GPUAdamW(_Base):
    def __init__(self, named_params, groups, betas=(0.9, 0.98), eps=1e-8) -> None:
        super().__init__(named_params, groups)
        pgs = [{"params": [], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in groups]
        for i, (_, p) in enumerate(named_params):
            pgs[self.group_of[i]]["params"].append(p)
        self.opt = torch.optim.AdamW([g for g in pgs if g["params"]], betas=betas, eps=eps, fused=True)
        self._live = [gi for gi, g in enumerate(pgs) if g["params"]]

    def set_lr(self, lrs: list[float]) -> None:
        for pg, gi in zip(self.opt.param_groups, self._live):
            pg["lr"] = lrs[gi]

    def step(self, grad_scale: float = 1.0) -> None:
        if grad_scale != 1.0:
            for _, p in self.named:
                if p.grad is not None:
                    p.grad.mul_(grad_scale)
        self.opt.step()

    def master_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: p.detach().float().cpu() for name, p in self.named}

    def state_dict(self, moments_dtype: torch.dtype = torch.float32) -> dict:  # noqa: ARG002 (small model)
        return {"kind": "gpu", "names": [n for n, _ in self.named], "opt": self.opt.state_dict()}

    def load_state_dict(self, sd: dict, master: dict[str, torch.Tensor]) -> None:
        if sd["kind"] != "gpu" or sd["names"] != [n for n, _ in self.named]:
            raise ValueError("optimizer state does not match this model's parameters")
        for name, p in self.named:
            p.data.copy_(master[name].to(p.device, p.dtype))
        self.opt.load_state_dict(sd["opt"])
