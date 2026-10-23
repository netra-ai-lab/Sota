"""One trainer for every stage: CTC fine-tuning, pruning (distillation + L0), mask-frozen distillation.

    python -m sota_train.train configs/stage2_prune_s80.yaml [--set optim.max_steps=50 ...]

Stages (see README):
  finetune  CTC loss on the DDD + LibriSpeech mix (stage 1 teacher adaptation, stage 3 recovery)
            optionally + ``kd.weight`` x frame-level KL to a teacher's CTC output distribution
  prune     DPHuBERT / DiariZen: hidden-state distillation from a frozen teacher (L1 - cosine at
            ``distill_layers``) + augmented-Lagrangian L0 constraint toward ``target_sparsity``.
            Gates (log_alpha) and the multipliers lambda1/2 train at ``gate_lr``; the
            multipliers ascend (their gradient is negated).
  distill   distillation only, on an already-pruned (gate-free) student (stage 2b).

Resumable: ``latest.pt`` in ``save.dir`` (default: the run directory) is written every
``save.every_steps`` steps or ``save.every_minutes`` minutes, whichever comes first, on
SIGINT/SIGTERM (after the current step), on a crash outside the optimizer step, and at the end.
Relaunching the same command resumes from it (``--resume auto``, the default).
Exit codes: 0 finished, 3 stopped by signal (do not relaunch), anything else = crash.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from . import CACHE_DIR, PROJECT_ROOT
from .ckpt import atomic_save, git_commit, load as load_ckpt, release_host_memory, rng_state, set_rng_state
from .data import Batch, DataConfig, load_eval_set, make_train_loader
from .evaluate import evaluate
from sota.model import ArchConfig, PrunableCTC, normalize_audio_
from .offload import CPUOffloadAdamW, GPUAdamW
from sota.text import BLANK, CharVocab

EXIT_SIGNAL = 3

DEFAULTS: dict = {
    "stage": "finetune",
    "run_dir": None,
    "init": None,  # {"arch", "model"} checkpoint the student starts from
    "teacher": None,  # same format; required for prune/distill
    "seed": 1234,
    "model": {
        "pruning_units": "conv,head,interm",
        "layerdrop": 0.0,
        "ffn_dropout": 0.0,
        "gradient_checkpointing": True,
        "freeze_feature_extractor": False,
        "precision": "offload",  # offload: bf16 on GPU + fp32 master/Adam on CPU; amp: fp32 + bf16 autocast
    },
    "optim": {
        "max_steps": 10000,
        "grad_accum": 1,
        "lr": 2e-4,  # student weights (encoder + frontend)
        "head_lr": None,  # CTC head; None = same as lr
        "gate_lr": 2e-2,  # log_alpha and the Lagrange multipliers
        "weight_decay": 0.0,
        "warmup_steps": 500,
        "decay_frac": 0.0,  # final fraction of steps with linear decay to final_lr_scale
        "final_lr_scale": 0.05,
        "max_grad_norm": 10.0,
        "betas": [0.9, 0.98],
    },
    "prune": {
        "target_sparsity": 0.8,
        "sparsity_warmup_frac": 0.25,
        "distill_layers": [0, 12, 24, 36, 48],
        "l1_weight": 1.0,
        "cos_weight": 1.0,
        "l2_weight": 0.0,
        "ctc_weight": 0.0,  # optional CTC term on top of distillation
        "mask_snapshots": True,
    },
    "kd": {  # finetune: CTC + weight * frame-level KL to the teacher's output distribution
        "weight": 0.0,
        "temperature": 1.0,
    },
    "data": {},
    "eval": {
        "every_steps": 500,
        "sets": {"ddd_dev": {"manifest": "data/ddd_dev.jsonl", "limit": 400},
                 "libri_dev": {"manifest": "data/libri_dev-clean.jsonl", "limit": 400}},
        "select": ["ddd_dev.cer", "libri_dev.wer"],  # mean of these picks best.pt (lower is better)
        "select_after_step": 0,  # prune: only once the sparsity target is reached
        "max_batch_seconds": 200.0,
        "at_start": False,
        "final": True,  # evaluate once more at the end of the stage
    },
    "save": {
        "dir": None,  # default run_dir; may be on another disk
        "every_steps": 250,
        "every_minutes": 45.0,
        "best": True,
        "enabled": True,  # false for throwaway benchmark runs
        "moments_dtype": "fp32",  # bf16: smaller latest.pt (1B: ~8 GB instead of ~12 GB)
    },
    "log_every": 10,
}


# --------------------------------------------------------------------------- config


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "sets":
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = deep_update(DEFAULTS, yaml.safe_load(open(path)) or {})
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    if cfg["run_dir"] is None:
        cfg["run_dir"] = f"runs/{path.stem}"
    for key in ("init", "teacher"):  # ${SOTA_MODELS_DIR}/... in the stock-checkpoint configs
        if cfg[key]:
            cfg[key] = os.path.expandvars(cfg[key])
    if cfg["stage"] not in ("finetune", "prune", "distill"):
        raise ValueError(f"unknown stage {cfg['stage']}")
    if cfg["stage"] in ("prune", "distill") and not cfg["teacher"]:
        raise ValueError(f"stage {cfg['stage']} needs a teacher checkpoint")
    if cfg["kd"]["weight"] > 0 and (cfg["stage"] != "finetune" or not cfg["teacher"]):
        raise ValueError("kd.weight > 0 needs stage finetune and a teacher checkpoint")
    return cfg


#: config keys that define *what* is being trained; a resume refuses to continue if any changed
IDENTITY_KEYS = ["stage", "init", "teacher", "seed", "model.pruning_units", "model.precision",
                 "prune.target_sparsity", "prune.distill_layers", "data.en_fraction", "kd.weight"]


def get_key(cfg: dict, key: str):
    node = cfg
    for p in key.split("."):
        node = node.get(p) if isinstance(node, dict) else None
    return node


# --------------------------------------------------------------------------- pieces


def lr_scale(step: int, cfg: dict) -> float:
    o = cfg["optim"]
    if step < o["warmup_steps"]:
        return (step + 1) / o["warmup_steps"]
    decay_start = int(o["max_steps"] * (1 - o["decay_frac"]))
    if o["decay_frac"] > 0 and step >= decay_start:
        frac = (step - decay_start) / max(o["max_steps"] - decay_start, 1)
        return 1.0 + (o["final_lr_scale"] - 1.0) * frac
    return 1.0


def target_sparsity(step: int, cfg: dict) -> float:
    p = cfg["prune"]
    warm = max(int(cfg["optim"]["max_steps"] * p["sparsity_warmup_frac"]), 1)
    return p["target_sparsity"] * min(1.0, step / warm)


def distill_loss(s: dict[int, torch.Tensor], t: dict[int, torch.Tensor], layers: list[int],
                 valid: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    """L1 - cosine (DiariZen's DistillLoss) averaged over layers and valid frames."""
    p = cfg["prune"]
    m = valid.unsqueeze(-1).float()
    n_frames = valid.sum().clamp(min=1)
    l1 = l2 = cos = 0.0
    for i in layers:
        a, b = s[i].float(), t[i].float()
        dim = a.shape[-1]
        if p["l1_weight"]:
            l1 = l1 + ((a - b).abs() * m).sum() / (n_frames * dim)
        if p["l2_weight"]:
            l2 = l2 + ((a - b).pow(2) * m).sum() / (n_frames * dim)
        if p["cos_weight"]:
            cos = cos + (F.cosine_similarity(a, b, dim=-1) * valid).sum() / n_frames
    k = len(layers)
    loss = (p["l1_weight"] * l1 + p["l2_weight"] * l2 - p["cos_weight"] * cos) / k
    return loss, {"l1": _f(l1) / k, "l2": _f(l2) / k, "cos": _f(cos) / k}


def kd_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, valid: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL(teacher || student) per frame over the vocabulary, averaged over valid frames (Hinton T^2 scaling)."""
    s = F.log_softmax(s_logits.float() / temperature, dim=-1)
    t = F.log_softmax(t_logits.float() / temperature, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp(min=1) * temperature**2


def _f(x) -> float:
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def ctc_loss(logits: torch.Tensor, frame_lens: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, int]:
    lp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    loss = F.ctc_loss(lp, batch.targets, frame_lens, batch.target_lens, blank=BLANK,
                      reduction="sum", zero_infinity=True)
    return loss, int(batch.target_lens.sum())


def build_student(cfg: dict, device: torch.device) -> tuple[PrunableCTC, dict | None]:
    ck = load_ckpt(Path(cfg["init"]))
    arch = ArchConfig.from_dict(ck["arch"])
    mc = cfg["model"]
    d = arch.to_dict()
    d.update(layerdrop=mc["layerdrop"], ffn_dropout=mc["ffn_dropout"])
    arch = ArchConfig.from_dict(d)
    if cfg["stage"] == "prune":
        if any([arch.prune_conv, arch.prune_heads, arch.prune_interm]):
            raise ValueError("init checkpoint already has gates; prune from a gate-free checkpoint")
        arch = arch.with_pruning(mc["pruning_units"])
    model = PrunableCTC(arch)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    missing = [k for k in missing if not k.endswith("log_alpha")]
    if missing or unexpected:
        raise RuntimeError(f"init checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.gradient_checkpointing = mc["gradient_checkpointing"]
    if mc["freeze_feature_extractor"]:
        for p in model.encoder_frontend.feature_extractor.parameters():
            p.requires_grad_(False)
    if mc["precision"] == "offload":
        model.to(device=device, dtype=torch.bfloat16)
        for hc in model.hard_concretes():
            hc.log_alpha.data = hc.log_alpha.data.float()
    else:
        model.to(device)
    return model, ck


def build_teacher(cfg: dict, device: torch.device) -> PrunableCTC:
    ck = load_ckpt(Path(cfg["teacher"]))
    t = PrunableCTC(ArchConfig.from_dict(ck["arch"]))
    t.load_state_dict(ck["model"])
    t.to(device=device, dtype=torch.bfloat16).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


# --------------------------------------------------------------------------- trainer


class Trainer:
    def __init__(self, cfg: dict, resume: str) -> None:
        self.cfg = cfg
        self.root = PROJECT_ROOT
        self.run_dir = self.root / cfg["run_dir"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(cfg["save"]["dir"]) if cfg["save"]["dir"] else self.run_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda", 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        # pruned units get zero grads, so their offloaded Adam moments decay into denormals,
        # which made the CPU update ~2.4x slower as sparsity grew; flush them to zero
        torch.set_flush_denormal(True)
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg["seed"])

        self.stage = cfg["stage"]
        self.model, _ = build_student(cfg, self.device)
        self.kd_weight = cfg["kd"]["weight"]
        self.teacher = build_teacher(cfg, self.device) if self.stage in ("prune", "distill") or self.kd_weight > 0 else None
        self.layers = list(cfg["prune"]["distill_layers"])
        n_layers = self.model.cfg.num_layers
        if self.stage in ("prune", "distill") and max(self.layers) > n_layers:
            raise ValueError(f"distill layer {max(self.layers)} > {n_layers} layers")

        o = cfg["optim"]
        named = [(n, p) for n, p in self.model.weight_parameters() if p.requires_grad]
        groups = [
            {"names": set(), "prefixes": ("final_proj.",), "lr": o["head_lr"] or o["lr"], "weight_decay": 0.0},
            {"names": None, "lr": o["lr"], "weight_decay": o["weight_decay"]},
        ]
        Opt = CPUOffloadAdamW if cfg["model"]["precision"] == "offload" else GPUAdamW
        self.opt = Opt(named, groups, betas=tuple(o["betas"]))
        self.base_lrs = [g["lr"] for g in groups]

        self.gates = self.model.gate_parameters()
        self.lambdas = torch.nn.Parameter(torch.zeros(2, device=self.device))
        self.gate_opt = None
        if self.stage == "prune":
            self.gate_opt = torch.optim.AdamW(
                [{"params": self.gates, "lr": o["gate_lr"], "weight_decay": 0.0},
                 {"params": [self.lambdas], "lr": o["gate_lr"], "weight_decay": 0.0}])
        self.full_params = float(self.model.num_params("full"))

        self.data_cfg = DataConfig(**cfg["data"])
        self.vocab = CharVocab.load(self.root / self.data_cfg.vocab) if self.data_cfg.vocab else CharVocab.default()
        self.eval_sets = {
            name: load_eval_set(self.root / s["manifest"], self.vocab, Path(self.data_cfg.asr_root),
                                limit=s.get("limit"), cache_dir=CACHE_DIR)
            for name, s in cfg["eval"]["sets"].items()
        }

        self.step, self.epoch, self.done_units = 0, 0, set()
        self.best = math.inf
        self.history_best_step = None
        self.last_save_bytes = None
        self._maybe_resume(resume)
        release_host_memory()  # init/teacher/resume checkpoints were read into CPU RAM; DataLoader forks next

    # ---- checkpoint

    def _state(self) -> dict:
        master = self.opt.master_state_dict()
        sd = {k: master[k] if k in master else v.detach().cpu() for k, v in self.model.state_dict().items()}
        return {
            "stage": self.stage,
            "arch": self.model.cfg.to_dict(),
            "model": sd,
            "weight_opt": self.opt.state_dict(
                moments_dtype=torch.bfloat16 if self.cfg["save"]["moments_dtype"] == "bf16" else torch.float32),
            "gate_opt": self.gate_opt.state_dict() if self.gate_opt else None,
            "lambdas": self.lambdas.detach().cpu(),
            "step": self.step,
            "epoch": self.epoch,
            "done_units": sorted(self.done_units),
            "best": self.best,
            "best_step": self.history_best_step,
            "rng": rng_state(),
            "config": self.cfg,
            "git": git_commit(self.root),
            "time": time.time(),
        }

    def save_latest(self) -> None:
        self._last_save_time, self._last_save_step = time.time(), self.step
        if not self.cfg["save"]["enabled"]:
            return
        state = self._state()
        if self.last_save_bytes is None:
            n = sum(p.numel() for _, p in self.opt.named)
            self.last_save_bytes = n * (8 if self.cfg["save"]["moments_dtype"] == "bf16" else 12) + 2**28
        ok = atomic_save(state, self.ckpt_dir / "latest.pt", min_free_bytes=int(self.last_save_bytes * 1.05) + 2**30)
        if ok:
            self.last_save_bytes = (self.ckpt_dir / "latest.pt").stat().st_size
        del state
        release_host_memory()  # the save's bf16 moment copies (~4 GB for the 1B) go back to the OS

    def save_best(self, metrics: dict) -> None:
        sd = self._state()["model"]
        atomic_save({"arch": self.model.cfg.to_dict(), "model": sd, "step": self.step, "metrics": metrics,
                     "stage": self.stage, "config": self.cfg}, self.ckpt_dir / "best.pt")

    def _maybe_resume(self, resume: str) -> None:
        path = self.ckpt_dir / "latest.pt"
        if resume == "never" or not path.exists():
            if resume == "must":
                raise SystemExit(f"--resume must: {path} does not exist")
            print(f"starting fresh -> {self.run_dir}", flush=True)
            return
        ck = load_ckpt(path)
        saved_cfg = deep_update(DEFAULTS, ck["config"])  # keys added since the checkpoint take their defaults
        for key in IDENTITY_KEYS:
            if get_key(saved_cfg, key) != get_key(self.cfg, key):
                raise SystemExit(f"refusing to resume {path}: {key} changed "
                                 f"({get_key(saved_cfg, key)!r} -> {get_key(self.cfg, key)!r}); "
                                 f"use a new run_dir or --resume never")
        if ck["arch"] != self.model.cfg.to_dict():
            raise SystemExit("refusing to resume: architecture differs from the checkpoint")
        self.model.load_state_dict({k: v for k, v in ck["model"].items()}, strict=True)
        if self.cfg["model"]["precision"] == "offload":
            for hc in self.model.hard_concretes():
                hc.log_alpha.data = hc.log_alpha.data.float()
        self.opt.load_state_dict(ck["weight_opt"], ck["model"])
        if self.gate_opt is not None and ck["gate_opt"] is not None:
            self.gate_opt.load_state_dict(ck["gate_opt"])
        self.lambdas.data.copy_(ck["lambdas"].to(self.device))
        self.step, self.epoch = ck["step"], ck["epoch"]
        self.done_units = set(ck["done_units"])
        self.best, self.history_best_step = ck["best"], ck.get("best_step")
        set_rng_state(ck["rng"])
        self.last_save_bytes = path.stat().st_size
        print(f"resumed from {path}: step {self.step}, epoch {self.epoch}, "
              f"{len(self.done_units)} units done this epoch, best {self.best:.4f}", flush=True)
        del ck

    # ---- data

    def _batches(self):
        while True:
            loader = make_train_loader(self.data_cfg, self.root, seed=self.cfg["seed"], epoch=self.epoch,
                                       done_units=self.done_units)
            got = False
            for batch in loader:
                self.done_units.update(batch.done_units)
                if len(batch) == 0:
                    continue
                got = True
                yield batch
            if not got and not self.done_units:
                raise RuntimeError("training data stream is empty")
            self.epoch += 1
            self.done_units = set()
            self.log({"event": "epoch_end", "epoch": self.epoch})

    # ---- one step

    def _micro(self, batch: Batch, n_micro: int) -> dict:
        b = batch
        audio = b.audio.to(self.device, non_blocking=True)
        lens = b.audio_lens.to(self.device, non_blocking=True)
        b.targets, b.target_lens = b.targets.to(self.device), b.target_lens.to(self.device)
        normalize_audio_(audio, lens)
        offload = self.cfg["model"]["precision"] == "offload"
        dtype = torch.bfloat16 if offload else torch.float32
        want_ctc = self.stage == "finetune" or self.cfg["prune"]["ctc_weight"] > 0
        hidden = self.layers if self.stage in ("prune", "distill") else None
        stats: dict = {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not offload):
            out = self.model(audio.to(dtype), lens, hidden_layers=hidden, return_logits=want_ctc)
            loss = torch.zeros((), device=self.device)
            if want_ctc:
                ctc, ntok = ctc_loss(out["logits"], out["frame_lens"], b)
                ctc = ctc / max(ntok, 1)
                stats["ctc"] = _f(ctc)
                loss = loss + (ctc if self.stage == "finetune" else self.cfg["prune"]["ctc_weight"] * ctc)
            if self.teacher is not None:
                with torch.no_grad():
                    tout = self.teacher(audio.to(torch.bfloat16), lens, hidden_layers=hidden,
                                        return_logits=self.kd_weight > 0)
            if self.kd_weight > 0:
                t = out["logits"].shape[1]
                valid = torch.arange(t, device=self.device)[None, :] < out["frame_lens"][:, None]
                kd = kd_loss(out["logits"], tout["logits"], valid, self.cfg["kd"]["temperature"])
                stats["kd"] = _f(kd)
                loss = loss + self.kd_weight * kd
            if hidden is not None:
                t = out["hiddens"][self.layers[0]].shape[1]
                valid = torch.arange(t, device=self.device)[None, :] < out["frame_lens"][:, None]
                dl, parts = distill_loss(out["hiddens"], tout["hiddens"], self.layers, valid, self.cfg)
                stats.update(distill=_f(dl), **parts)
                loss = loss + dl
            if self.stage == "prune":
                expected = 1.0 - self.model.num_params("expected") / self.full_params
                tgt = target_sparsity(self.step, self.cfg)
                diff = expected - tgt
                reg = self.lambdas[0] * diff + self.lambdas[1] * diff * diff
                loss = loss + reg
                stats.update(reg=_f(reg), expected_sparsity=_f(expected), target_sparsity=tgt)
        (loss / n_micro).backward()
        stats["loss"] = _f(loss)
        return stats

    def train_step(self, batches) -> dict:
        o = self.cfg["optim"]
        scale = lr_scale(self.step, self.cfg)
        self.opt.set_lr([lr * scale for lr in self.base_lrs])
        t0 = time.time()
        stats_all: list[dict] = []
        secs = secs_en = 0.0
        for _ in range(o["grad_accum"]):
            batch = next(batches)
            secs += batch.seconds
            secs_en += sum(int(n) for n, l in zip(batch.audio_lens, batch.langs) if l == "en") / 16000
            stats_all.append(self._micro(batch, o["grad_accum"]))
        gnorm = self.opt.grad_norm()
        clip = min(1.0, o["max_grad_norm"] / (gnorm + 1e-6)) if o["max_grad_norm"] else 1.0
        self._in_opt_step = True
        if math.isfinite(gnorm):
            self.opt.step(grad_scale=clip)
            if self.gate_opt is not None:
                if self.lambdas.grad is not None:
                    self.lambdas.grad.neg_()  # gradient *ascent* on the multipliers
                self.gate_opt.step()
        else:
            print(f"step {self.step}: non-finite grad norm, update skipped", flush=True)
        self.opt.zero_grad()
        if self.gate_opt is not None:
            self.gate_opt.zero_grad(set_to_none=True)
        self._in_opt_step = False
        self.step += 1
        stats = {k: sum(s[k] for s in stats_all if k in s) / len(stats_all) for k in stats_all[0]}
        stats.update(step=self.step, lr=self.base_lrs[1] * scale, grad_norm=gnorm, audio_s=secs,
                     en_frac=secs_en / max(secs, 1e-9),
                     step_time=time.time() - t0, epoch=self.epoch,
                     mem_gb=torch.cuda.max_memory_allocated(self.device) / 2**30)
        if self.stage == "prune":
            stats.update(lambda1=float(self.lambdas[0].detach()), lambda2=float(self.lambdas[1].detach()))
        return stats

    # ---- eval / logging

    def log(self, rec: dict) -> None:
        rec = {"time": round(time.time(), 1), **rec}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def run_eval(self) -> dict:
        res = evaluate(self.model, self.eval_sets, self.vocab, device=self.device,
                       max_batch_seconds=self.cfg["eval"]["max_batch_seconds"], examples=2)
        flat = {f"{n}.{k}": v for n, r in res.items() for k, v in r.items() if k != "examples"}
        if self.stage == "prune":
            self.model.eval()
            flat["params_deterministic"] = float(self.model.num_params("deterministic"))
            flat["sparsity_deterministic"] = 1 - flat["params_deterministic"] / self.full_params
            self.model.train()
            if self.cfg["prune"]["mask_snapshots"]:
                snap = {n: p.detach().cpu() for n, p in self.model.named_parameters() if n.endswith("log_alpha")}
                (self.run_dir / "masks").mkdir(exist_ok=True)
                torch.save(snap, self.run_dir / "masks" / f"step_{self.step:06d}.pt")
        sel = [flat[k] for k in self.cfg["eval"]["select"] if k in flat]
        metric = sum(sel) / len(sel) if sel else math.inf
        flat["select_metric"] = metric
        self.log({"event": "eval", "step": self.step, **flat,
                  "examples": {n: r.get("examples") for n, r in res.items()}})
        print(f"[eval step {self.step}] " + " ".join(f"{k}={v:.4f}" for k, v in flat.items()
                                                      if isinstance(v, float)), flush=True)
        if (self.cfg["save"]["best"] and self.cfg["save"]["enabled"] and self.step >= self.cfg["eval"]["select_after_step"]
                and metric < self.best):
            self.best, self.history_best_step = metric, self.step
            self.save_best(flat)
        return flat

    # ---- main loop

    def fit(self) -> int:
        stop = {"flag": False}

        def on_signal(signum, frame):  # noqa: ARG001
            print(f"\nsignal {signum}: finishing this step, then checkpointing", flush=True)
            stop["flag"] = True

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        sv, ev = self.cfg["save"], self.cfg["eval"]
        self._last_save_time, self._last_save_step = time.time(), self.step
        self._in_opt_step = False
        self.log({"event": "start", "step": self.step, "config": self.cfg, "git": git_commit(self.root)})
        batches = self._batches()
        self.model.train()
        if ev["at_start"] and self.step == 0:
            self.run_eval()
        ooms = 0
        try:
            while self.step < self.cfg["optim"]["max_steps"] and not stop["flag"]:
                try:
                    stats = self.train_step(batches)
                    ooms = 0
                except torch.OutOfMemoryError:
                    if self._in_opt_step:
                        raise
                    # a rare long/odd batch: drop it, free memory, keep going
                    ooms += 1
                    self.opt.zero_grad()
                    if self.gate_opt is not None:
                        self.gate_opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"step {self.step}: CUDA OOM, batch skipped ({ooms} in a row)", flush=True)
                    self.log({"event": "oom", "step": self.step})
                    if ooms >= 5:
                        raise
                    continue
                if self.step % self.cfg["log_every"] == 0 or self.step == 1:
                    self.log({"event": "train", **stats})
                    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in stats.items()), flush=True)
                if self.step % ev["every_steps"] == 0:
                    self.run_eval()
                due_steps = self.step - self._last_save_step >= sv["every_steps"]
                due_time = time.time() - self._last_save_time >= sv["every_minutes"] * 60
                if due_steps or due_time:
                    self.save_latest()
        except Exception:
            traceback.print_exc()
            if not self._in_opt_step and self.step > self._last_save_step:
                print("crash outside the optimizer step: saving the last completed step", flush=True)
                self.opt.zero_grad()
                self.save_latest()
            return 1
        if self.step > self._last_save_step:
            self.save_latest()
        if stop["flag"] and self.step < self.cfg["optim"]["max_steps"]:
            print(f"stopped at step {self.step}; relaunch the same command to resume", flush=True)
            return EXIT_SIGNAL
        if ev["final"] and self.step % ev["every_steps"] != 0:
            self.run_eval()
        print(f"done: step {self.step}, best select_metric {self.best:.4f} at step {self.history_best_step}", flush=True)
        (self.run_dir / "DONE").write_text(json.dumps({"step": self.step, "best": self.best,
                                                        "best_step": self.history_best_step}))
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. optim.max_steps=50")
    ap.add_argument("--resume", choices=["auto", "never", "must"], default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    sys.exit(Trainer(cfg, args.resume).fit())


if __name__ == "__main__":
    main()
