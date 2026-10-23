"""Resume round-trip of the real trainer on a tiny model and real DDD data (needs CUDA + the DDD shards)."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from sota_train import ASR_ROOT, PROJECT_ROOT
from sota.model import ArchConfig, PrunableCTC
from sota_train.train import Trainer, load_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (PROJECT_ROOT / "data/ddd_units.json").exists() or not ASR_ROOT.exists(),
    reason="needs CUDA and the prepared DDD index",
)


def tiny_ckpt(path: Path) -> None:
    torch.manual_seed(0)
    cfg = ArchConfig(conv_layers=[(32, 10, 5), (32, 3, 2), (32, 3, 2), (32, 3, 2), (32, 3, 2), (32, 2, 2), (32, 2, 2)],
                     num_heads=[2, 2], ffn_dims=[64, 64], model_dim=32, head_dim=16,
                     pos_conv_kernel=8, pos_conv_groups=2, vocab_size=9812)
    m = PrunableCTC(cfg)
    torch.save({"arch": cfg.to_dict(), "model": m.state_dict()}, path)


def write_cfg(tmp: Path, **over) -> Path:
    cfg = {
        "stage": "prune", "run_dir": str(tmp / "run"),
        "init": str(tmp / "tiny.pt"), "teacher": str(tmp / "tiny.pt"),
        "model": {"pruning_units": "head,interm", "precision": "offload", "gradient_checkpointing": True},
        "optim": {"max_steps": 40, "warmup_steps": 2, "lr": 1e-3},
        "prune": {"distill_layers": [0, 1, 2], "target_sparsity": 0.5, "sparsity_warmup_frac": 0.5},
        "data": {"en_fraction": 0.0, "max_batch_seconds": 30.0, "buffer_size": 24, "num_workers": 1},
        "eval": {"every_steps": 100, "sets": {"ddd_dev": {"manifest": "data/ddd_dev.jsonl", "limit": 4}},
                 "select": ["ddd_dev.cer"]},
        "save": {"every_steps": 15, "every_minutes": 60},
        "log_every": 1,
    }
    for k, v in over.items():
        node = cfg
        *parts, last = k.split(".")
        for p in parts:
            node = node[p]
        node[last] = v
    path = tmp / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_resume_roundtrip(tmp_path):
    tiny_ckpt(tmp_path / "tiny.pt")
    t1 = Trainer(load_config(write_cfg(tmp_path), []), "auto")
    assert t1.fit() == 0 and t1.step == 40
    saved = torch.load(tmp_path / "run/latest.pt", weights_only=False)
    assert saved["step"] == 40 and saved["done_units"], "progress through the data must be recorded"
    master1 = {k: v.clone() for k, v in t1.opt.master_state_dict().items()}
    lambdas1 = t1.lambdas.detach().cpu().clone()
    alphas1 = {n: p.detach().cpu().clone() for n, p in t1.model.named_parameters() if n.endswith("log_alpha")}
    done1 = set(t1.done_units)
    del t1
    torch.cuda.empty_cache()

    t2 = Trainer(load_config(write_cfg(tmp_path, **{"optim.max_steps": 45}), []), "must")
    assert t2.step == 40 and t2.done_units == done1
    for k, v in t2.opt.master_state_dict().items():
        assert torch.equal(v, master1[k]), k
    assert torch.equal(t2.lambdas.detach().cpu(), lambdas1)
    for n, p in t2.model.named_parameters():
        if n.endswith("log_alpha"):
            assert torch.equal(p.detach().cpu(), alphas1[n]), n
        else:  # GPU copy is the bf16 image of the fp32 master
            assert torch.equal(p.detach().cpu(), master1[n].to(p.dtype)), n
    assert t2.opt.step_count == 40
    assert t2.fit() == 0 and t2.step == 45
    assert t2.done_units >= done1 or t2.epoch > 0

    recs = [json.loads(l) for l in open(tmp_path / "run/metrics.jsonl")]
    steps = [r["step"] for r in recs if r.get("event") == "train"]
    assert steps == list(range(1, 46))
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in recs if r.get("event") == "train")


def test_resume_refuses_changed_identity(tmp_path):
    tiny_ckpt(tmp_path / "tiny.pt")
    t1 = Trainer(load_config(write_cfg(tmp_path, **{"optim.max_steps": 2}), []), "auto")
    assert t1.fit() == 0
    del t1
    with pytest.raises(SystemExit, match="target_sparsity changed"):
        Trainer(load_config(write_cfg(tmp_path, **{"prune.target_sparsity": 0.6}), []), "auto")
