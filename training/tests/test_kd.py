"""Logit distillation (kd) for the finetune stage: loss values, config checks, a short real run."""

import json

import pytest
import torch

from sota_train.train import Trainer, kd_loss, load_config
from test_resume import pytestmark as needs_cuda_and_data
from test_resume import tiny_ckpt, write_cfg


def test_kd_loss_zero_for_identical_and_masks_padding():
    torch.manual_seed(0)
    s, t = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    assert kd_loss(t, t, valid, 1.0).abs() < 1e-6
    assert kd_loss(s, t, valid, 1.0) > 0
    # padded frames do not count
    s2 = s.clone()
    s2[:, 3:] = 100.0
    valid[:, 3:] = False
    assert torch.allclose(kd_loss(s2, t, valid, 1.0), kd_loss(s[:, :3], t[:, :3], valid[:, :3], 1.0))


def test_kd_needs_finetune_and_teacher(tmp_path):
    with pytest.raises(ValueError, match="kd.weight"):
        load_config(write_cfg(tmp_path, stage="finetune", teacher=None, **{"kd": {"weight": 1.0}}), [])
    with pytest.raises(ValueError, match="kd.weight"):
        load_config(write_cfg(tmp_path, **{"kd": {"weight": 1.0}}), [])


@needs_cuda_and_data
def test_finetune_with_kd_runs(tmp_path):
    tiny_ckpt(tmp_path / "tiny.pt")
    cfg = load_config(write_cfg(tmp_path, stage="finetune", **{"kd": {"weight": 1.0, "temperature": 1.0},
                                                               "model.precision": "amp",
                                                               "optim.max_steps": 6}), [])
    tr = Trainer(cfg, "never")
    assert tr.teacher is not None
    assert tr.fit() == 0
    rows = [r for r in map(json.loads, open(tmp_path / "run/metrics.jsonl")) if r.get("event") == "train"]
    assert rows and all("kd" in r and "ctc" in r for r in rows)
