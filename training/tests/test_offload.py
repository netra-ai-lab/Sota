import pytest
import torch

from sota_train.offload import CPUOffloadAdamW

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_offload_matches_torch_adamw():
    torch.manual_seed(0)
    shapes = [(64, 32), (32,), (7, 5, 3)]
    init = [torch.randn(s).to(torch.bfloat16).float() for s in shapes]  # start from bf16-exact values
    gpu = [torch.nn.Parameter(t.clone().to("cuda", torch.bfloat16)) for t in init]
    ref = [torch.nn.Parameter(t.clone()) for t in init]
    groups = [{"names": {"p0"}, "lr": 1e-2, "weight_decay": 0.0},
              {"names": None, "lr": 3e-3, "weight_decay": 0.1}]
    off = CPUOffloadAdamW([(f"p{i}", p) for i, p in enumerate(gpu)], groups, betas=(0.9, 0.98), eps=1e-8)
    opt = torch.optim.AdamW([{"params": [ref[0]], "lr": 1e-2, "weight_decay": 0.0},
                             {"params": ref[1:], "lr": 3e-3, "weight_decay": 0.1}], betas=(0.9, 0.98), eps=1e-8)
    for step in range(20):
        for p, r in zip(gpu, ref):
            g = torch.randn(r.shape).to(torch.bfloat16)  # bf16-representable, so both see the same grad
            p.grad = g.to("cuda")
            r.grad = g.float() * 0.5
        off.step(grad_scale=0.5)
        opt.step()
        off.zero_grad()
    master = off.master_state_dict()
    for i, r in enumerate(ref):
        torch.testing.assert_close(master[f"p{i}"], r.detach(), atol=1e-5, rtol=1e-5)
        # the GPU copy is the bf16 rounding of the master
        torch.testing.assert_close(gpu[i].detach().float().cpu(), master[f"p{i}"].to(torch.bfloat16).float())


@cuda
def test_offload_state_roundtrip():
    p = torch.nn.Parameter(torch.randn(10, device="cuda", dtype=torch.bfloat16))
    off = CPUOffloadAdamW([("w", p)], [{"names": None, "lr": 1e-2, "weight_decay": 0.0}])
    for _ in range(3):
        p.grad = torch.randn_like(p)
        off.step()
    sd, master = off.state_dict(), {k: v.clone() for k, v in off.master_state_dict().items()}
    q = torch.nn.Parameter(torch.zeros(10, device="cuda", dtype=torch.bfloat16))
    off2 = CPUOffloadAdamW([("w", q)], [{"names": None, "lr": 1e-2, "weight_decay": 0.0}])
    off2.load_state_dict(sd, master)
    g = torch.randn_like(p)
    p.grad, q.grad = g.clone(), g.clone()
    off.step()
    off2.step()
    assert torch.equal(off.master[0], off2.master[0])
    assert torch.equal(p, q)
