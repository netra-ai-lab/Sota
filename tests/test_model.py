import copy

import pytest
import torch

from sota.model import ArchConfig, PrunableCTC


def tiny(vocab: int = 20, units: str = "head,interm,attlayer,ffnlayer") -> ArchConfig:
    cfg = ArchConfig(
        conv_layers=[(16, 10, 5), (16, 3, 2), (12, 2, 2)],
        num_heads=[4, 4, 4], ffn_dims=[64, 64, 64], model_dim=32, head_dim=8,
        pos_conv_kernel=4, pos_conv_groups=2, vocab_size=vocab,
    )
    return cfg.with_pruning(units) if units else cfg


def randomize_gates(model: PrunableCTC, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for hc in model.hard_concretes():
        hc.log_alpha.data = torch.randn(hc.n_in, generator=g) * 3
    # make sure at least one whole sub-layer disappears and one survives
    model.encoder.layers[1].ffn.hard_concrete_for_layer.log_alpha.data.fill_(-10.0)
    model.encoder.layers[0].self_attn.hard_concrete_for_layer.log_alpha.data.fill_(10.0)


def batch(n: int = 3, seconds: float = 1.0):
    g = torch.Generator().manual_seed(1)
    lens = torch.tensor([int(16000 * seconds * f) for f in (1.0, 0.7, 0.45)][:n])
    audio = torch.randn(n, int(lens.max()), generator=g)
    for i, l in enumerate(lens.tolist()):
        audio[i, l:] = 0
    return audio, lens


def test_full_count_equals_real_parameter_count():
    m = PrunableCTC(tiny(units=""))
    real = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("final_proj."))
    assert m.num_params("full") == real


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prune_reproduces_masked_model(seed):
    torch.manual_seed(seed)
    m = PrunableCTC(tiny())
    randomize_gates(m, seed)
    m.eval()
    audio, lens = batch()
    with torch.no_grad():
        ref = m(audio, lens, hidden_layers=[0, 1, 2, 3])
    expected_params = m.num_params("deterministic")
    pruned = copy.deepcopy(m)
    new_cfg = pruned.prune()
    rebuilt = PrunableCTC(new_cfg)
    rebuilt.load_state_dict(pruned.state_dict())
    rebuilt.eval()
    assert rebuilt.num_params("full") == expected_params
    assert not list(rebuilt.hard_concretes())
    with torch.no_grad():
        out = rebuilt(audio, lens, hidden_layers=[0, 1, 2, 3])
    assert torch.equal(out["frame_lens"], ref["frame_lens"])
    for i, n in enumerate(ref["frame_lens"].tolist()):
        torch.testing.assert_close(out["logits"][i, :n], ref["logits"][i, :n], atol=1e-4, rtol=1e-4)
        for k in ref["hiddens"]:
            torch.testing.assert_close(out["hiddens"][k][i, :n], ref["hiddens"][k][i, :n], atol=1e-4, rtol=1e-4)


def test_conv_pruning_is_only_approximate():
    """The conv stack normalises over channels (LayerNorm mode), so dropping channels shifts the
    statistics: why ``conv`` is not in the default pruning units."""
    m = PrunableCTC(tiny(units="conv"))
    for hc in m.hard_concretes():
        hc.log_alpha.data = torch.randn(hc.n_in) * 3
    m.eval()
    audio, lens = batch()
    with torch.no_grad():
        ref = m(audio, lens)["logits"]
    pruned = copy.deepcopy(m)
    rebuilt = PrunableCTC(pruned.prune())
    rebuilt.load_state_dict(pruned.state_dict())
    with torch.no_grad():
        out = rebuilt.eval()(audio, lens)["logits"]
    assert not torch.allclose(out, ref, atol=1e-3)


def test_open_gates_change_nothing_after_prune():
    m = PrunableCTC(tiny())
    for hc in m.hard_concretes():
        hc.log_alpha.data.fill_(20.0)  # deterministic mask ~1 everywhere, nothing removed
    m.eval()
    plain = PrunableCTC(tiny(units=""))
    plain.load_state_dict({k: v for k, v in m.state_dict().items() if not k.endswith("log_alpha")})
    plain.eval()
    audio, lens = batch()
    with torch.no_grad():
        a, b = m(audio, lens)["logits"], plain(audio, lens)["logits"]
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)
    cfg = copy.deepcopy(m).prune()
    assert cfg.num_heads == [4, 4, 4] and cfg.ffn_dims == [64, 64, 64]


def test_masks_pinned_once_per_forward_with_checkpointing():
    """With checkpointing, recompute must reuse the same sampled gates: gradients then match
    a non-checkpointed run with the same RNG state."""
    torch.manual_seed(0)
    m = PrunableCTC(tiny()).train()
    audio, lens = batch()
    grads = []
    for ckpt in (False, True):
        m.gradient_checkpointing = ckpt
        m.zero_grad()
        torch.manual_seed(123)
        out = m(audio, lens, hidden_layers=[3])
        out["logits"].float().pow(2).mean().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], atol=1e-5, rtol=1e-4)


def test_prune_keeps_bias_of_blocks_with_every_unit_closed():
    # Every head of layer 0 and every FFN unit of layer 1 closed by their unit gates (no layer
    # gates): the masked model still adds those output projections' biases, so the cut must too.
    torch.manual_seed(0)
    m = PrunableCTC(tiny(units="head,interm"))
    for layer in m.encoder.layers:
        layer.self_attn.output_proj.bias.data.normal_()
        layer.ffn.output_proj.bias.data.normal_()
    m.encoder.layers[0].self_attn.hard_concrete_for_heads.log_alpha.data.fill_(-10.0)
    m.encoder.layers[1].ffn.hard_concrete_for_intermediate.log_alpha.data.fill_(-10.0)
    m.eval()
    audio, lens = batch()
    with torch.no_grad():
        ref = m(audio, lens)
    expected_params = m.num_params("deterministic")
    pruned = copy.deepcopy(m)
    new_cfg = pruned.prune()
    assert new_cfg.num_heads[0] == 0 and new_cfg.ffn_dims[1] == 0
    rebuilt = PrunableCTC(new_cfg)
    rebuilt.load_state_dict(pruned.state_dict())
    rebuilt.eval()
    assert rebuilt.num_params("full") == expected_params
    with torch.no_grad():
        out = rebuilt(audio, lens)
    for i, n in enumerate(ref["frame_lens"].tolist()):
        torch.testing.assert_close(out["logits"][i, :n], ref["logits"][i, :n], atol=1e-4, rtol=1e-4)
