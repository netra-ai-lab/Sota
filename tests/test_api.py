import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import sota
from sota.audio import load_audio
from sota.model import ArchConfig, PrunableCTC
from sota.text import CharVocab


def tiny_release(tmp_path, vocab_size=None):
    vocab = CharVocab.default()
    cfg = ArchConfig(
        conv_layers=[(16, 10, 5), (16, 3, 2), (12, 2, 2)],
        num_heads=[4, 2, 3], ffn_dims=[64, 16, 40], model_dim=32, head_dim=8,  # non-uniform, like a pruned model
        pos_conv_kernel=4, pos_conv_groups=2, vocab_size=vocab_size or len(vocab),
    )
    torch.manual_seed(0)
    model = PrunableCTC(cfg).eval()
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    return model


def test_release_folder_round_trip(tmp_path):
    ref = tiny_release(tmp_path)
    asr = sota.load_model(tmp_path, device="cpu")
    assert asr.num_parameters == sum(p.numel() for p in ref.parameters())
    x, lens = torch.randn(1, 8000), torch.tensor([8000])
    with torch.no_grad():
        assert torch.equal(asr.model(x, lens)["logits"], ref(x, lens)["logits"])
    assert len(asr.vocab) == 9812  # packaged vocabulary when the folder has none


def test_batched_equals_one_by_one(tmp_path):
    tiny_release(tmp_path)
    asr = sota.load_model(tmp_path, device="cpu")
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16000, 5000, 11000)]
    batched = asr.transcribe(clips, batch_size=3)
    assert batched == [asr.transcribe(c) for c in clips]
    assert isinstance(asr.transcribe(clips[0]), str)


def test_load_audio_files_and_rates(tmp_path):
    sf = pytest.importorskip("soundfile")
    stereo = np.stack([np.ones(8000), np.zeros(8000)], axis=1).astype(np.float32)
    sf.write(tmp_path / "a.wav", stereo, 8000)
    wav = load_audio(tmp_path / "a.wav")
    assert wav.dtype == torch.float32 and wav.ndim == 1 and abs(len(wav) - 16000) <= 1
    assert torch.allclose(wav[100:-100].mean(), torch.tensor(0.5), atol=1e-3)
    assert len(load_audio((np.zeros(44100, dtype=np.float32), 44100))) == 16000


def test_unknown_short_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        sota.load_model("not-a-model")


def test_import_stays_light():
    import subprocess
    import sys

    code = "import sys, sota; print(','.join(m for m in ('fairseq2', 'yaml', 'pyarrow', 'sota_train') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""
