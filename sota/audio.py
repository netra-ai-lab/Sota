"""Audio input: any file soundfile can read, or an array, as a mono 16 kHz float32 tensor."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000

#: what ``SOTA.transcribe`` accepts per utterance: a path, a 16 kHz waveform, or (waveform, sample_rate)
AudioInput = Union[str, Path, np.ndarray, torch.Tensor, tuple]


def resample(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return wav
    import soxr

    return soxr.resample(wav, sr, SAMPLE_RATE).astype(np.float32, copy=False)


def load_audio(audio: AudioInput) -> torch.Tensor:
    """Decode / convert one input to a 1-D float32 tensor at 16 kHz (channels averaged)."""
    if isinstance(audio, tuple):
        wav, sr = audio
    elif isinstance(audio, (str, Path)):
        wav, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    else:
        wav, sr = audio, SAMPLE_RATE
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:  # (channels, samples) or (samples, channels): average the shorter axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    if wav.ndim != 1 or wav.size == 0:
        raise ValueError(f"expected a non-empty mono or stereo waveform, got shape {wav.shape}")
    return torch.from_numpy(np.ascontiguousarray(resample(wav, int(sr))))
