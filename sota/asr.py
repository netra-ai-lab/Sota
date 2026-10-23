"""The ``SOTA`` speech recogniser: load a model, transcribe audio (greedy CTC, no language model)."""

from __future__ import annotations

from pathlib import Path

import torch

from .audio import AudioInput, load_audio
from .hub import DEFAULT_MODEL, load_weights, resolve
from .model import ArchConfig, PrunableCTC, normalize_audio_
from .text import CharVocab


class SOTA:
    """A Khmer + English CTC model with its vocabulary.

    >>> asr = SOTA.from_pretrained("sota-base")
    >>> asr.transcribe("speech.wav")
    """

    def __init__(self, model: PrunableCTC, vocab: CharVocab, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.vocab = vocab

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path = DEFAULT_MODEL, device: str | torch.device | None = None) -> SOTA:
        arch, state, vocab_path = load_weights(resolve(name_or_path))
        model = PrunableCTC(ArchConfig.from_dict(arch))
        model.load_state_dict(state)
        vocab = CharVocab.load(vocab_path) if vocab_path else CharVocab.default()
        return cls(model, vocab, device)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def transcribe(self, audio: AudioInput | list[AudioInput], batch_size: int = 8) -> str | list[str]:
        """Text for one input, or a list of texts for a list of inputs (same order).

        An input is a file path, a 16 kHz waveform (numpy array or tensor), or a
        ``(waveform, sample_rate)`` tuple. Each file is decoded in one pass, so very long
        recordings should be split into utterances first.
        """
        single = not isinstance(audio, list)
        items = [audio] if single else audio
        wavs = [load_audio(a) for a in items]
        order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]), reverse=True)  # less padding per batch
        texts: list[str] = [""] * len(wavs)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            lens = torch.tensor([len(wavs[i]) for i in idx])
            batch = torch.zeros(len(idx), int(lens.max()))
            for row, i in enumerate(idx):
                batch[row, : len(wavs[i])] = wavs[i]
            normalize_audio_(batch, lens)
            batch, lens = batch.to(self.device), lens.to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.model(batch, lens)
            ids = out["logits"].argmax(-1).cpu()
            for row, i in enumerate(idx):
                texts[i] = self.vocab.greedy(ids[row, : int(out["frame_lens"][row])].tolist())
        return texts[0] if single else texts


def load_model(name_or_path: str | Path = DEFAULT_MODEL, device: str | torch.device | None = None) -> SOTA:
    """``SOTA.from_pretrained``: a short name (``sota-base``, ``sota-large``, ``teacher-300m``,
    ``teacher-1b``), a Hugging Face repo id, a release folder or a training checkpoint."""
    return SOTA.from_pretrained(name_or_path, device)


def transcribe(audio: AudioInput | list[AudioInput], model: str | Path = DEFAULT_MODEL,
               device: str | torch.device | None = None, batch_size: int = 8) -> str | list[str]:
    """One-shot helper: load ``model`` and transcribe ``audio``. Reuse ``load_model`` for many calls."""
    return load_model(model, device).transcribe(audio, batch_size=batch_size)
