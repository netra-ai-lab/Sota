"""Greedy CTC decoding and scoring.

Khmer: CER over NFC codepoints with whitespace removed, the unit of streaming_asr's
2.24% DDD-Test figure. References are first reduced to characters the Omnilingual
vocabulary can emit (it has no Khmer punctuation), so a model is never charged for
a symbol it cannot produce. English: WER over lowercased words, plus CER.
"""

from __future__ import annotations

import time

import editdistance
import torch

from .data import EvalItem, eval_batches
from sota.model import PrunableCTC, normalize_audio_
from sota.text import CharVocab


@torch.no_grad()
def transcribe(model: PrunableCTC, items: list[EvalItem], vocab: CharVocab, *, device: torch.device,
               max_batch_seconds: float = 200.0) -> tuple[list[str], float]:
    """Hypotheses in the order of ``items`` and the wall time spent."""
    was_training = model.training
    model.eval()
    hyps: dict[str, str] = {}
    autocast = next(model.parameters()).dtype == torch.float32
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    t0 = time.time()
    for batch_items, audio, lens in eval_batches(items, max_batch_seconds):
        audio, lens = audio.to(device), lens.to(device)
        normalize_audio_(audio, lens)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            out = model(audio.to(next(model.parameters()).dtype), lens)
        ids = out["logits"].argmax(-1).cpu()
        for i, it in enumerate(batch_items):
            hyps[it.key] = vocab.greedy(ids[i, : int(out["frame_lens"][i])].tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    model.train(was_training)
    return [hyps[it.key] for it in items], elapsed


def score(items: list[EvalItem], hyps: list[str]) -> dict:
    ce = cn = we = wn = 0
    for it, h in zip(items, hyps):
        r_chars, h_chars = list(it.text.replace(" ", "")), list(h.replace(" ", ""))
        ce += editdistance.eval(r_chars, h_chars)
        cn += len(r_chars)
        if it.lang == "en":
            r_w, h_w = it.text.split(), h.split()
            we += editdistance.eval(r_w, h_w)
            wn += len(r_w)
    out = {"cer": ce / max(cn, 1), "n": len(items)}
    if wn:
        out["wer"] = we / wn
    return out


def evaluate(model: PrunableCTC, sets: dict[str, list[EvalItem]], vocab: CharVocab, *, device: torch.device,
             max_batch_seconds: float = 200.0, examples: int = 0) -> dict:
    results = {}
    for name, items in sets.items():
        hyps, secs = transcribe(model, items, vocab, device=device, max_batch_seconds=max_batch_seconds)
        res = score(items, hyps)
        audio_s = sum(it.audio.shape[0] for it in items) / 16000
        res["rtf"] = secs / max(audio_s, 1e-9)
        if examples:
            res["examples"] = [{"ref": it.text, "hyp": h} for it, h in list(zip(items, hyps))[:examples]]
        results[name] = res
    return results
