"""Training and evaluation data: DDD (Khmer) + LibriSpeech (English), resumable.

Training data is cut into *units*: a DDD parquet shard (~0.24 h, ~100 utterances) or a
fixed chunk of LibriSpeech utterances of similar size. Every epoch has a deterministic
unit order (from the seed and the epoch number). DataLoader workers take every W-th
unit of each source and interleave the two sources so that English makes up
``en_fraction`` of the audio seconds. Examples go through a length-sorting buffer
and come out as duration-budgeted batches (ported from streaming_asr's
``DynamicBatcher``).

Resumption: a unit counts as *done* once all of its examples have left the
buffer inside a yielded batch. The ids of units finished since the previous
batch ride along on each batch (``Batch.done_units``). The trainer keeps that set
in its checkpoint and, on resume, rebuilds the same epoch order minus the finished
units. Delivery is at-least-once: a crash can repeat the tail of a few units, but no
unit is skipped.

Optional FLEURS sources (``fleurs_km_units`` / ``fleurs_en_train``) are mixed in at a fixed
share of their language's audio (``fleurs_fraction``). They are far smaller than DDD and
LibriSpeech, so each worker cycles through them in a reshuffled order for as long as the
epoch lasts; their units are never marked done, and the epoch still ends with DDD/LibriSpeech.
"""

from __future__ import annotations

import io
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from sota.text import CharVocab, normalize_text

from . import ASR_ROOT

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 320


@dataclass
class Example:
    key: str
    audio: np.ndarray
    text: str  # normalised, vocabulary-filtered
    target: list[int]
    lang: str


@dataclass
class Batch:
    keys: list[str]
    audio: torch.Tensor  # (N, S) float32, zero-padded, not yet normalised
    audio_lens: torch.Tensor  # (N,)
    targets: torch.Tensor  # (N, L), padded with 1 (<pad>)
    target_lens: torch.Tensor  # (N,)
    texts: list[str]
    langs: list[str]
    done_units: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def seconds(self) -> float:
        return float(self.audio_lens.sum()) / SAMPLE_RATE


def collate(examples: list[Example]) -> Batch:
    n_s = max(e.audio.shape[0] for e in examples)
    n_t = max(len(e.target) for e in examples)
    audio = torch.zeros(len(examples), n_s)
    targets = torch.full((len(examples), n_t), 1, dtype=torch.long)
    audio_lens = torch.zeros(len(examples), dtype=torch.long)
    target_lens = torch.zeros(len(examples), dtype=torch.long)
    for i, e in enumerate(examples):
        audio[i, : e.audio.shape[0]] = torch.from_numpy(e.audio)
        audio_lens[i] = e.audio.shape[0]
        targets[i, : len(e.target)] = torch.tensor(e.target)
        target_lens[i] = len(e.target)
    return Batch(
        keys=[e.key for e in examples], audio=audio, audio_lens=audio_lens, targets=targets,
        target_lens=target_lens, texts=[e.text for e in examples], langs=[e.lang for e in examples],
    )


def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:  # a few DDD rows are 48 kHz
        import torchaudio.functional as AF

        audio = AF.resample(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), sr, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_wav_bytes(raw: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return _to_mono_16k(audio, sr)


def read_audio_file(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return _to_mono_16k(audio, sr)


def ctc_usable(n_samples: int, n_target: int) -> bool:
    """CTC needs at least one frame per token (plus slack for repeats)."""
    return n_target > 0 and n_samples // SAMPLES_PER_FRAME >= n_target + 1


# --------------------------------------------------------------------------- training stream


@dataclass
class DataConfig:
    ddd_units: str = "data/ddd_units.json"
    libri_train: str = "data/libri_train.jsonl"
    vocab: str | None = None  # a vocab.json; None = the one shipped with sota
    en_fraction: float = 0.5  # share of audio seconds from LibriSpeech
    fleurs_km_units: str | None = None  # {shard: [rows]} like ddd_units, repeated as needed
    fleurs_en_train: str | None = None  # JSONL like libri_train, repeated as needed
    fleurs_fraction: float = 0.0  # share of each language's audio seconds taken from FLEURS
    libri_unit_size: int = 100  # utterances per LibriSpeech unit (~0.35 h)
    min_duration: float = 1.0
    max_duration: float = 25.0
    max_batch_seconds: float = 160.0
    max_batch_size: int = 32
    buffer_size: int = 384
    num_workers: int = 6
    asr_root: str = str(ASR_ROOT)


class MixedUnitStream(IterableDataset):
    def __init__(self, cfg: DataConfig, root: Path, *, seed: int, epoch: int, done_units: set[str]) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed, self.epoch = seed, epoch
        self.done = set(done_units)
        self.vocab = CharVocab.load(root / cfg.vocab) if cfg.vocab else CharVocab.default()
        self.asr_root = Path(cfg.asr_root)
        ddd = json.loads((root / cfg.ddd_units).read_text())
        self.ddd = {f"d:{shard}": rows for shard, rows in ddd.items()}
        self.libri: dict[str, list[dict]] = {}
        if cfg.en_fraction > 0:
            items = [json.loads(l) for l in open(root / cfg.libri_train, encoding="utf-8")]
            random.Random(self.seed).shuffle(items)  # fixed unit membership across epochs
            for k in range(0, len(items), cfg.libri_unit_size):
                self.libri[f"l:{k // cfg.libri_unit_size}"] = items[k : k + cfg.libri_unit_size]
        # repeated FLEURS pools: "f:" Khmer shards (read like DDD), "g:" English chunks (read like Libri)
        self.fleurs_km: dict[str, list[int]] = {}
        self.fleurs_en: dict[str, list[dict]] = {}
        if cfg.fleurs_fraction > 0 and cfg.fleurs_km_units:
            units = json.loads((root / cfg.fleurs_km_units).read_text())
            for shard, rows in units.items():  # FLEURS shards hold ~1 h; split to DDD-unit size
                for k in range(0, len(rows), 64):
                    self.fleurs_km[f"f:{shard}#{k // 64}"] = rows[k : k + 64]
        if cfg.fleurs_fraction > 0 and cfg.fleurs_en_train and cfg.en_fraction > 0:
            items = [json.loads(l) for l in open(root / cfg.fleurs_en_train, encoding="utf-8")]
            random.Random(self.seed).shuffle(items)
            for k in range(0, len(items), cfg.libri_unit_size):
                self.fleurs_en[f"g:{k // cfg.libri_unit_size}"] = items[k : k + cfg.libri_unit_size]

    def unit_order(self) -> tuple[list[str], list[str]]:
        rng = random.Random(self.seed * 100_003 + self.epoch)
        d, l = sorted(self.ddd), sorted(self.libri)
        rng.shuffle(d)
        rng.shuffle(l)
        return [u for u in d if u not in self.done], [u for u in l if u not in self.done]

    def remaining_units(self) -> tuple[int, int]:
        d, l = self.unit_order()
        return len(d), len(l)

    # -- reading

    def _read_ddd(self, unit: str) -> Iterator[Example]:
        shard = unit[2:].split("#")[0]
        allowed = set(self.ddd[unit] if unit in self.ddd else self.fleurs_km[unit])
        try:
            table = pq.read_table(self.asr_root / shard, columns=["audio", "transcript"])
        except Exception:
            return
        audios = table.column("audio").to_pylist()
        texts = table.column("transcript").to_pylist()
        for row, (a, t) in enumerate(zip(audios, texts)):
            if row not in allowed or not a or not a.get("bytes"):
                continue
            ex = self._make(f"{shard}:{row}", lambda a=a: decode_wav_bytes(a["bytes"]), t or "", "km")
            if ex is not None:
                yield ex

    def _read_libri(self, unit: str) -> Iterator[Example]:
        for it in self.libri[unit] if unit in self.libri else self.fleurs_en[unit]:
            ex = self._make(it["id"], lambda p=it["path"]: read_audio_file(p), it["text"], "en")
            if ex is not None:
                yield ex

    def _make(self, key: str, load, text: str, lang: str) -> Example | None:
        text = self.vocab.filter(normalize_text(text, lang))
        target, _ = self.vocab.encode(text)
        if not target:
            return None
        try:
            audio = load()
        except Exception:
            return None
        dur = audio.shape[0] / SAMPLE_RATE
        if not (self.cfg.min_duration <= dur <= self.cfg.max_duration):
            return None
        if not ctc_usable(audio.shape[0], len(target)):
            return None
        return Example(key, audio, text, target, lang)

    # -- batching

    def _flush(self, buffer: list[Example], rng: random.Random) -> list[list[Example]]:
        buffer.sort(key=lambda e: e.audio.shape[0])
        batches, cur, longest = [], [], 0
        for ex in buffer:
            n = max(longest, ex.audio.shape[0])
            if cur and (n * (len(cur) + 1) / SAMPLE_RATE > self.cfg.max_batch_seconds
                        or len(cur) >= self.cfg.max_batch_size):
                batches.append(cur)
                cur, n = [], ex.audio.shape[0]
            cur.append(ex)
            longest = n
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        d_units, l_units = self.unit_order()
        d_units, l_units = d_units[wid::nw], l_units[wid::nw]
        rng = random.Random(self.seed * 7919 + self.epoch * 31 + wid)
        sec = {"km": 0.0, "en": 0.0}
        fleurs_sec = {"km": 0.0, "en": 0.0}
        cycles = {"km": self._cycle(sorted(self.fleurs_km), rng), "en": self._cycle(sorted(self.fleurs_en), rng)}
        buffer: list[Example] = []
        pending: list[str] = []  # units fully read into the buffer, not yet flushed
        di = li = 0
        while di < len(d_units) or li < len(l_units):
            total = sec["km"] + sec["en"]
            want_en = li < len(l_units) and (
                di >= len(d_units) or sec["en"] < self.cfg.en_fraction * max(total, 1e-9)
            )
            lang = "en" if want_en else "km"
            fleurs = cycles[lang] is not None and fleurs_sec[lang] < self.cfg.fleurs_fraction * max(sec[lang], 1e-9)
            if fleurs:
                unit = next(cycles[lang])
            elif want_en:
                unit, li = l_units[li], li + 1
            else:
                unit, di = d_units[di], di + 1
            reader = self._read_libri(unit) if want_en else self._read_ddd(unit)
            for ex in reader:
                dur = ex.audio.shape[0] / SAMPLE_RATE
                sec[ex.lang] += dur
                if fleurs:
                    fleurs_sec[ex.lang] += dur
                buffer.append(ex)
            if not fleurs:  # repeated FLEURS units are never done
                pending.append(unit)
            if len(buffer) >= self.cfg.buffer_size:
                yield from self._emit(buffer, pending, rng)
                buffer, pending = [], []
        if buffer or pending:
            yield from self._emit(buffer, pending, rng)

    @staticmethod
    def _cycle(units: list[str], rng: random.Random) -> Iterator[str] | None:
        if not units:
            return None

        def gen() -> Iterator[str]:
            while True:
                order = list(units)
                rng.shuffle(order)
                yield from order

        return gen()

    def _emit(self, buffer: list[Example], pending: list[str], rng: random.Random) -> Iterator[Batch]:
        batches = self._flush(buffer, rng) if buffer else []
        if not batches:
            # units that yielded nothing still need to be marked done; an empty batch carries them
            yield Batch([], torch.zeros(0, 0), torch.zeros(0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long),
                        torch.zeros(0, dtype=torch.long), [], [], list(pending))
            return
        for i, b in enumerate(batches):
            batch = collate(b)
            if i == len(batches) - 1:
                batch.done_units = list(pending)
            yield batch


def make_train_loader(cfg: DataConfig, root: Path, *, seed: int, epoch: int, done_units: set[str]) -> DataLoader:
    ds = MixedUnitStream(cfg, root, seed=seed, epoch=epoch, done_units=done_units)
    return DataLoader(
        ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=True,
        prefetch_factor=2 if cfg.num_workers > 0 else None, persistent_workers=False,
        # spawn, not fork: a forked worker keeps the parent's pages copy-on-write, and the trainer
        # rewrites ~11.6 GB of offloaded optimizer state every step, so each fork pinned a stale copy
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )


# --------------------------------------------------------------------------- evaluation sets


@dataclass
class EvalItem:
    key: str
    audio: np.ndarray
    text: str  # normalised + vocabulary-filtered reference
    lang: str


def load_eval_set(path: Path, vocab: CharVocab, asr_root: Path, *, limit: int | None = None,
                  cache_dir: Path | None = None) -> list[EvalItem]:
    """Read a JSONL eval manifest: rows with either ``path`` (audio file) or ``shard``+``row``.

    The decoded set is cached as one ``.pt`` file next to the manifest's name, since
    pulling a few hundred rows out of parquet shards takes far longer than reading
    them back.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if limit is not None:
        rows = rows[:limit]
    cache = None
    if cache_dir is not None:
        cache = cache_dir / f"{path.stem}.{len(rows)}.pt"
        if cache.exists():
            data = torch.load(cache, weights_only=False)
            return [EvalItem(**d) for d in data]
    items: list[EvalItem] = []
    by_shard: dict[str, list[dict]] = {}
    for r in rows:
        if "path" in r:
            text = vocab.filter(normalize_text(r["text"], r["lang"]))
            items.append(EvalItem(r["id"], read_audio_file(r["path"]), text, r["lang"]))
        else:
            by_shard.setdefault(r["shard"], []).append(r)
    for shard, rs in by_shard.items():
        table = pq.read_table(asr_root / shard, columns=["audio"])
        audios = table.column("audio")
        for r in rs:
            a = audios[r["row"]].as_py()
            text = vocab.filter(normalize_text(r["text"], r["lang"]))
            items.append(EvalItem(f"{shard}:{r['row']}", decode_wav_bytes(a["bytes"]), text, r["lang"]))
    order = {(r.get("id") or f"{r['shard']}:{r['row']}"): i for i, r in enumerate(rows)}
    items.sort(key=lambda it: order[it.key])
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save([vars(it) for it in items], cache)
    return items


def eval_batches(items: list[EvalItem], max_batch_seconds: float = 200.0) -> Iterator[tuple[list[EvalItem], torch.Tensor, torch.Tensor]]:
    order = sorted(range(len(items)), key=lambda i: items[i].audio.shape[0])
    cur: list[EvalItem] = []
    for i in order:
        it = items[i]
        n = max([it.audio.shape[0]] + [c.audio.shape[0] for c in cur])
        if cur and n * (len(cur) + 1) / SAMPLE_RATE > max_batch_seconds:
            yield _pad(cur)
            cur = []
        cur.append(it)
    if cur:
        yield _pad(cur)


def _pad(items: list[EvalItem]) -> tuple[list[EvalItem], torch.Tensor, torch.Tensor]:
    n = max(it.audio.shape[0] for it in items)
    audio = torch.zeros(len(items), n)
    lens = torch.zeros(len(items), dtype=torch.long)
    for i, it in enumerate(items):
        audio[i, : it.audio.shape[0]] = torch.from_numpy(it.audio)
        lens[i] = it.audio.shape[0]
    return items, audio, lens
