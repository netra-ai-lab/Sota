"""Checkpoint files: atomic writes, a free-space guard, and resume.

``latest.pt`` holds the full trainer state (weights, optimizer moments, gates and
Lagrange multipliers, schedule position, data progress, RNG states). It is written
to ``latest.pt.tmp`` and renamed, so a crash mid-write leaves the previous
checkpoint intact (pattern from streaming_asr's ``save_checkpoint``). Before writing,
the free space on the target filesystem is checked against the size of the last
save. If the temp file would not fit, the save is skipped with a warning rather
than filling the disk.
"""

from __future__ import annotations

import ctypes
import gc
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except Exception:
        return None


def atomic_save(payload: dict, path: Path, *, min_free_bytes: int = 0) -> bool:
    """Write ``payload`` to ``path`` via a temp file; return False if skipped for lack of space."""
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < min_free_bytes:
        print(f"WARNING: not saving {path.name}: {free / 2**30:.1f} GB free < {min_free_bytes / 2**30:.1f} GB needed",
              flush=True)
        return False
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.time()
    torch.save(payload, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)
    size = path.stat().st_size
    print(f"saved {path} ({size / 2**30:.2f} GB, {time.time() - t0:.0f} s)", flush=True)
    return True


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def release_host_memory() -> None:
    """Return freed CPU memory to the OS (glibc keeps it otherwise); 30 GB RAM is tight for the 1B."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def export_weights(src: Path, dst: Path) -> None:
    """Strip a trainer ``latest.pt`` down to ``{arch, model, step}`` (the teacher / init for the next stage)."""
    ck = load(src)
    payload = {"arch": ck["arch"], "model": ck["model"], "step": ck.get("step")}
    del ck
    release_host_memory()
    if not atomic_save(payload, dst):
        raise SystemExit(f"not enough free space to write {dst}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "export-weights":
        raise SystemExit("usage: python -m sota_train.ckpt export-weights <latest.pt> <out.pt>")
    export_weights(Path(sys.argv[2]), Path(sys.argv[3]))
