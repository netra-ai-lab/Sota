"""Build the training-unit index and evaluation manifests.

    data/ddd_units.json        {shard: [row indices]}: DDD train rows, minus DDD-Test,
                               streaming_asr's exclude list and the dev shards
    data/ddd_dev.jsonl         <= 400 utterances from held-out DDD shards (checkpoint selection)
    data/ddd_test.jsonl        DDD-Test (streaming_asr manifests/val_indomain.jsonl), report only
    data/fleurs_km_test.jsonl  FLEURS km test (streaming_asr manifests/test_fleurs_test.jsonl)
    data/libri_train.jsonl     train-clean-100 + train-clean-360
    data/libri_{dev-clean,test-clean,test-other}.jsonl

  --only fleurs (the fine-tune-first mix; downloads FLEURS en_us from the Hub if missing):
    data/fleurs_en_{train,dev,test}.jsonl  FLEURS en_us, normalized transcription
    data/fleurs_km_dev.jsonl   FLEURS km validation (streaming_asr manifests/dev_fleurs.jsonl)
    data/fleurs_km_units.json  FLEURS km train rows by shard (mixed in at a fixed share, see data.py)

    .venv/bin/python scripts/01_prepare_data.py [--only ddd|libri|libri-eval|fleurs]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sota_train import ASR_ROOT, FLEURS_EN_DIR, LIBRI_ROOT, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "data"
FLEURS_EN_ROOT = FLEURS_EN_DIR
csv.field_size_limit(1 << 24)


def manifest_keys(path: Path) -> list[dict]:
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        rows.append({"shard": r["shard_path"], "row": int(r["row_index"]), "text": r["transcript"], "lang": "km"})
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path.relative_to(PROJECT_ROOT)}: {len(rows)} rows")


def prepare_ddd(args) -> None:
    test = manifest_keys(ASR_ROOT / "manifests/val_indomain.jsonl")
    exclude = set(json.load(open(ASR_ROOT / "manifests/train_combined.json"))["exclude_keys"])
    exclude |= {f"{r['shard']}:{r['row']}" for r in test}
    shards: dict[str, list[dict]] = defaultdict(list)
    with open(ASR_ROOT / "data/dataset/train.tsv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            shards[r["shard_path"]].append(r)
    names = sorted(shards)
    rng = random.Random(args.seed)
    dev_shards = set(rng.sample(names, args.dev_shards))
    units, dev, hours = {}, [], 0.0
    for shard in names:
        keep = []
        for r in shards[shard]:
            key = f"{shard}:{r['row_index']}"
            dur = float(r["duration"] or 0)
            if key in exclude or not r["transcript"].strip() or not (args.min_dur <= dur <= args.max_dur):
                continue
            if shard in dev_shards:
                dev.append({"shard": shard, "row": int(r["row_index"]), "text": r["transcript"], "lang": "km"})
            else:
                keep.append(int(r["row_index"]))
                hours += dur / 3600
        if keep:
            units[shard] = keep
    rng.shuffle(dev)
    OUT.mkdir(exist_ok=True)
    (OUT / "ddd_units.json").write_text(json.dumps(units))
    print(f"  data/ddd_units.json: {len(units)} shards, {sum(map(len, units.values()))} rows, {hours:.1f} h")
    write_jsonl(OUT / "ddd_dev.jsonl", dev[: args.dev_utts])
    write_jsonl(OUT / "ddd_test.jsonl", test)
    write_jsonl(OUT / "fleurs_km_test.jsonl", manifest_keys(ASR_ROOT / "manifests/test_fleurs_test.jsonl"))


def libri_rows(subset: str) -> list[dict]:
    root = LIBRI_ROOT / subset
    if not root.exists():
        raise SystemExit(f"{root} missing: run scripts/00_get_assets.sh first")
    rows = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in open(trans, encoding="utf-8"):
            uid, text = line.rstrip("\n").split(" ", 1)
            rows.append({"id": uid, "path": str(trans.parent / f"{uid}.flac"), "text": text,
                         "lang": "en", "spk": uid.split("-")[0]})
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(lambda r: sf.info(r["path"]).duration, rows))
    for r, d in zip(rows, durs):
        r["duration"] = round(d, 3)
    return rows


def prepare_libri_eval(args) -> None:
    for s in ("dev-clean", "test-clean", "test-other"):
        write_jsonl(OUT / f"libri_{s}.jsonl", libri_rows(s))


def prepare_libri(args) -> None:
    train = []
    for s in ("train-clean-100", "train-clean-360"):
        rows = libri_rows(s)
        print(f"  {s}: {len(rows)} utts, {sum(r['duration'] for r in rows) / 3600:.1f} h")
        train += rows
    write_jsonl(OUT / "libri_train.jsonl", train)
    prepare_libri_eval(args)


def fleurs_en_rows(split: str) -> list[dict]:
    """Write one FLEURS en_us split as 16 kHz WAVs (the Hub parquet already holds WAV bytes)."""
    from huggingface_hub import hf_hub_download

    src = hf_hub_download("google/fleurs", f"parquet-data/en_us/{split}-00000-of-00001.parquet",
                          repo_type="dataset", local_dir=FLEURS_EN_ROOT / "raw")
    table = pq.read_table(src, columns=["id", "num_samples", "audio", "transcription", "gender"])
    out = FLEURS_EN_ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in table.to_pylist():
        wav = out / r["audio"]["path"]
        if not wav.exists():
            wav.write_bytes(r["audio"]["bytes"])
        rows.append({"id": f"fleurs_en_{split}_{wav.stem}", "path": str(wav), "text": r["transcription"],
                     "lang": "en", "spk": f"fleurs_en_{r['id']}", "duration": round(r["num_samples"] / 16000, 3)})
    print(f"  fleurs en_us {split}: {len(rows)} utts, {sum(r['duration'] for r in rows) / 3600:.1f} h")
    return rows


def prepare_fleurs(args) -> None:
    train = fleurs_en_rows("train")
    write_jsonl(OUT / "fleurs_en_train.jsonl", train)
    write_jsonl(OUT / "fleurs_en_dev.jsonl", fleurs_en_rows("validation"))
    write_jsonl(OUT / "fleurs_en_test.jsonl", fleurs_en_rows("test"))
    write_jsonl(OUT / "fleurs_km_dev.jsonl", manifest_keys(ASR_ROOT / "manifests/dev_fleurs.jsonl"))

    units, n, hours = {}, 0, 0.0
    for shard in sorted((ASR_ROOT / "data/fleurs_train").glob("*.parquet")):
        t = pq.read_table(shard, columns=["transcript", "duration"]).to_pylist()
        keep = [i for i, r in enumerate(t) if (r["transcript"] or "").strip()
                and args.min_dur <= float(r["duration"] or 0) <= args.max_dur]
        units[str(shard.relative_to(ASR_ROOT))] = keep
        n += len(keep)
        hours += sum(float(t[i]["duration"]) for i in keep) / 3600
    (OUT / "fleurs_km_units.json").write_text(json.dumps(units))
    print(f"  data/fleurs_km_units.json: {n} FLEURS km train rows ({hours:.1f} h)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ddd", "libri", "libri-eval", "fleurs"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dev-shards", type=int, default=12)
    ap.add_argument("--dev-utts", type=int, default=400)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=25.0)
    args = ap.parse_args()
    if args.only in (None, "ddd"):
        prepare_ddd(args)
    if args.only in (None, "libri"):
        prepare_libri(args)
    if args.only == "libri-eval":
        prepare_libri_eval(args)
    if args.only == "fleurs":
        prepare_fleurs(args)


if __name__ == "__main__":
    main()
