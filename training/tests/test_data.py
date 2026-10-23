"""FLEURS mixing in the training stream: fixed share per language, repeated units never done."""

import json

import numpy as np

from sota_train.data import SAMPLE_RATE, DataConfig, Example, MixedUnitStream


def fake_stream(tmp_path, fleurs_fraction):
    (tmp_path / "ddd.json").write_text(json.dumps({f"ddd/{i}.parquet": list(range(100)) for i in range(40)}))
    (tmp_path / "fk.json").write_text(json.dumps({"fk/0.parquet": list(range(128))}))
    rows = [{"id": f"l{i}", "path": "x", "text": "a"} for i in range(4000)]
    (tmp_path / "libri.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = [{"id": f"g{i}", "path": "x", "text": "a"} for i in range(300)]
    (tmp_path / "fe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "vocab.json").write_text(json.dumps(["<s>", "<pad>", "</s>", "<unk>", "a"]))
    cfg = DataConfig(ddd_units="ddd.json", libri_train="libri.jsonl", vocab="vocab.json",
                     fleurs_km_units="fk.json", fleurs_en_train="fe.jsonl", fleurs_fraction=fleurs_fraction,
                     libri_unit_size=100, buffer_size=384, num_workers=0)
    s = MixedUnitStream(cfg, tmp_path, seed=1, epoch=0, done_units=set())

    def ex(key, lang):  # 2 s per example
        return Example(key, np.zeros(2 * SAMPLE_RATE, np.float32), "a", [4], lang)

    s._read_ddd = lambda unit: iter([ex(f"{unit}:{r}", "km") for r in (s.ddd.get(unit) or s.fleurs_km[unit])])
    s._read_libri = lambda unit: iter([ex(it["id"], "en") for it in (s.libri.get(unit) or s.fleurs_en[unit])])
    return s


def test_fleurs_share_and_done_units(tmp_path):
    s = fake_stream(tmp_path, 0.2)
    keys, done = [], set()
    for b in s:
        keys += b.keys
        done.update(b.done_units)
    km = [k for k in keys if k.startswith(("d:", "f:"))]
    en = [k for k in keys if not k.startswith(("d:", "f:"))]
    assert abs(sum(k.startswith("f:") for k in km) / len(km) - 0.2) < 0.06
    assert abs(sum(k.startswith("g") for k in en) / len(en) - 0.2) < 0.06
    assert done == set(s.ddd) | set(s.libri), "every DDD/Libri unit done, no FLEURS unit"
    assert len([k for k in en if not k.startswith("g")]) == 4000, "each Libri utterance once per epoch"


def test_no_fleurs_when_fraction_zero(tmp_path):
    s = fake_stream(tmp_path, 0.0)
    keys = [k for b in s for k in b.keys]
    assert not any(k.startswith(("f:", "g")) for k in keys)
