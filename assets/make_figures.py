"""Draw the README figures from the training outputs (Netra chart style).

    python assets/make_figures.py      # reads training/runs/eval/*.json and */stage2_prune_s80/pruned.json

Writes assets/results.png (test error, two model sizes) and assets/layers.png (what pruning kept).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "training" / "runs"
sys.path.insert(0, str(HERE / "brand"))
from brand_colors import BASELINES, CHART, COLOR, OURS, apply, legend_dot, ours_bar  # noqa: E402

TESTS = [("ddd_test", "DDD-Test\n(km, CER)"), ("fleurs_km_test", "FLEURS-km\n(CER)"), ("fleurs_en_test", "FLEURS-en\n(WER)"),
         ("libri_test_clean", "test-clean\n(WER)"), ("libri_test_other", "test-other\n(WER)")]


def scores(*names: str) -> dict[str, float]:
    out = {}
    for name in names:
        path = RUNS / "eval" / f"{name}.json"
        if path.exists():
            for k, v in json.loads(path.read_text())["results"].items():
                out.setdefault(k, 100 * v.get("wer", v["cer"]))
    return out


def results() -> None:
    groups = [
        ("Large: from omniASR CTC-1B", [("Stock 1B · 975M", scores("stock_1b", "stock_1b_fleurs"), BASELINES[3]),
                                         ("Fine-tuned 1B · 975M", scores("large_teacher"), BASELINES[0]),
                                         ("sota-large · 204M", scores("large"), None)]),
        ("Base: from omniASR CTC-300M", [("Stock 300M · 325M", scores("stock_300m"), BASELINES[3]),
                                          ("Fine-tuned 300M · 325M", scores("base_teacher"), BASELINES[0]),
                                          ("sota-base · 73M", scores("base"), None)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(TESTS)) * 1.8
    bar_w, pitch = 0.3, 0.42
    for ax, (title, series) in zip(axes, groups):
        for j, (label, s, color) in enumerate(series):
            vals = [s.get(k, np.nan) for k, _ in TESTS]
            pos = x + (j - 1) * pitch
            kw = ours_bar() if color is None else {"color": color, "zorder": 3}
            ax.bar(pos, vals, bar_w, **kw)
            for p, v in zip(pos, vals):
                ax.text(p, (0 if np.isnan(v) else v) + 0.5, "n/a" if np.isnan(v) else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5, color=COLOR["foreground"])
        ax.set_xticks(x, [t for _, t in TESTS], fontsize=9.5)
        ax.set_title(title, loc="left", fontsize=12)
        ax.legend(handles=[legend_dot(OURS if c is None else c, label) for label, _, c in series],
                  loc="upper left", ncol=1, fontsize=9)
    axes[0].set_ylabel("Error rate (%), lower is better")
    fig.savefig(HERE / "results.png")
    plt.close(fig)


def layers() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 5.6), gridspec_kw={"height_ratios": [2.2, 1]}, sharex="col")
    for col, (size, distilled, full_ffn, full_heads) in enumerate([("large", {12, 24, 36, 48}, 5120, 16),
                                                                     ("base", {6, 12, 18, 24}, 4096, 16)]):
        p = json.loads((RUNS / size / "stage2_prune_s80" / "pruned.json").read_text())
        n = len(p["ffn_dims"])
        idx = np.arange(1, n + 1)
        colors = [OURS if i in distilled else BASELINES[3] for i in idx]
        ffn_ax, head_ax = axes[0, col], axes[1, col]
        ffn_ax.bar(idx, p["ffn_dims"], 0.75, color=colors, zorder=3)
        ffn_ax.set_title(f"sota-{size}: {p['encoder_params_after'] / 1e6:.0f}M of {p['encoder_params_before'] / 1e6:.0f}M "
                         f"encoder parameters kept ({100 * p['sparsity']:.0f}% pruned)", loc="left", fontsize=11)
        ffn_ax.set_ylabel(f"FFN units (of {full_ffn:,})")
        head_ax.bar(idx, p["num_heads"], 0.75, color=colors, zorder=3)
        head_ax.set_ylabel(f"Heads (of {full_heads})")
        head_ax.set_xlabel("Encoder layer")
        head_ax.set_xticks([i for i in idx if i == 1 or i % (12 if n > 24 else 6) == 0])
    axes[0, 0].legend(handles=[legend_dot(OURS, "distilled to the teacher"), legend_dot(BASELINES[3], "other layers")],
                      loc="upper center", fontsize=9)
    fig.savefig(HERE / "layers.png")
    plt.close(fig)


if __name__ == "__main__":
    apply()
    plt.rcParams["hatch.color"] = CHART["figureBackground"]
    plt.rcParams["font.sans-serif"] = ["Kantumruy Pro", "DejaVu Sans"]  # Latin-only labels; Noto Sans Khmer has no Latin
    results()
    layers()
    print("wrote", HERE / "results.png", HERE / "layers.png")
