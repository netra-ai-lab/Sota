"""Draw the README figures from the training outputs (Netra chart style).

    python assets/make_figures.py      # reads training/runs/eval/*.json and */stage2_prune_s80/pruned.json

Writes assets/pruning.png (what is pruned; sparsity, speed and accuracy), assets/results.png (test error,
two model sizes) and assets/layers.png (what pruning kept in each layer).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "training" / "runs"
sys.path.insert(0, str(HERE / "brand"))
from brand_colors import BASELINES, CHART, COLOR, OURS, apply, legend_dot, ours_bar  # noqa: E402

TESTS = [("ddd_test", "DDD\n(Khmer, CER)"), ("fleurs_km_test", "FLEURS-km\n(Khmer, CER)"),
         ("libri_test_clean", "LibriSpeech\n(English, WER)")]


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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    x = np.arange(len(TESTS)) * 1.6
    bar_w, pitch = 0.3, 0.42
    for ax, (title, series) in zip(axes, groups):
        for j, (label, s, color) in enumerate(series):
            vals = [s.get(k, np.nan) for k, _ in TESTS]
            pos = x + (j - 1) * pitch
            kw = ours_bar() if color is None else {"color": color, "zorder": 3}
            ax.bar(pos, vals, bar_w, **kw)
            for p, v in zip(pos, vals):
                ax.text(p, (0 if np.isnan(v) else v) + 0.5, "n/a" if np.isnan(v) else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=COLOR["foreground"])
        ax.set_xticks(x, [t for _, t in TESTS], fontsize=10)
        ax.set_title(title, loc="left", fontsize=12)
        ax.legend(handles=[legend_dot(OURS if c is None else c, label) for label, _, c in series],
                  loc="upper right", ncol=1, fontsize=9)
    axes[0].set_ylabel("Error rate (%), lower is better")
    axes[0].set_ylim(0, 22)  # headroom for the legends
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


def _eval(name: str) -> dict:
    return json.loads((RUNS / "eval" / f"{name}.json").read_text())


def _mean_rtf(name: str) -> float:
    r = _eval(name)["results"]
    return float(np.mean([r[k]["rtf"] for k in ("ddd_test", "fleurs_km_test", "libri_test_clean", "libri_test_other")]))


def pruning() -> None:
    """Figure 1: (a) what is pruned inside an encoder layer, (b) what it buys and costs."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    ink, muted, meta, card, border = COLOR["foreground"], COLOR["muted"], COLOR["muted-2"], COLOR["card"], COLOR["border"]
    white = CHART["figureBackground"]
    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, text="", fc=card, ec=border, fs=10.5, r=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1.2))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=ink)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=ink, lw=1.2))

    def pruned_cell(x, y, w, h):
        ax.add_patch(Rectangle((x, y), w, h, fc=white, ec=meta, lw=0.8, ls=(0, (2, 1.5))))
        ax.plot([x + 0.25 * w, x + 0.75 * w], [y + 0.25 * h, y + 0.75 * h], color=meta, lw=0.8)
        ax.plot([x + 0.25 * w, x + 0.75 * w], [y + 0.75 * h, y + 0.25 * h], color=meta, lw=0.8)

    def unit_row(x0, y0, n, kept, w, h, gap):
        keep = set(np.random.default_rng(3).choice(n, size=kept, replace=False).tolist())
        for i in range(n):
            x = x0 + i * (w + gap)
            if i in keep:
                ax.add_patch(Rectangle((x, y0), w, h, fc=OURS, ec="none"))
            else:
                pruned_cell(x, y0, w, h)
            ax.add_patch(Circle((x + w / 2, y0 + h + 1.6), min(0.55, 0.42 * (w + gap)), fc=OURS if i in keep else white, ec=ink, lw=0.7))
        return x0 + n * (w + gap) - gap

    # the encoder at a glance
    ax.text(1, 65.5, "(a) What is pruned", fontsize=13, color=ink, weight="bold", va="top")
    box(2, 5, 22, 7, "CNN feature extractor\nkept, frozen", fs=9.5)
    for k in (2, 1):
        box(2 + k * 0.8, 19.4 + k * 0.8, 22, 22, fc=white)
    box(2, 19.4, 22, 22, "Transformer layer\n× 48 (large)\n× 24 (base)\n\nheads and FFN\nunits gated,\nthen removed", fc=white, ec=ink, fs=9.5)
    box(2, 50, 22, 7, "CTC head, 9,812 chars\nkept", fs=9.5)
    arrow(13, 12, 13, 19.2)
    arrow(13, 43.2, 13, 50)
    ax.plot([24.8, 31], [43, 55.5], color=meta, lw=0.9, ls=":")
    ax.plot([24, 31], [19.4, 1.5], color=meta, lw=0.9, ls=":")

    # one layer, with the real counts of sota-large layer 12
    p = json.loads((RUNS / "large" / "stage2_prune_s80" / "pruned.json").read_text())
    layer = 12
    heads_kept, ffn_kept = p["num_heads"][layer - 1], p["ffn_dims"][layer - 1]
    box(31, 1.5, 68, 54, fc=white, ec=ink, r=1.6)
    ax.text(33, 53.2, f"One encoder layer (sota-large, layer {layer})", fontsize=10.5, color=muted, va="top")

    box(34, 33.5, 62, 15)
    ax.text(35.5, 46.6, "Multi-head self-attention", fontsize=10.5, color=ink, va="top", weight="bold")
    end = unit_row(36, 36, 16, heads_kept, 2.1, 4.2, 0.7)
    ax.text(end + 1.2, 38.1, f"{heads_kept} of 16\nheads kept", fontsize=9.5, color=ink, va="center")

    box(34, 9, 62, 20)
    ax.text(35.5, 27.1, "Feed-forward network", fontsize=10.5, color=ink, va="top", weight="bold")
    ax.text(35.5, 24.6, "1280 → 5,120 units → 1280;  one cell = 160 units", fontsize=8.5, color=muted, va="top")
    cells = 32
    end = unit_row(36, 12.5, cells, round(cells * ffn_kept / 5120), 1.05, 4.6, 0.3)
    ax.text(end + 1.2, 14.8, f"{ffn_kept:,} of\n5,120 units\nkept", fontsize=9.5, color=ink, va="center")
    arrow(64.5, 29.2, 64.5, 33.3)
    ax.text(65.3, 31.2, "LayerNorm + residual", fontsize=8, color=meta, va="center")

    ly = 7.0
    ax.add_patch(Rectangle((35, ly - 1), 2, 2.4, fc=OURS, ec="none"))
    ax.text(37.8, ly + 0.2, "kept", fontsize=9, va="center", color=ink)
    pruned_cell(43, ly - 1, 2, 2.4)
    ax.text(45.8, ly + 0.2, "pruned (removed at the cut)", fontsize=9, va="center", color=ink)
    ax.add_patch(Circle((36, ly - 3.3), 0.55, fc=white, ec=ink, lw=0.7))
    ax.text(37.8, ly - 3.3, "Hard Concrete gate z ∈ [0, 1], one per head or unit", fontsize=9, va="center", color=ink)

    # (b) the trade-off: each metric of the pruned model relative to its fine-tuned teacher (1x)
    bx = fig.add_subplot(gs[0, 1])
    rows = [("Parameters", lambda e: e["params_total"] / 1e6, "{:.0f}M"),
            ("Compute (GMAC/s)", lambda e: e["gmacs_per_audio_second"], "{:.1f}"),
            ("Inference time (RTF)", None, "{:.4f}"),
            ("DDD CER (%)", lambda e: 100 * e["results"]["ddd_test"]["cer"], "{:.2f}"),
            ("FLEURS-km CER (%)", lambda e: 100 * e["results"]["fleurs_km_test"]["cer"], "{:.2f}"),
            ("LibriSpeech WER (%)", lambda e: 100 * e["results"]["libri_test_clean"]["wer"], "{:.2f}")]
    styles = {"large": dict(fc=OURS, label="sota-large vs fine-tuned 1B", dy=-0.17),
              "base": dict(fc=white, label="sota-base vs fine-tuned 300M", dy=0.17)}
    for size, st in styles.items():
        t, o = _eval(f"{size}_teacher"), _eval(size)
        for r, (name, get, fmt) in enumerate(rows):
            tv, ov = (_mean_rtf(f"{size}_teacher"), _mean_rtf(size)) if get is None else (get(t), get(o))
            ratio, y = ov / tv, r + st["dy"]
            bx.plot([1, ratio], [y, y], color=meta, lw=1.3, zorder=2)
            bx.scatter([1], [y], s=34, color=BASELINES[0], zorder=3)
            bx.scatter([ratio], [y], s=78, facecolor=st["fc"], edgecolor=OURS, linewidth=1.6, zorder=4)
            text = f"{fmt.format(tv)} → {fmt.format(ov)}" if get is not None else f"{tv / ov:.1f}× faster"
            side = -1 if ratio < 1 else 1
            bx.text(ratio * (0.86 if side < 0 else 1.14), y, text, ha="right" if side < 0 else "left", va="center",
                    fontsize=8.5, color=ink)
    bx.axvline(1, color=BASELINES[0], lw=1, ls="--", zorder=1)
    bx.set_xscale("log")
    bx.set_xlim(0.07, 4)
    bx.set_xticks([0.1, 0.25, 0.5, 1, 2], ["0.1×", "0.25×", "0.5×", "1×", "2×"])
    bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    bx.set_yticks(range(len(rows)), [r[0] for r in rows])
    bx.set_ylim(len(rows) - 0.4, -0.75)
    bx.grid(axis="x", color=border, ls="--", lw=0.7)
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Pruned model relative to its teacher (left of 1× is smaller, faster or more accurate)")
    sparsity = 100 * (1 - _eval("large")["params_encoder"] / _eval("large_teacher")["params_encoder"])
    bx.set_title(f"(b) {sparsity:.0f}% sparsity: what it buys and costs", loc="left", fontsize=13, weight="bold")
    bx.legend(handles=[legend_dot(OURS, styles["large"]["label"]),
                       matplotlib.lines.Line2D([], [], marker="o", ls="None", markersize=9, markerfacecolor=white,
                                               markeredgecolor=OURS, markeredgewidth=1.6, label=styles["base"]["label"]),
                       legend_dot(BASELINES[0], "teacher (1×)")],
              loc="lower left", fontsize=8.5)
    fig.savefig(HERE / "pruning.png")
    plt.close(fig)


if __name__ == "__main__":
    apply()
    plt.rcParams["hatch.color"] = CHART["figureBackground"]
    plt.rcParams["font.sans-serif"] = ["Kantumruy Pro", "DejaVu Sans"]  # Latin-only labels; Noto Sans Khmer has no Latin
    results()
    layers()
    pruning()
    print("wrote results.png, layers.png, pruning.png in", HERE)
