"""Draw the README figures from the training outputs (Netra chart style).

    python assets/make_figures.py      # reads training/runs/eval/*.json and */stage2_prune_s80/pruned.json

Writes assets/pruning.png (the encoder before and after pruning), assets/tradeoff.png (speed against error),
assets/results.png (test error, two model sizes) and assets/layers.png (what pruning kept in each layer).
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
    """Figure 1: the encoder before and after pruning, in the style of the DiariZen figure."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    ink, white = COLOR["foreground"], CHART["figureBackground"]
    kinds = {"cnn": (COLOR["accent-maroon"], COLOR["tint-maroon"]), "mha": (OURS, COLOR["tint-orange"]),
             "ffn": (COLOR["accent-yellow"], COLOR["tint-yellow"])}
    p = json.loads((RUNS / "large" / "stage2_prune_s80" / "pruned.json").read_text())
    cells, cw, ch, gap = 4, 3.0, 2.6, 0.75
    shown = (12, 40)  # one wide layer, and one of the deep layers that lost every head

    def kept(n, full):
        return 0 if n == 0 else max(1, round(cells * n / full))

    columns = [("cnn", cells)]
    for layer in shown:
        columns += [("mha", kept(p["num_heads"][layer - 1], 16)), ("ffn", kept(p["ffn_dims"][layer - 1], 5120))]

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set_xlim(0, 118)
    ax.set_ylim(3, 26)
    ax.set_aspect("equal")
    ax.axis("off")
    col_h = cells * ch + (cells - 1) * gap
    y0 = 10.5

    def cell(x, y, kind, on):
        solid, tint = kinds[kind]
        if on:
            ax.add_patch(Rectangle((x, y), cw, ch, fc=solid, ec=ink, lw=0.6))
        else:
            ax.add_patch(Rectangle((x, y), cw, ch, fc=tint, ec=solid, lw=1.0, ls=(0, (2.2, 1.6))))

    def dots(x):
        for dy in (-1.2, 0, 1.2):
            ax.plot(x, y0 + col_h / 2 + dy, "o", ms=2.2, color=ink)

    def model(x0, pruned, title):
        x, groups = x0 + 2.2, []
        for i, (kind, n) in enumerate(columns):
            if i in (1, 3):
                dots(x + 0.6)
                x += 2.4
                groups.append(x)
            if pruned and n == 0:
                continue
            if pruned:  # the survivors, packed into a shorter column
                top = y0 + col_h / 2 + (n * ch + (n - 1) * gap) / 2
                for j in range(n):
                    cell(x, top - (j + 1) * ch - j * gap, kind, True)
            else:
                on = set(np.random.default_rng(len(groups) * 7 + i).choice(cells, size=n, replace=False).tolist())
                for j in range(cells):
                    cell(x, y0 + j * (ch + gap), kind, j in on)
            x += cw + 1.0
        dots(x + 0.6)
        x += 2.6
        ax.add_patch(FancyBboxPatch((x0, y0 - 4.0), x - x0, col_h + 6.2, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="none", ec=ink, lw=1.3, ls=(0, (5, 3)) if pruned else "-"))
        ax.text((x0 + x) / 2, y0 - 5.4, title, ha="center", va="top", fontsize=12, color=ink)
        if not pruned:
            for gx, layer in zip(groups, shown):
                ax.text(gx + cw + 0.5, y0 - 0.7, f"layer {layer}", ha="center", va="top", fontsize=8.5,
                        color=COLOR["muted"])
        return x

    left = model(1, False, "Fine-tuned omniASR CTC-1B · 975M")
    ax.add_patch(FancyArrowPatch((left + 1.5, y0 + col_h / 2), (left + 21.5, y0 + col_h / 2), arrowstyle="-|>",
                                 mutation_scale=16, color=ink, lw=1.4))
    ax.text(left + 11.5, y0 + col_h / 2 + 1.2, "Pruning", ha="center", va="bottom", fontsize=12, color=ink)
    ax.text(left + 11.5, y0 + col_h / 2 - 1.2, f"{100 * p['sparsity']:.0f}% of encoder\nweights removed",
            ha="center", va="top", fontsize=9.5, color=COLOR["muted"])
    right = model(left + 23, True, f"sota-large · {p['params_total_after'] / 1e6:.0f}M")

    lx = right + 4.5
    for k, (kind, label) in enumerate([("cnn", "CNN"), ("mha", "MHA heads"), ("ffn", "FFN units")]):
        y = y0 + col_h - 1.2 - k * 4.4
        cell(lx, y, kind, True)
        ax.text(lx + cw + 1.2, y + ch / 2, label, va="center", fontsize=11, color=ink)
    cell(lx, y0 - 1.6, "mha", False)
    ax.text(lx + cw + 1.2, y0 - 1.6 + ch / 2, "pruned", va="center", fontsize=11, color=ink)
    fig.savefig(HERE / "pruning.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def tradeoff() -> None:
    """Figure 2: size, speed and error of each 80%-sparse model next to its fine-tuned teacher."""
    ink = COLOR["foreground"]
    pairs = [("large_teacher", "large", "Large"), ("base_teacher", "base", "Base")]
    panels = [("Size", "parameters (M), lower is smaller", lambda n: _eval(n)["params_total"] / 1e6, "{:.0f}"),
              ("Speed", "× its teacher, higher is faster", None, "{:.1f}×"),
              ("DDD", "Khmer CER (%), lower is better", lambda n: 100 * _eval(n)["results"]["ddd_test"]["cer"], "{:.2f}"),
              ("FLEURS-km", "Khmer CER (%), lower is better", lambda n: 100 * _eval(n)["results"]["fleurs_km_test"]["cer"], "{:.2f}"),
              ("LibriSpeech", "English WER (%), lower is better", lambda n: 100 * _eval(n)["results"]["libri_test_clean"]["wer"], "{:.2f}")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.9))
    w = 0.36
    for ax, (title, unit, get, fmt) in zip(axes, panels):
        vals = [(1.0, _mean_rtf(t) / _mean_rtf(o)) if get is None else (get(t), get(o)) for t, o, _ in pairs]
        top_all = max(max(v) for v in vals)
        top = 0
        for g, (t_name, o_name, group) in enumerate(pairs):
            if get is None:  # speed from the measured real-time factor
                tv, ov = 1.0, _mean_rtf(t_name) / _mean_rtf(o_name)
            else:
                tv, ov = get(t_name), get(o_name)
            top = max(top, tv, ov)
            ax.bar(g - w / 2 - 0.02, tv, w, color=BASELINES[0], zorder=3)
            ax.bar(g + w / 2 + 0.02, ov, w, **ours_bar())
            ax.text(g - w / 2 - 0.02, tv + 0.015 * top_all, fmt.format(tv), ha="center", va="bottom", fontsize=9, color=ink)
            change = "" if get is None else "\n" + f"{100 * (ov / tv - 1):+.0f}%".replace("-", "−")
            ax.text(g + w / 2 + 0.02, ov + 0.015 * top_all, f"{fmt.format(ov)}{change}", ha="center", va="bottom", fontsize=9,
                    color=ink, weight="bold", linespacing=1.15)
        ax.set_xticks(range(len(pairs)), [p[2] for p in pairs], fontsize=10.5)
        ax.set_xlim(-0.6, len(pairs) - 0.4)
        ax.set_ylim(0, top * 1.22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{title}\n", loc="center", fontsize=12, weight="bold")
        ax.text(0.5, 1.02, unit, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5, color=COLOR["muted"])
    fig.legend(handles=[legend_dot(BASELINES[0], "fine-tuned teacher"), legend_dot(OURS, "SOTA, 80% encoder sparsity")],
               loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.5)
    fig.savefig(HERE / "tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    apply()
    plt.rcParams["hatch.color"] = CHART["figureBackground"]
    plt.rcParams["font.sans-serif"] = ["Kantumruy Pro", "DejaVu Sans"]  # Latin-only labels; Noto Sans Khmer has no Latin
    results()
    layers()
    pruning()
    tradeoff()
    print("wrote results.png, layers.png, pruning.png, tradeoff.png in", HERE)
