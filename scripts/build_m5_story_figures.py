#!/usr/bin/env python3
"""M5 showcase story figures, themed to the fieldnotes system.

Every number is read from the frozen evidence files under
data/v2/evidence/ — nothing is typed into this script. Palette follows the
owner's final fieldnotes set: sumi-green naive #1E342E, sakura-rose
LightGBM #d4708c, plus two owner-approved challenger tones for the M5
figures only: sage #89927a (Chronos-2) and diluted crimson #b86a62 (NHITS).
Bar order never places rose beside crimson.

Outputs (2x export):
  frontend/public/images/story/m5_*.png          (showcase story tab)
  --portfolio-assets  <dir>/m5-benchmark-*.png   (portfolio dialog entry)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import json

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "data" / "v2" / "evidence"
STORY_IMAGES = REPO / "frontend" / "public" / "images" / "story"

SURFACE = "#fbf9f4"  # cream card surface the figures sit on
INK = "#25251f"
INK2 = "#52514e"
MUTED = "#77776c"
GRID = "#d8d1c4"
EDGE = "#c9c2b2"

NAIVE = "#1E342E"       # sumi-green (owner pick, neutral baseline)
LIGHTGBM = "#d4708c"    # sakura rose (owner pick)
CHRONOS = "#89927a"     # sage (site accent; owner-approved M5 assignment)
NHITS = "#b86a62"       # diluted crimson (owner-approved M5 assignment)
NEGATIVE = "#a43b36"    # established diverging negative

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.monospace": ["IBM Plex Mono", "Courier New", "DejaVu Sans Mono"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": EDGE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

DPI = 300


def _load_wrmsse() -> dict[str, float]:
    lgbm = json.loads((EVIDENCE / "m5_lightgbm" / "m5_lightgbm_summary.json").read_text())
    naive = json.loads(
        (EVIDENCE / "m5_seasonal_naive" / "m5_seasonal_naive_summary.json").read_text()
    )
    chronos = json.loads(
        (EVIDENCE / "m5_chronos_benchmark" / "m5_chronos_summary.json").read_text()
    )
    return {
        "naive_wrmsse": float(naive["overall_wrmsse"]),
        "lgbm_wrmsse": float(lgbm["challenger_wrmsse"]),
        "chronos_wrmsse": float(chronos["challenger_wrmsse"]),
    }


def _load_gate_scores() -> dict[str, float]:
    chronos = json.loads(
        (EVIDENCE / "m5_chronos_feasibility" / "m5_chronos_feasibility_summary.json").read_text()
    )["scores"]
    nhits = json.loads(
        (EVIDENCE / "m5_nhits_feasibility" / "m5_nhits_feasibility_summary.json").read_text()
    )["scores"]
    return {
        "naive": float(chronos["seasonal_naive"]),
        "lightgbm": float(chronos["global_lightgbm"]),
        "chronos": float(chronos["chronos_2_zero_shot"]),
        "nhits": float(nhits["nhits"]),
    }


def _bar_panel(ax, items: list[tuple[str, float, str]], title: str, note: str) -> None:
    names = [label for label, _, _ in items]
    values = [value for _, value, _ in items]
    colors = [color for _, _, color in items]
    positions = np.arange(len(items))
    ax.bar(positions, values, width=0.62, color=colors, edgecolor="none", zorder=3)
    ax.set_xticks(positions, names, fontsize=10.5, color=INK2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=10, loc="left")
    top = max(values)
    ax.set_ylim(0, top * 1.24)
    for position, value in zip(positions, values):
        ax.text(
            position, value + top * 0.03, f"{value:.3f}",
            ha="center", va="bottom", fontsize=10.5, color=INK,
        )
    ax.text(0.0, -0.17, note, transform=ax.transAxes, fontsize=8.5,
            color=MUTED, ha="left", va="top", fontfamily="IBM Plex Mono")


def fig_wrmsse_benchmark(scales: dict[str, float]) -> None:
    naive = scales["naive_wrmsse"]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    items = [
        ("seasonal naive", naive, NAIVE),
        ("global lightgbm", scales["lgbm_wrmsse"], LIGHTGBM),
        ("chronos-2 zero-shot", scales["chronos_wrmsse"], CHRONOS),
    ]
    _bar_panel(
        ax, items,
        "official m5 holdout — weighted (wrmsse), 12 hierarchy levels, lower is better",
        "d_1914–d_1941 · 42,840 nodes · weights ending d_1913",
    )
    ax.axhline(naive, color=NAIVE, lw=1.0, ls=(0, (4, 4)), alpha=0.55, zorder=2)
    ax.text(1.52, naive, "naive", fontsize=8.5, color=NAIVE, va="bottom",
            ha="center", alpha=0.8)
    ax.annotate(
        "champion · +6.5% vs naive", xy=(1, scales["lgbm_wrmsse"]),
        xytext=(1.02, naive * 1.55), fontsize=9.5, color=INK2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    ax.annotate(
        "underforecasts total units by 32.3%", xy=(2, scales["chronos_wrmsse"]),
        xytext=(1.28, scales["chronos_wrmsse"] * 0.9), fontsize=9.5, color=NEGATIVE,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    fig.tight_layout()
    _save(fig, "m5_wrmsse_benchmark.png", "m5-benchmark-wrmsse.png")


def fig_challenger_gates(gates: dict[str, float], scales: dict[str, float]) -> None:
    fig, (ax_gate, ax_full) = plt.subplots(1, 2, figsize=(9.6, 4.0), width_ratios=[1, 1])
    naive_gate = gates["naive"]
    threshold = naive_gate * 0.95  # predeclared: ≥5% better than seasonal naive

    # gate panel — rose never beside crimson, so the order is
    # naive | nhits | chronos | lightgbm (crimson flanked by sumi and sage)
    gate_items = [
        ("seasonal naive", naive_gate, NAIVE),
        ("nhits", gates["nhits"], NHITS),
        ("chronos-2", gates["chronos"], CHRONOS),
        ("lightgbm", gates["lightgbm"], LIGHTGBM),
    ]
    _bar_panel(
        ax_gate, gate_items,
        "the feasibility gate — 600-series subset rmsse",
        "predeclared: ≥5% over naive AND better than incumbent",
    )
    ax_gate.axhline(threshold, color=MUTED, lw=1.0, ls=(0, (4, 4)), zorder=2)
    ax_gate.text(3.45, threshold, "−5% line", fontsize=8.5, color=MUTED,
                 va="bottom", ha="right")
    ax_gate.annotate(
        "stops here — worse than the\nincumbent by 10.6%",
        xy=(1, gates["nhits"]), xytext=(0.30, gates["naive"] + 0.155),
        fontsize=9, color=INK2, ha="left",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )

    full_items = [
        ("seasonal naive", scales["naive_wrmsse"], NAIVE),
        ("global lightgbm", scales["lgbm_wrmsse"], LIGHTGBM),
        ("chronos-2 zero-shot", scales["chronos_wrmsse"], CHRONOS),
    ]
    _bar_panel(
        ax_full, full_items,
        "the full holdout — official wrmsse",
        "chronos cleared the gate, failed the benchmark",
    )
    ax_full.annotate(
        "passed the gate,\nfailed the benchmark",
        xy=(2, scales["chronos_wrmsse"]), xytext=(0.28, scales["chronos_wrmsse"] * 0.86),
        fontsize=9, color=INK2, ha="left",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    fig.tight_layout(w_pad=2.4)
    _save(fig, "m5_challenger_gates.png", "m5-benchmark-challenger-gates.png")


LEVEL_LABELS = {
    "total": "total", "state": "state", "store": "store",
    "category": "category", "department": "department",
    "state_category": "state × category", "state_department": "state × dept",
    "store_category": "store × category", "store_department": "store × dept",
    "item": "item", "item_state": "item × state", "item_store": "item × store",
}


def fig_chronos_by_level() -> None:
    levels = pd.read_csv(EVIDENCE / "m5_chronos_benchmark" / "m5_chronos_by_level.csv")
    levels = levels.iloc[::-1]  # bottom levels at the top of the chart
    improvements = levels["relative_improvement"].to_numpy() * 100
    labels = [LEVEL_LABELS[name] for name in levels["level_name"]]
    series_counts = levels["series_count"].to_numpy()
    colors = [CHRONOS if value > 0 else NEGATIVE for value in improvements]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    positions = np.arange(len(levels))
    ax.barh(positions, improvements, height=0.62, color=colors, zorder=3)
    ax.axvline(0, color=INK2, lw=1.0, zorder=4)
    ax.set_yticks(positions, labels, fontsize=10.5, color=INK2)
    ax.set_title(
        "chronos-2 vs seasonal naive by hierarchy level — relative improvement, %",
        fontsize=11.5, color=INK, pad=12, loc="left",
    )
    span = np.abs(improvements).max()
    # asymmetric limits: the positive side never exceeds +21%, so the right
    # half stays free for the series-count column
    ax.set_xlim(-span * 1.30, span * 0.55)
    for position, value in zip(positions, improvements):
        offset = span * 0.03 if value >= 0 else -span * 0.03
        ax.text(
            value + offset, position, f"{value:+.1f}%",
            va="center", ha="left" if value >= 0 else "right",
            fontsize=9.5, color=INK,
        )
    count_x = span * 0.53
    for position, count in zip(positions, series_counts):
        ax.text(count_x, position, f"{count:,}", va="center", ha="right",
                fontsize=8.5, color=MUTED, fontfamily="IBM Plex Mono")
    ax.text(count_x, len(levels) - 0.45, "series", va="bottom", ha="right",
            fontsize=8.5, color=MUTED, fontfamily="IBM Plex Mono", clip_on=False)
    ax.text(0.0, -0.085, "positive (sage) = chronos better · negative (red) = worse",
            transform=ax.transAxes, fontsize=9, color=MUTED, ha="left",
            va="top", fontfamily="IBM Plex Mono")
    fig.tight_layout()
    _save(fig, "m5_chronos_by_level.png", "m5-benchmark-chronos-by-level.png")


def fig_method_schematic() -> None:
    """Card preview: the M5 gate flow, placeholder-SVG style, no metrics."""
    card_surface = "#e4e4d7"
    ink = "#25251f"
    ink2 = "#52514e"
    muted = "#6e6d66"
    sage = "#89927a"
    crimson = "#a43b36"
    wine = "#482728"
    rose = "#d4708c"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": card_surface,
        "axes.facecolor": card_surface,
        "text.color": ink,
    })

    fig, ax = plt.subplots(figsize=(6.42, 3.60))
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])  # full bleed: box widths map to the real canvas
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, sub=None, edge=sage, face="#fbf9f4", fs=11.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3))
        cy = y + h / 2 + (0.045 if sub else 0)
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=fs, color=ink, zorder=4)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.075, sub, ha="center", va="center",
                    fontsize=8.6, color=muted, zorder=4, fontfamily="IBM Plex Mono")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=ink2, lw=1.4,
                                    shrinkA=2, shrinkB=2))

    box(0.02, 0.60, 0.29, 0.24, "walmart m5", "30,490 series · kaggle", edge=ink2)
    box(0.02, 0.22, 0.29, 0.24, "source gate", "sha-pinned · local files", edge=ink2)
    box(0.37, 0.41, 0.27, 0.30, "model suite", "naive · lightgbm\nnhits · chronos-2", edge=ink2)
    box(0.71, 0.62, 0.27, 0.22, "feasibility gate", "600-series subset", edge=sage)
    box(0.71, 0.20, 0.27, 0.22, "official holdout", "12-level wrmsse", edge=crimson)

    arrow(0.31, 0.72, 0.37, 0.66)
    arrow(0.31, 0.34, 0.37, 0.46)
    arrow(0.64, 0.64, 0.71, 0.72)
    arrow(0.845, 0.62, 0.845, 0.42)  # gate passes down to the holdout
    ax.text(0.862, 0.53, "clears", fontsize=8.5, color=muted, ha="left",
            va="center", fontfamily="IBM Plex Mono")

    ax.text(0.5, 0.06, "two stopped challengers, one champion — same rules as the census track",
            ha="center", fontsize=10.5, color=ink2)
    ax.text(0.5, 0.945, "the m5 gate pipeline", ha="center", fontsize=13.5, color=ink)
    _save_card(fig, "m5_method_schematic.png", "m5-benchmark-method-schematic.png")


def _save(fig: plt.Figure, story_name: str, portfolio_name: str) -> None:
    STORY_IMAGES.mkdir(parents=True, exist_ok=True)
    story_path = STORY_IMAGES / story_name
    fig.savefig(story_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"written: {story_path}")
    if _save.portfolio_assets is not None:
        portfolio_target = _save.portfolio_assets / portfolio_name
        shutil.copyfile(story_path, portfolio_target)
        print(f"copied:  {portfolio_target}")


def _save_card(fig: plt.Figure, story_name: str, portfolio_name: str) -> None:
    """Save the card schematic with invisible aspect wings (2.004 target)."""
    import io

    card_surface = "#e4e4d7"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=card_surface, bbox_inches="tight",
                pad_inches=0.2)
    buf.seek(0)
    img = plt.imread(buf)
    h, w = img.shape[:2]
    target = 320.7 / 160
    wing = matplotlib.colors.to_rgba(card_surface)
    if w < h * target:
        pad = int(round(h * target)) - w
        wings = np.full((h, 1, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 1), img,
                              wings.repeat(pad - pad // 2, 1)], axis=1)
    elif h < w / target:
        pad = int(round(w / target)) - h
        wings = np.full((1, w, 4), wing)
        img = np.concatenate([wings.repeat(pad // 2, 0), img,
                              wings.repeat(pad - pad // 2, 0)], axis=0)
    STORY_IMAGES.mkdir(parents=True, exist_ok=True)
    story_path = STORY_IMAGES / story_name
    plt.imsave(story_path, img)
    print(f"written: {story_path} ({img.shape[1]}x{img.shape[0]}, "
          f"aspect {img.shape[1] / img.shape[0]:.3f})")
    if _save.portfolio_assets is not None:
        portfolio_target = _save.portfolio_assets / portfolio_name
        shutil.copyfile(story_path, portfolio_target)
        print(f"copied:  {portfolio_target}")


_save.portfolio_assets = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--portfolio-assets", type=Path, default=None,
        help="also copy each figure into this directory (portfolio-site/assets)",
    )
    args = parser.parse_args()
    _save.portfolio_assets = args.portfolio_assets
    scales = _load_wrmsse()
    gates = _load_gate_scores()
    fig_wrmsse_benchmark(scales)
    fig_challenger_gates(gates, scales)
    fig_chronos_by_level()
    fig_method_schematic()


if __name__ == "__main__":
    main()
