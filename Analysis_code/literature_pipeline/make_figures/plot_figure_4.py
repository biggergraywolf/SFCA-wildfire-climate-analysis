#!/usr/bin/env python3
"""Reproduce manuscript Figure 4 from NMF topic summary tables.

Compatible with Python 3.8.10. Run directly in VS Code, or use the optional
``--data-dir`` and ``--output-dir`` arguments to override the default folders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


DATA_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Figure_4_topics_and_time\data"
)

OUTPUT_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Figure_4_topics_and_time\output"
)

# Enlarge every font in the figure by the same factor.
FONT_SCALE = 2.0


def scaled_font(size: float) -> float:
    """Return a font size multiplied by the global font scale."""
    return size * FONT_SCALE

# Saturated sequential palette: the light end remains sufficiently dark for
# clear display in print and on screen.
TOPIC_CMAP = LinearSegmentedColormap.from_list(
    "topic_teal_gradient",
    ["#163F52", "#205D69", "#2F7D7A", "#4C9C91"],
)

# Concise labels for the 16 NMF topics in panel a. These abbreviations only
# affect display; the original topic labels in the CSV files remain unchanged.
TOPIC_ABBR = {
    "risk governance, adaptation and resilience": "RGAR",
    "forest regeneration, mortality and resilience": "FRMR",
    "biodiversity, habitat and conservation": "BHC",
    "climate variability, drought and fire activity": "CVDFA",
    "future projections under climate scenarios": "FPCS",
    "smoke exposure, air quality and health": "SAQH",
    "palaeofire history and holocene regimes": "PHHR",
    "carbon and greenhouse-gas emissions": "CGE",
    "fire-weather danger indices": "FWDI",
    "soil biogeochemistry, permafrost and carbon": "SBPC",
    "compound extremes and climate-related hazards": "CECH",
    "remote sensing, mapping and monitoring": "RSMM",
    "burn severity and post-fire recovery": "BSPR",
    "machine-learning prediction and detection": "MLPD",
    "aerosol radiative forcing and transport": "ARFT",
    "fuel management, prescribed fire and suppression": "FMPFS",
}


def normalize_label(value: object) -> str:
    """Normalize a CSV topic label before applying its abbreviation."""
    return str(value).replace("_", " ").strip().lower()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save one high-quality 300 dpi JPEG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.jpg",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="jpg",
        pil_kwargs={"quality": 95, "subsampling": 0},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    topics = pd.read_csv(args.data_dir / "nmf_topics_labeled.csv")
    periods = pd.read_csv(args.data_dir / "nmf_topic_period_labeled.csv")
    if len(topics) != 16 or int(topics["record_n"].sum()) != 9563:
        raise ValueError("Expected 16 NMF topics summing to the 9,563-record primary corpus.")

    # One shared top-to-bottom order is used for both panels. This corrects the
    # row reversal in the earlier plotting script.
    ordered = topics.sort_values("share", ascending=False).reset_index(drop=True)
    topic_order = ordered["topic_id"].tolist()
    period_order = ["1900-1989", "1990-1999", "2000-2009", "2010-2019", "2020-2025"]
    matrix = (
        periods.pivot_table(
            index="topic_id", columns="analysis_period", values="share_within_period", fill_value=0
        )
        .reindex(index=topic_order, columns=period_order, fill_value=0)
    )

    # The original Seaborn font scale was 0.95. Multiplying it by 2 enlarges
    # all default text, including ordinary ticks and colorbar text.
    sns.set_theme(style="white", font_scale=0.95 * FONT_SCALE)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.edgecolor": "#BFC5C9",
            "axes.linewidth": 0.9,
            "axes.labelsize": scaled_font(11),
            "xtick.labelsize": scaled_font(9.5),
            "ytick.labelsize": scaled_font(9.5),
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20.0, 8.2),
        gridspec_kw={"width_ratios": [1.08, 1.12]},
    )

    y = np.arange(len(ordered))
    bar_height = 0.76
    bar_colors = TOPIC_CMAP(np.linspace(0.06, 0.94, len(ordered)))
    axes[0].barh(
        y,
        ordered["share"] * 100,
        height=bar_height,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    axes[0].set_yticks(y)
    topic_tick_labels = [
        TOPIC_ABBR.get(normalize_label(value), str(value))
        for value in ordered["topic_label"]
    ]
    axes[0].set_yticklabels(
        topic_tick_labels,
        fontsize=scaled_font(9.4),
    )
    axes[0].invert_yaxis()

    # Fit the vertical range closely to the first and last bars. The small
    # 0.06-unit pad prevents clipping while removing excess top/bottom space.
    vertical_pad = 0.06
    axes[0].set_ylim(
        len(ordered) - 1 + bar_height / 2 + vertical_pad,
        -bar_height / 2 - vertical_pad,
    )

    axes[0].set_xlabel("Share of primary corpus (%)")
    axes[0].grid(False)
    axes[0].tick_params(axis="y", length=0, pad=8)

    sns.heatmap(
        matrix * 100,
        ax=axes[1],
        cmap="YlOrBr",
        vmin=0,
        vmax=60,
        cbar_kws={"label": "Share within period (%)"},
        yticklabels=False,
        linewidths=0.35,
        linecolor="white",
    )
    axes[1].set_xlabel("Publication period")
    axes[1].set_ylabel("")
    period_tick_labels = [period.replace("-", "-\n", 1) for period in period_order]
    axes[1].set_xticklabels(
        period_tick_labels,
        rotation=0,
        ha="center",
    )
    axes[1].tick_params(
        axis="x",
        rotation=0,
        labelsize=scaled_font(9.5),
    )

    colorbar = axes[1].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=scaled_font(9.5))
    colorbar.ax.yaxis.label.set_size(scaled_font(11))

    # Lowercase panel letters are placed inside the upper-right corner.
    for ax, panel_letter in zip(axes, ("(a)", "(b)")):
        ax.text(
            0.995,
            0.985,
            panel_letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=scaled_font(10),
            fontweight="normal",
            color="#111111",
            clip_on=True,
            zorder=10,
        )

    fig.tight_layout(pad=1.0, w_pad=1.2)
    save_figure(fig, args.output_dir, "Figure_4_topics_and_time")


if __name__ == "__main__":
    main()
