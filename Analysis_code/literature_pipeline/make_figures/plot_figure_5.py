#!/usr/bin/env python3
"""Reproduce manuscript Figure 5 from released SFCA summary tables.

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


COLORS = {"core": "#B2472D", "navy": "#183B56", "teal": "#2F7D7A"}

# Uniform multiplier for every font in the figure.
FONT_SCALE = 1.5


def scaled_font(size: float) -> float:
    """Return a font size enlarged by the global multiplier."""
    return size * FONT_SCALE


DATA_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Figure_5_sfca_evidence_chain\data"
)

OUTPUT_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Figure_5_sfca_evidence_chain\output"
)

# Saturated sequential palette for panel a. The light end remains sufficiently
# dark for clear display in print and on screen.
CONCEPT_CMAP = LinearSegmentedColormap.from_list(
    "sfca_blue_teal_gradient",
    ["#153B54", "#1D5365", "#286E72", "#438F85"],
)

# Short display labels. These mappings do not modify the source CSV files.
CONCEPT_ABBR = {
    "emissions carbon": "CE",
    "fire weather": "FW",
    "burned area": "BA",
    "fire severity": "FS",
    "nonco2 biophysical": "NCB",
    "fuel legacy": "FL",
    "suppression control": "SC",
    "radiative temperature": "RT",
    "postfire recovery": "PFR",
    "source explicit": "SR",
    "human ignition": "HI",
    "natural ignition": "NI",
    "climate attribution": "CA",
    "feedback closure": "FC",
    "counterfactual": "CF",
}

LADDER_ABBR = {
    "primary wildfire-climate corpus": "Corpus",
    "ignition/source explicitly resolved": "Source",
    "+ climate-attribution language": "+CA",
    "+ fire-response endpoint": "+FR",
    "+ emission/biophysical endpoint": "+EBP",
    "+ explicit counterfactual": "+CF",
    "+ feedback closure": "+FC",
}

# Muted stage colors for panel b. The palette reuses Figure 5's blue-teal,
# navy and warm-rust families so each stage is distinct without looking
# disconnected from panels a and c.
LADDER_COLOR_MAP = {
    "primary wildfire-climate corpus": "#2F7D7A",
    "ignition/source explicitly resolved": "#4C7C8A",
    "+ climate-attribution language": "#B2472D",
    "+ fire-response endpoint": "#C66A3D",
    "+ emission/biophysical endpoint": "#C99A47",
    "+ explicit counterfactual": "#6B7E91",
    "+ feedback closure": "#183B56",
}

ENDPOINT_ABBR = {
    "climate attribution": "CA",
    "burned area": "BA",
    "fire severity": "FS",
    "emissions carbon": "CE",
    "nonco2 biophysical": "NCB",
    "radiative temperature": "RT",
    "postfire recovery": "PFR",
    "feedback closure": "FC",
}

CLASS_ABBR = {
    "both or comparative": "Both/comp.",
    "human only": "Human",
    "natural only": "Natural",
    "not resolved": "Unresolved",
}


def normalize_label(value: object) -> str:
    """Normalize CSV labels before applying display abbreviations."""
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

    concepts = pd.read_csv(args.data_dir / "sfca_concept_summary.csv")
    ladder = pd.read_csv(args.data_dir / "sfca_evidence_ladder.csv")
    endpoint_long = pd.read_csv(args.data_dir / "endpoint_coverage_by_ignition_class.csv")
    denominators = pd.read_csv(args.data_dir / "ignition_class_denominators.csv")
    if int(concepts["record_n"].max()) > 9563 or int(ladder.iloc[0]["record_n"]) != 9563:
        raise ValueError("SFCA summaries must use the 9,563-record primary corpus.")
    if int(denominators["record_n"].sum()) != 9563:
        raise ValueError("Ignition/source-class denominators must sum to 9,563.")

    class_order = denominators.sort_values("record_n", ascending=True)  # retained only for validation
    del class_order
    row_order = (
        endpoint_long.sort_values("class_order")[["ignition_class", "class_label"]]
        .drop_duplicates()["ignition_class"]
        .tolist()
    )
    row_labels = (
        endpoint_long.sort_values("class_order")[["ignition_class", "class_label"]]
        .drop_duplicates()
        .set_index("ignition_class")["class_label"]
        .to_dict()
    )
    column_order = (
        endpoint_long.sort_values("endpoint_order")[["endpoint_key", "endpoint_label"]]
        .drop_duplicates()["endpoint_key"]
        .tolist()
    )
    column_labels = (
        endpoint_long.sort_values("endpoint_order")[["endpoint_key", "endpoint_label"]]
        .drop_duplicates()
        .set_index("endpoint_key")["endpoint_label"]
        .to_dict()
    )
    endpoint_matrix = endpoint_long.pivot(
        index="ignition_class", columns="endpoint_key", values="share_percent"
    ).reindex(index=row_order, columns=column_order)
    endpoint_matrix.index = [
        CLASS_ABBR.get(normalize_label(row_labels[value]), row_labels[value])
        for value in endpoint_matrix.index
    ]
    endpoint_matrix.columns = [
        ENDPOINT_ABBR.get(normalize_label(column_labels[value]), column_labels[value])
        for value in endpoint_matrix.columns
    ]

    # Original Seaborn font scale was 0.9; multiplying by 1.5 enlarges all
    # default text, including ordinary ticks, heatmap annotations and colorbar text.
    sns.set_theme(style="white", font_scale=0.9 * FONT_SCALE)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.edgecolor": "#BFC5C9",
            "axes.linewidth": 0.9,
            "axes.labelsize": scaled_font(10.5),
        }
    )

    fig = plt.figure(figsize=(14.2, 9.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 1.0],
        height_ratios=[0.90, 1.05],
        wspace=0.26,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    ordered_concepts = concepts.sort_values("share", ascending=False).reset_index(drop=True)
    y_a = np.arange(len(ordered_concepts))
    bar_height = 0.74
    concept_colors = CONCEPT_CMAP(np.linspace(0.05, 0.95, len(ordered_concepts)))
    ax_a.barh(
        y_a,
        ordered_concepts["share"] * 100,
        height=bar_height,
        color=concept_colors,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    concept_tick_labels = [
        CONCEPT_ABBR.get(normalize_label(value), str(value))
        for value in ordered_concepts["concept"]
    ]
    ax_a.set_yticks(y_a)
    ax_a.set_yticklabels(
        concept_tick_labels,
        fontsize=scaled_font(9.5),
    )
    ax_a.invert_yaxis()

    # Fit panel a closely to the first and last bars, retaining only a small
    # safety margin to prevent clipping.
    vertical_pad = 0.06
    ax_a.set_ylim(
        len(ordered_concepts) - 1 + bar_height / 2 + vertical_pad,
        -bar_height / 2 - vertical_pad,
    )

    ax_a.set_xlabel("Primary corpus mentioning concept (%)")
    ax_a.tick_params(axis="y", length=0, pad=7)
    ax_a.grid(False)

    y_b = np.arange(len(ladder))
    ladder_colors = [
        LADDER_COLOR_MAP.get(normalize_label(stage), "#607D8B")
        for stage in ladder["stage"]
    ]
    ax_b.barh(
        y_b,
        ladder["record_n"],
        color=ladder_colors,
        edgecolor="white",
        linewidth=0.55,
    )
    ladder_tick_labels = [
        LADDER_ABBR.get(normalize_label(value), str(value))
        for value in ladder["stage"]
    ]
    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels(
        ladder_tick_labels,
        fontsize=scaled_font(9.2),
    )
    ax_b.invert_yaxis()
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.set_xlim(0, 35000)
    ax_b.set_xlabel("Records (symlog scale)")
    ax_b.tick_params(axis="y", pad=7)
    ax_b.grid(False)
    for y_pos, value in zip(y_b, ladder["record_n"].astype(int)):
        if value >= 100:
            label_x = value * 1.05
        elif value > 0:
            label_x = value + 0.28
        else:
            label_x = 0.13
        ax_b.text(
            label_x,
            y_pos,
            f"{value:,}",
            va="center",
            fontsize=scaled_font(8),
        )

    sns.heatmap(
        endpoint_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=45,
        ax=ax_c,
        cbar_kws={"label": "Within-source-class share (%)"},
    )
    ax_c.set_xlabel("Downstream endpoint")
    ax_c.set_ylabel("")
    ax_c.tick_params(
        axis="x",
        rotation=0,
        labelsize=scaled_font(8.5),
    )
    ax_c.tick_params(
        axis="y",
        rotation=0,
        labelsize=scaled_font(8.5),
    )

    # Lowercase panel letters replace the former panel titles.
    for ax, panel_letter in zip(
        (ax_a, ax_b, ax_c),
        ("(a)", "(b)", "(c)"),
    ):
        ax.text(
            0.995,
            0.990,
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

    fig.subplots_adjust(left=0.10, right=0.965, top=0.965, bottom=0.10)
    save_figure(fig, args.output_dir, "Figure_5_sfca_coverage_and_ladder_endpoint_gap")


if __name__ == "__main__":
    main()
