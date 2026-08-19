#!/usr/bin/env python3
"""Reproduce Supplementary Figure S2 from concept-overlap tables.

Compatible with Python 3.8.10. Run this file directly in VS Code, or use
--data-dir and --output-dir to temporarily override the default folders.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# CSV数据输入文件夹
DATA_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Supplementary_Figure_S2_concept_coupling\data"
)

# 图片输出文件夹
OUTPUT_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Supplementary_Figure_S2_concept_coupling\output"
)


SELECTED = [
    "source_explicit",
    "climate_attribution",
    "fire_weather",
    "burned_area",
    "emissions_carbon",
    "radiative_temperature",
    "postfire_recovery",
    "feedback_closure",
]

# 图中概念名称的简写。仅改变显示标签，不修改CSV字段。
CONCEPT_ABBR = {
    "source_explicit": "SE",
    "climate_attribution": "CA",
    "fire_weather": "FW",
    "burned_area": "BA",
    "emissions_carbon": "CE",
    "radiative_temperature": "RT",
    "postfire_recovery": "PFR",
    "feedback_closure": "FC",
}

# 图b学科名称的简写。仅改变显示标签，不修改CSV字段。
DISCIPLINE_ABBR = {
    "ecology and biology": "EB",
    "forestry and land management": "FLM",
    "atmospheric and climate": "AC",
    "other or multidisciplinary": "OM",
    "earth observation and geosciences": "EOG",
    "health": "H",
    "engineering, policy and society": "EPS",
    "environmental sciences": "ES",
}

# 图b沿用发布版排序，该顺序与图a的因果链顺序有意不同。
DISCIPLINE_COLUMNS = [
    "burned_area",
    "climate_attribution",
    "emissions_carbon",
    "feedback_closure",
    "fire_weather",
    "postfire_recovery",
    "radiative_temperature",
    "source_explicit",
]


def normalize_label(value: object) -> str:
    """Normalize labels before applying display abbreviations."""
    return str(value).replace("_", " ").strip().lower()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """保存300 dpi的高质量JPG图片。"""
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

    # 读取数据。
    jaccard = pd.read_csv(
        args.data_dir / "sfca_concept_jaccard.csv",
        index_col=0,
    )
    discipline_concept = pd.read_csv(
        args.data_dir / "discipline_concept_index.csv"
    )
    discipline_summary = pd.read_csv(
        args.data_dir / "discipline_summary.csv"
    )

    # 检查Jaccard矩阵是否包含所有指定概念。
    if (
        not set(SELECTED).issubset(jaccard.index)
        or not set(SELECTED).issubset(jaccard.columns)
    ):
        raise ValueError(
            "The Jaccard table does not contain every selected SFCA concept."
        )

    top_disciplines = (
        discipline_summary.nlargest(8, "record_n")["broad_discipline"]
        .tolist()
    )

    # 使用无网格白色背景，并将所有字体统一设置为10 pt。
    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 10,
            "axes.edgecolor": "#BFC5C9",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.25),
    )

    # =========================================================
    # 子图a：概念两两共现
    # =========================================================
    selected_jaccard = jaccard.loc[SELECTED, SELECTED].copy()
    selected_jaccard.index = [
        CONCEPT_ABBR.get(value, value)
        for value in selected_jaccard.index
    ]
    selected_jaccard.columns = [
        CONCEPT_ABBR.get(value, value)
        for value in selected_jaccard.columns
    ]

    upper_triangle_mask = np.triu(
        np.ones(
            (len(SELECTED), len(SELECTED)),
            dtype=bool,
        ),
        1,
    )

    heatmap_a = sns.heatmap(
        selected_jaccard,
        mask=upper_triangle_mask,
        cmap="mako",
        vmin=0,
        vmax=1,
        linewidths=0,
        ax=axes[0],
        cbar_kws={
            "label": "Jaccard index",
            "pad": 0.045,
        },
    )

    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(
        axis="x",
        rotation=0,
        labelsize=10,
        pad=5,
    )
    axes[0].tick_params(
        axis="y",
        rotation=0,
        labelsize=10,
        pad=5,
    )
    axes[0].grid(False)

    # 图a色标字体统一为10 pt。
    colorbar_a = heatmap_a.collections[0].colorbar
    colorbar_a.ax.tick_params(labelsize=10)
    colorbar_a.ax.yaxis.label.set_size(10)

    # =========================================================
    # 子图b：学科不对称性
    # =========================================================
    matrix = (
        discipline_concept.loc[
            discipline_concept["discipline"].isin(top_disciplines)
            & discipline_concept["concept"].isin(DISCIPLINE_COLUMNS)
        ]
        .pivot(
            index="discipline",
            columns="concept",
            values="prevalence",
        )
        .fillna(0)
        .reindex(
            index=top_disciplines,
            columns=DISCIPLINE_COLUMNS,
        )
    )

    matrix.index = [
        DISCIPLINE_ABBR.get(normalize_label(value), str(value))
        for value in matrix.index
    ]
    matrix.columns = [
        CONCEPT_ABBR.get(value, value)
        for value in matrix.columns
    ]

    heatmap_b = sns.heatmap(
        matrix * 100,
        cmap="rocket_r",
        vmin=0,
        vmax=25,
        linewidths=0,
        ax=axes[1],
        cbar_kws={
            "label": "Within-discipline share (%)",
            "pad": 0.045,
        },
    )

    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(
        axis="x",
        rotation=0,
        labelsize=10,
        pad=5,
    )
    axes[1].tick_params(
        axis="y",
        rotation=0,
        labelsize=10,
        pad=5,
    )

    # 图b色标字体统一为10 pt。
    colorbar_b = heatmap_b.collections[0].colorbar
    colorbar_b.ax.tick_params(labelsize=10)
    colorbar_b.ax.yaxis.label.set_size(10)

    # 删除原分图标题，在两幅图框内部右上角添加(a)和(b)。
    for ax, panel_letter in zip(
        axes,
        ("(a)", "(b)"),
    ):
        ax.text(
            0.975,
            0.975,
            panel_letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="normal",
            clip_on=True,
            zorder=10,
        )

    # 删除总标题，并调整两幅图与色标之间的空间。
    fig.subplots_adjust(
        left=0.055,
        right=0.965,
        top=0.965,
        bottom=0.095,
        wspace=0.30,
    )

    save_figure(
        fig,
        args.output_dir,
        "Supplementary_Figure_S2_concept_coupling",
    )


if __name__ == "__main__":
    main()
