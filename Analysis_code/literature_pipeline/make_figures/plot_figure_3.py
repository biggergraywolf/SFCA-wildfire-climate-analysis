#!/usr/bin/env python3
"""Reproduce manuscript Figure 3 from released figure-source tables.

Compatible with Python 3.8.10. Run this file directly in VS Code, or pass
--data-dir and --output-dir to temporarily override the default paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLORS = {
    "core": "#B2472D",
    "context": "#D99A4E",
    "peripheral": "#7C8795",
    "navy": "#183B56",
    "teal": "#2F7D7A",
}

# 所有字体统一放大为原来的2倍。
FONT_SCALE = 2.0


def scaled_font(size: float) -> float:
    """Return a font size enlarged by the global scale factor."""
    return size * FONT_SCALE


# CSV数据输入文件夹
DATA_DIR = Path(
    r"C:/Users/win10/Desktop/43_map_code/SFCA_V4.3_all_figure_source_data_and_code"
    r"/SFCA_V4.3_figure_packages/Figure_3_corpus_and_growth/data"
)

# 图片输出文件夹
OUTPUT_DIR = Path(
    r"C:/Users/win10/Desktop/43_map_code/SFCA_V4.3_all_figure_source_data_and_code"
    r"/SFCA_V4.3_figure_packages/Figure_3_corpus_and_growth/output"
)

# 子图a的5个纵坐标简写。
# R-WoS = Retrieved Web of Science records
# U-NR  = Unique and non-retracted records
# E-AR  = Eligible Article/Review records
# CYA   = Complete years with abstracts
# PTC   = Primary text-analysis corpus
FUNNEL_LABELS = [
    "R-WoS",
    "U-NR",
    "E-AR",
    "CYA",
    "PTC",
]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """保存300 dpi的高质量JPG图片。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / f"{stem}.jpg",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="jpg",
        pil_kwargs={
            "quality": 95,
            "subsampling": 0,
        },
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    # 读取数据
    funnel = pd.read_csv(
        args.data_dir / "corpus_funnel_counts.csv"
    )

    annual = pd.read_csv(
        args.data_dir / "annual_relevance_counts.csv"
    )

    # 检查漏斗图中的记录数量
    expected = [13012, 12999, 12744, 11625, 9563]

    if funnel["records"].astype(int).tolist() != expected:
        raise ValueError(
            f"Unexpected funnel counts: "
            f"{funnel['records'].tolist()}"
        )

    # 仅保留截至2025年的数据
    annual_plot = annual.loc[
        annual["publication_year"].le(2025)
    ].copy()

    # 检查年度数据总数
    if int(annual_plot["total"].sum()) != 11891:
        raise ValueError(
            "The 1982–2025 unique/non-retracted annual "
            "series must sum to 11,891."
        )

    # 全局绘图样式：所有字号均为原代码的2倍。
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": scaled_font(10),
            "axes.labelsize": scaled_font(11),
            "axes.edgecolor": "#BCC3C9",
            "axes.linewidth": 0.9,
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    # 放大画布，为放大2倍后的字体保留充足空间。
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.0, 7.5),
        gridspec_kw={
            "width_ratios": [1.00, 1.24]
        },
    )

    # =========================================================
    # 子图a：文献语料库构建流程
    # =========================================================

    y = np.arange(len(funnel))

    funnel_colors = [
        "#C4CED6",
        "#8FA2B2",
        "#59758C",
        COLORS["navy"],
        COLORS["teal"],
    ]

    axes[0].barh(
        y,
        funnel["records"],
        height=0.64,
        color=funnel_colors,
        edgecolor="white",
        linewidth=0.65,
        zorder=2,
    )

    axes[0].set_yticks(y)

    axes[0].set_yticklabels(
        FUNNEL_LABELS,
        fontsize=scaled_font(9.6),
    )

    axes[0].invert_yaxis()

    axes[0].tick_params(
        axis="y",
        length=0,
        pad=8,
    )

    # 动态预留右侧空间，确保所有数字均位于边框内部。
    maximum_count = float(
        funnel["records"].max()
    )

    value_offset = maximum_count * 0.012

    axes[0].set_xlim(
        0,
        maximum_count * 1.20,
    )

    axes[0].set_xlabel("Records")
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axes[0].grid(False
        # axis="x",
        # color="#E8EBED",
        # linewidth=0.65,
        # alpha=0.80,
        # zorder=0,
    )

    # 添加记录数量。
    for y_pos, value in zip(
        y,
        funnel["records"].astype(int),
    ):
        axes[0].text(
            value + value_offset,
            y_pos,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=scaled_font(9.7),
            color="#222222",
            clip_on=True,
            zorder=3,
        )

    # =========================================================
    # 子图b：年度文献增长与相关性层级
    # =========================================================

    axes[1].stackplot(
        annual_plot["publication_year"],
        annual_plot["Core_direct"],
        annual_plot["Contextual_or_indirect"],
        annual_plot[
            "Peripheral_screening_candidate"
        ],
        labels=[
            "Core-direct",
            "Contextual/indirect",
            "Peripheral candidate",
        ],
        colors=[
            COLORS["core"],
            COLORS["context"],
            COLORS["peripheral"],
        ],
        alpha=0.95,
        edgecolor="white",
        linewidth=0.75,
        zorder=2,
    )

    axes[1].set_xlim(1982, 2025)

    # 收紧纵轴上界，减少顶部无效留白，但仍保留适当呼吸空间。
    tier_columns = [
        "Core_direct",
        "Contextual_or_indirect",
        "Peripheral_screening_candidate",
    ]
    annual_total = annual_plot[tier_columns].sum(axis=1)
    peak_total = float(annual_total.max())
    axes[1].set_ylim(0, peak_total * 1.025)

    axes[1].set_xlabel("Year")
    axes[1].set_ylabel(
        "Publications per year"
    )
    axes[1].set_xticks(np.arange(1985, 2026, 5))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axes[1].grid(False)

    axes[1].legend(
        frameon=False,
        loc="upper left",
        ncol=1,
        fontsize=scaled_font(9.2),
        handlelength=1.6,
        labelspacing=0.35,
        borderaxespad=0.70,
    )

    # 统一主图边框和刻度，使两幅子图视觉风格一致。
    for ax in axes:
        ax.tick_params(axis="both", width=0.8, length=3.5, color="#4A4A4A")
        for spine in ax.spines.values():
            spine.set_color("#BCC3C9")
            spine.set_linewidth(0.9)

    # 在两幅子图图框内的右上角添加(a)和(b)。
    for ax, panel_letter in zip(axes, ("(a)", "(b)")):
        ax.text(
            0.985,
            0.985,
            panel_letter,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=scaled_font(10),
            fontweight="normal",
            color="black",
            clip_on=True,
            zorder=10,
        )

    # 调整整体布局。
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.155,
        wspace=0.22,
    )

    # 输出300 dpi JPG。
    save_figure(
        fig,
        args.output_dir,
        "Figure_3_corpus_and_growth",
    )


if __name__ == "__main__":
    main()
