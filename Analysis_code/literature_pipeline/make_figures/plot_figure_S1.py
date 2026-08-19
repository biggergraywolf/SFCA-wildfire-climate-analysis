#!/usr/bin/env python3
"""Reproduce Supplementary Figure S1 from released screening diagnostics.

Compatible with Python 3.8.10. Run this file directly in VS Code, or use
--data-dir and --output-dir to temporarily override the default folders.
"""

from __future__ import annotations

import argparse
import json
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


COLORS = {
    "core": "#B2472D",
    "context": "#D99A4E",
    "peripheral": "#7C8795",
}

TIER_ORDER = [
    "Core_direct",
    "Contextual_or_indirect",
    "Peripheral_screening_candidate",
]

# CSV数据输入文件夹
DATA_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Supplementary_Figure_S1_screening_diagnostics\data"
)

# 图片输出文件夹
OUTPUT_DIR = Path(
    r"C:\Users\win10\Desktop\43_map_code\SFCA_V4.3_all_figure_source_data_and_code"
    r"\SFCA_V4.3_figure_packages\Supplementary_Figure_S1_screening_diagnostics\output"
)


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

    # 读取筛选概率记录和交叉验证诊断结果。
    records = pd.read_csv(
        args.data_dir / "screening_probability_records.csv"
    )
    diagnostics = json.loads(
        (args.data_dir / "ml_relevance_validation.json").read_text(
            encoding="utf-8"
        )
    )

    # 检查输入数据是否与发布版源数据一致。
    if (
        len(records) != 13012
        or int(records["included_in_probability_density"].sum()) != 12999
    ):
        raise ValueError(
            "Expected 13,012 assigned records and "
            "12,999 probability-density records."
        )

    tier_counts = (
        records["relevance_tier"]
        .value_counts()
        .reindex(TIER_ORDER)
    )

    if tier_counts.astype(int).tolist() != [8618, 2178, 2216]:
        raise ValueError(
            f"Unexpected tier counts: {tier_counts.to_dict()}"
        )

    retained = records.loc[
        records["included_in_probability_density"]
    ].copy()

    # 使用无网格白色背景，并将图中所有字体统一设置为10 pt。
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
        3,
        figsize=(14.0, 4.35),
    )

    # =========================================================
    # 子图a：筛选概率分布
    # =========================================================
    for tier, color, label in [
        ("Core_direct", COLORS["core"], "Core-direct"),
        ("Contextual_or_indirect", COLORS["context"], "Contextual"),
        (
            "Peripheral_screening_candidate",
            COLORS["peripheral"],
            "Peripheral",
        ),
    ]:
        axes[0].hist(
            retained.loc[
                retained["relevance_tier"].eq(tier),
                "ml_relevance_probability",
            ],
            bins=35,
            density=True,
            alpha=0.55,
            color=color,
            label=label,
        )

    axes[0].set_xlabel("Predicted relevance probability")
    axes[0].set_ylabel("Density")
    axes[0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        fontsize=10,
        borderaxespad=0,
    )
    axes[0].grid(False)

    # =========================================================
    # 子图b：规则恢复交叉验证混淆矩阵
    # =========================================================
    confusion = np.asarray(diagnostics["confusion_matrix"])

    sns.heatmap(
        confusion,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
        xticklabels=["Predicted −", "Predicted +"],
        yticklabels=["Seed −", "Seed +"],
        annot_kws={"fontsize": 10},
    )

    # =========================================================
    # 子图c：筛选层级记录数
    # =========================================================
    axes[2].bar(
        ["Core-direct", "Contextual", "Peripheral"],
        tier_counts,
        color=[
            COLORS["core"],
            COLORS["context"],
            COLORS["peripheral"],
        ],
    )

    for x_pos, value in enumerate(tier_counts.astype(int)):
        axes[2].text(
            x_pos,
            value + 100,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    axes[2].set_ylabel("Records")
    axes[2].tick_params(axis="x", rotation=20)

    # 额外预留顶部空间，防止8,618与上边框发生冲突。
    axes[2].set_ylim(
        0,
        float(tier_counts.max()) * 1.15,
    )
    axes[2].grid(False)

    # 统一三幅子图边框和刻度，确保边框与字体保持适当距离。
    for ax in axes:
        ax.tick_params(
            axis="both",
            labelsize=10,
            width=0.8,
            color="#4A4A4A",
        )

        for spine in ax.spines.values():
            spine.set_color("#BFC5C9")
            spine.set_linewidth(0.9)

    # 删除原分图标题，在各子图框内部右上角添加(a)、(b)和(c)。
    for ax, panel_letter in zip(
        axes,
        ("(a)", "(b)", "(c)"),
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
            color="#111111",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.2,
            },
            clip_on=True,
            zorder=10,
        )

    # 删除总标题，并手动调整边距以避免文字、数字和边框冲突。
    fig.subplots_adjust(
        left=0.060,
        right=0.985,
        top=0.965,
        bottom=0.190,
        wspace=0.225,
    )

    save_figure(
        fig,
        args.output_dir,
        "Supplementary_Figure_S1_screening_diagnostics",
    )


if __name__ == "__main__":
    main()
