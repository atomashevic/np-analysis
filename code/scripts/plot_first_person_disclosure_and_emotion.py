#!/usr/bin/env python3
"""Combine first-person disclosure prevalence and emotion composition by phase."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "analysis/manual_coding_motives"
DISCLOSURE = OUTDIR / "first_person_share_by_phase.csv"
EMOTIONS = OUTDIR / "first_person_emotion_by_lifecycle_group.csv"
GROUPS = ["phase 1", "phase 2", "phase 3", "phases 4-6"]
EMOTION_ORDER = ["sadness", "anger", "fear", "disgust", "Other"]
COLORS = {
    "sadness": "#4E79A7",
    "anger": "#E15759",
    "fear": "#F28E2B",
    "disgust": "#76B7B2",
    "Other": "#A79F9F",
}


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=7)
    axis.title.set_fontsize(9)
    axis.yaxis.label.set_fontsize(8)
    axis.xaxis.label.set_fontsize(8)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)


def plot_disclosure(axis: plt.Axes, data: pd.DataFrame) -> None:
    data = data.sort_values("phase_id")
    x = np.arange(len(data))
    share = data["adjudicated_share"].to_numpy() * 100
    lower = (data["adjudicated_share"] - data["adjudicated_wilson_low"]).to_numpy()
    upper = (data["adjudicated_wilson_high"] - data["adjudicated_share"]).to_numpy()
    axis.errorbar(
        x,
        share,
        yerr=np.vstack([lower, upper]) * 100,
        color="#333333",
        marker="o",
        markerfacecolor="#56B4E9",
        markeredgecolor="#222222",
        linewidth=1.5,
        markersize=5,
        capsize=3,
    )
    labels = [
        f"{int(row.phase_id)}\nn={int(row.original_coding_units):,}"
        for row in data.itertuples(index=False)
    ]
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 52)
    axis.set_xlabel("Campaign phase")
    axis.set_ylabel("First-person disclosure share (%)")
    axis.set_title("A  Decline in first-person disclosure")


def plot_emotions(axis: plt.Axes, data: pd.DataFrame) -> None:
    data = data[data["model"] == "gemma"].copy()
    x = np.arange(len(GROUPS))
    totals = data.drop_duplicates("phase_group").set_index("phase_group")["group_n"]
    labels = [f"{group.title()}\nn={int(totals[group]):,}" for group in GROUPS]
    bottom = np.zeros(len(GROUPS))
    for emotion in EMOTION_ORDER:
        subset = data[data["emotion"] == emotion].set_index("phase_group").loc[GROUPS]
        values = subset["share"].to_numpy() * 100
        axis.bar(
            x,
            values,
            bottom=bottom,
            color=COLORS[emotion],
            label=emotion.title(),
            width=0.78,
        )
        bottom += values
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Campaign phase group")
    axis.set_ylabel("Emotion-label composition (%)")
    axis.set_title("B  Emotion labels of first-person accounts (Gemma 4 31B)")


def main() -> None:
    disclosure = pd.read_csv(DISCLOSURE)
    emotions = pd.read_csv(EMOTIONS)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.35),
        gridspec_kw={"width_ratios": [1.05, 1]},
        constrained_layout=True,
    )
    plot_disclosure(axes[0], disclosure)
    plot_emotions(axes[1], emotions)
    for axis in axes:
        style_axis(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
    )
    stem = OUTDIR / "fig_first_person_disclosure_and_emotion_by_phase"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
