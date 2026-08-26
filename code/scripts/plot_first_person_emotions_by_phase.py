#!/usr/bin/env python3
"""Plot phase distributions of primary emotion labels in first-person accounts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "analysis/manual_coding_motives"
SUMMARY = OUTDIR / "first_person_emotion_by_lifecycle_group.csv"
GROUPS = ["phase 1", "phase 2", "phase 3", "phases 4-6"]
EMOTIONS = ["sadness", "anger", "fear", "disgust", "Other"]
COLORS = {
    "sadness": "#0072B2",
    "anger": "#D55E00",
    "fear": "#CC79A7",
    "disgust": "#009E73",
    "Other": "#999999",
}
MARKERS = {"sadness": "o", "anger": "s", "fear": "^", "disgust": "D"}


def ordered(data: pd.DataFrame, emotion: str) -> pd.DataFrame:
    return data[data["emotion"] == emotion].set_index("phase_group").loc[GROUPS]


def main() -> None:
    data = pd.read_csv(SUMMARY)
    data = data[data["model"] == "gemma"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    x = np.arange(len(GROUPS))
    totals = data.drop_duplicates("phase_group").set_index("phase_group")["group_n"]
    labels = [f"{group.title()}\n(n={int(totals[group]):,})" for group in GROUPS]

    bottom = np.zeros(len(GROUPS))
    for emotion in EMOTIONS:
        values = ordered(data, emotion)["share"].to_numpy() * 100
        axes[0].bar(
            x,
            values,
            bottom=bottom,
            color=COLORS[emotion],
            label=emotion.title(),
        )
        bottom += values
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Share of first-person accounts (%)")
    axes[0].set_title("A  Dominant Gemma label")

    for emotion in EMOTIONS[:-1]:
        subset = ordered(data, emotion)
        values = subset["share"].to_numpy() * 100
        lower = (subset["share"] - subset["wilson_low"]).to_numpy() * 100
        upper = (subset["wilson_high"] - subset["share"]).to_numpy() * 100
        axes[1].errorbar(
            x,
            values,
            yerr=np.vstack([lower, upper]),
            color=COLORS[emotion],
            marker=MARKERS[emotion],
            linewidth=1.4,
            markersize=4.5,
            capsize=2.5,
            label=emotion.title(),
        )
    axes[1].set_ylim(0, 70)
    axes[1].set_ylabel("Label share (%)")
    axes[1].set_title("B  Major labels with 95% Wilson intervals")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7)
        axis.title.set_fontsize(9)
        axis.yaxis.label.set_fontsize(8)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
    )
    figure.savefig(
        OUTDIR / "fig_first_person_emotion_by_phase.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        OUTDIR / "fig_first_person_emotion_by_phase.pdf", bbox_inches="tight"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
