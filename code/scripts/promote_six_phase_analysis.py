#!/usr/bin/env python3
"""Promote the six-phase NB partition to canonical analysis artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts import retweet_phase_analysis as phase_analysis
    from scripts import six_phase_counterfactual as six_phase
except ModuleNotFoundError:
    import retweet_phase_analysis as phase_analysis
    import six_phase_counterfactual as six_phase

ROOT = Path(__file__).resolve().parents[1]
NETWORK_OUT = ROOT / "analysis" / "retweet_network_phases"
EMOTION_OUT = ROOT / "analysis" / "emotions"
PHASE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
EMOTION_ORDER = [
    "sadness",
    "anger",
    "disgust",
    "fear",
    "joy",
    "trust",
    "anticipation",
    "surprise",
    "neutral",
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def load_network_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return phase_analysis.load_inputs(
        ROOT / "data" / "np_network.csv",
        ROOT / "data" / "np.csv",
        ROOT / "data" / "np_nodes.csv",
    )


def promote_tables(phases: pd.DataFrame) -> None:
    source = six_phase.DEFAULT_OUTDIR
    mapping = {
        "phase_boundaries.csv": "phase_boundaries.csv",
        "phase_snapshot_metrics.csv": "phase_snapshot_metrics.csv",
        "phase_node_metrics.csv": "phase_node_metrics.csv",
        "first_phase_author_cohort.csv": "first_phase_author_cohort.csv",
        "first_phase_author_cohort_activity.csv": "first_phase_author_cohort_activity.csv",
        "phase1_vs_later_author_retweet_comparison.csv": "phase1_vs_later_author_retweet_comparison.csv",
        "phase1_vs_later_author_post_outcomes.csv": "phase1_vs_later_author_post_outcomes.csv",
        "phase1_vs_later_author_retweet_summary.csv": "phase1_vs_later_author_retweet_summary.csv",
        "report.md": "report.md",
    }
    NETWORK_OUT.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in mapping.items():
        shutil.copy2(source / source_name, NETWORK_OUT / target_name)

    report_path = NETWORK_OUT / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "# Six-phase counterfactual results",
        "# Six-phase primary results",
        1,
    ).replace(
        "_Re-estimated phase-dependent manuscript results · 11 July 2026_",
        "_Canonical phase-dependent manuscript results · 11 July 2026_",
        1,
    )
    report_path.write_text(report, encoding="utf-8")

    model = pd.read_csv(
        ROOT / "analysis" / "retweet_phase_robustness" / "nb_3hour_model_selection.csv"
    )
    model["chosen"] = model["k"].eq(6)
    model["model"] = "shared-dispersion negative binomial, three-hour bins"
    model.to_csv(NETWORK_OUT / "phase_model_selection.csv", index=False)
    phases.to_csv(NETWORK_OUT / "phase_boundaries.csv", index=False)


def prepare_assigned_network(
    phases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, object]]:
    network, raw, nodes = load_network_inputs()
    events, _ = phase_analysis.prepare_retweet_events(network, raw)
    events, hourly = phase_analysis.build_hourly_counts(events)
    events["phase_id"] = six_phase.assign_phase_ids(events["timestamp_utc"], phases)
    assigned = events[events["phase_id"].notna()].copy()
    assigned["phase_id"] = assigned["phase_id"].astype(int)
    assigned = assigned.merge(
        phases[["phase_id", "phase_label"]], on="phase_id", how="left"
    )
    hourly["phase_id"] = six_phase.assign_phase_ids(hourly["hour_start_utc"], phases)
    hourly = hourly.merge(
        phases[["phase_id", "phase_label"]], on="phase_id", how="left"
    )
    _, _, graphs = phase_analysis.compute_snapshot_outputs(
        assigned, phases, nodes, top_share_pct=0.10
    )
    return assigned, hourly, nodes, graphs


def save_hourly_assignments(hourly: pd.DataFrame) -> None:
    output = hourly.copy()
    for column in ["hour_start_utc", "hour_end_utc"]:
        output[column] = output[column].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    output.to_csv(NETWORK_OUT / "hourly_retweet_counts.csv", index=False)


def plot_phase_timeline(hourly: pd.DataFrame, phases: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.0), constrained_layout=True)
    ax.bar(
        hourly["hour_start_utc"],
        hourly["n_retweets"],
        width=pd.Timedelta(hours=0.9),
        color="#4C78A8",
        alpha=0.82,
        linewidth=0,
    )
    ymax = float(hourly["n_retweets"].max())
    for idx, phase in enumerate(phases.itertuples(index=False)):
        ax.axvspan(
            phase.start_utc,
            phase.end_utc,
            color=PHASE_COLORS[idx],
            alpha=0.09,
            linewidth=0,
        )
        ax.axvline(phase.start_utc, color="#666666", linewidth=0.55, alpha=0.7)
        midpoint = phase.start_utc + (phase.end_utc - phase.start_utc) / 2
        ax.text(
            midpoint,
            ymax * 1.03,
            f"p{phase.phase_id}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax.set_ylim(0, ymax * 1.14)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Retweets per hour")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    for suffix in ["png", "pdf"]:
        fig.savefig(NETWORK_OUT / f"fig_hourly_counts_changepoints.{suffix}", dpi=400)
    plt.close(fig)


def update_tweet_phase_file(phases: pd.DataFrame) -> None:
    tweets = pd.read_csv(ROOT / "data" / "np_without_duplicates.csv", dtype=str)
    phase_ids = six_phase.assign_phase_ids(tweets["time"], phases)
    labels = phase_ids.map(
        {phase_id: f"phase_{phase_id}" for phase_id in phases["phase_id"]}
    )
    pd.DataFrame({"id_str": tweets["id_str"], "phase": labels.fillna("")}).to_csv(
        ROOT / "results" / "tweet_phase.csv", index=False
    )


def load_phase_emotion_data() -> pd.DataFrame:
    path = six_phase.DEFAULT_OUTDIR / "emotion_distribution_by_phase.csv"
    data = pd.read_csv(path)
    data = data[data["post_type"].isin(["post", "retweet"])].copy()
    expected = set(EMOTION_ORDER) | {"unknown"}
    if set(data["emotion"]) != expected:
        raise ValueError("Phase-emotion table does not contain the expected labels")
    if data.duplicated(["phase_id", "post_type", "emotion"]).any():
        raise ValueError("Phase-emotion table contains duplicate cells")
    grouped = data.groupby(["phase_id", "post_type"], observed=True)
    if not np.allclose(grouped["share"].sum().to_numpy(), 1.0):
        raise ValueError("Phase-emotion shares do not sum to one")
    if not np.allclose(data["share"], data["n"] / data["phase_post_type_total"]):
        raise ValueError("Phase-emotion shares do not match their denominators")
    return data


def emotion_matrix(data: pd.DataFrame, post_type: str) -> pd.DataFrame:
    subset = data[data["post_type"].eq(post_type)]
    return (
        subset.pivot(index="emotion", columns="phase_id", values="share")
        .reindex(index=EMOTION_ORDER, columns=range(1, 7))
        .fillna(0)
        .mul(100)
    )


def plot_emotion_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    panel_label: str,
    vmax: float,
) -> object:
    values = matrix.to_numpy()
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(6), [f"P{phase_id}" for phase_id in range(1, 7)])
    ax.set_yticks(
        range(len(EMOTION_ORDER)), [emotion.capitalize() for emotion in EMOTION_ORDER]
    )
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=6)
    ax.text(
        -0.16,
        1.04,
        panel_label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
    )
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        text_color = "#111111" if value >= 28 else "#FFFFFF"
        ax.text(
            column,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=6.5,
        )
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(EMOTION_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_emotion_phase_heatmaps(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.25), constrained_layout=True)
    image = plot_emotion_panel(
        axes[0],
        emotion_matrix(data, "post"),
        "Original posts (n=3,872)",
        "A",
        65.0,
    )
    plot_emotion_panel(
        axes[1],
        emotion_matrix(data, "retweet"),
        "Retweet-weighted source labels (n=20,416)",
        "B",
        65.0,
    )
    axes[1].set_yticklabels([])
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Within-phase share (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    paper_out = ROOT / "paper_figures" / "figures"
    paper_out.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        explicit = EMOTION_OUT / f"emotion_distribution_by_phase_heatmap.{suffix}"
        legacy = EMOTION_OUT / f"emotions_in_time_stacked_plots.{suffix}"
        manuscript = paper_out / f"emotion_distribution_by_phase_heatmap.{suffix}"
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(explicit, dpi=600)
        shutil.copyfile(explicit, legacy)
        shutil.copyfile(explicit, manuscript)
    plt.close(fig)


def plot_network_outputs(
    phases: pd.DataFrame,
    assigned: pd.DataFrame,
    nodes: pd.DataFrame,
    graphs: dict[int, object],
) -> None:
    node_metrics = pd.read_csv(NETWORK_OUT / "phase_node_metrics.csv")
    cohort_activity = pd.read_csv(
        NETWORK_OUT / "first_phase_author_cohort_activity.csv"
    )
    cohort_comparison = pd.read_csv(
        NETWORK_OUT / "phase1_vs_later_author_retweet_comparison.csv"
    )
    phase_analysis.plot_phase_distributions(
        node_metrics, NETWORK_OUT / "fig_phase_distributions.png"
    )
    phase_analysis.plot_first_phase_cohort_activity(
        cohort_activity, NETWORK_OUT / "fig_first_phase_author_cohort_activity.png"
    )
    phase_analysis.plot_phase1_vs_later_author_retweet_comparison(
        cohort_comparison,
        NETWORK_OUT / "fig_phase1_vs_later_author_retweet_comparison.png",
    )
    phase_analysis.plot_phase_networks(
        phase_df=phases,
        node_df=node_metrics,
        graph_map=graphs,
        nodes_df=nodes,
        outdir=NETWORK_OUT,
        viz_max_nodes=1500,
        seed=42,
    )
    for phase_id in [7, 8]:
        stale = NETWORK_OUT / f"fig_phase_{phase_id}_networks.png"
        if stale.exists():
            stale.unlink()


def main() -> None:
    configure_style()
    six_phase.main()
    phases = six_phase.load_phases(six_phase.DEFAULT_PHASES)
    promote_tables(phases)
    assigned, hourly, nodes, graphs = prepare_assigned_network(phases)
    save_hourly_assignments(hourly)
    plot_phase_timeline(hourly, phases)
    plot_network_outputs(phases, assigned, nodes, graphs)
    update_tweet_phase_file(phases)
    EMOTION_OUT.mkdir(parents=True, exist_ok=True)
    plot_emotion_phase_heatmaps(load_phase_emotion_data())


if __name__ == "__main__":
    main()
