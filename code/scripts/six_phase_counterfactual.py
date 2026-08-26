#!/usr/bin/env python3
"""Re-estimate phase-dependent results under the three-hour NB K=6 partition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

try:
    from scripts import analyze_manual_coding_motives as motive_analysis
    from scripts import retweet_phase_analysis as phase_analysis
except ModuleNotFoundError:
    import analyze_manual_coding_motives as motive_analysis
    import retweet_phase_analysis as phase_analysis

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASES = (
    ROOT
    / "analysis"
    / "retweet_phase_robustness"
    / "nb_3hour_phase_boundaries_bic_minimum.csv"
)
DEFAULT_OUTDIR = ROOT / "analysis" / "six_phase_counterfactual"
EMOTION_TRANSLATION = {
    "radost": "joy",
    "tuga": "sadness",
    "poverenje": "trust",
    "gadjenje": "disgust",
    "gađenje": "disgust",
    "strah": "fear",
    "bes": "anger",
    "iznenadjenje": "surprise",
    "iznenađenje": "surprise",
    "anticipacija": "anticipation",
    "iščekivanje": "anticipation",
    "neutralno": "neutral",
    "nepoznato": "unknown",
}
LIFECYCLE_GROUPS = {
    1: "emergence",
    2: "viral amplification",
    3: "contraction",
    4: "later afterlife",
    5: "later afterlife",
    6: "later afterlife",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phases(path: Path) -> pd.DataFrame:
    phases = pd.read_csv(path)
    phases["phase_id"] = phases["phase_id"].astype(int)
    phases["phase_label"] = phases["phase_id"].map(lambda value: f"phase_{value}")
    phases["start_utc"] = pd.to_datetime(phases["start_utc"], utc=True)
    phases["end_utc"] = pd.to_datetime(phases["end_utc"], utc=True)
    phases["start_hour_idx"] = phases["start_bin_idx"].astype(int) * 3
    phases["end_hour_idx"] = phases["end_bin_idx_exclusive"].astype(int) * 3 - 1
    return phases


def assign_phase_ids(times: pd.Series, phases: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(times, utc=True, errors="coerce")
    assigned = pd.Series(pd.NA, index=times.index, dtype="Int64")
    for phase in phases.itertuples(index=False):
        mask = (parsed >= phase.start_utc) & (parsed < phase.end_utc)
        assigned.loc[mask] = int(phase.phase_id)
    return assigned


def prepare_network_outputs(
    phases: pd.DataFrame, outdir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, nx.DiGraph], dict[str, int]]:
    network, raw, nodes = phase_analysis.load_inputs(
        ROOT / "data" / "np_network.csv",
        ROOT / "data" / "np.csv",
        ROOT / "data" / "np_nodes.csv",
    )
    events, qa = phase_analysis.prepare_retweet_events(network, raw)
    all_events = phase_analysis.prepare_all_events(network, raw)
    events, hourly = phase_analysis.build_hourly_counts(events)
    events["phase_id"] = assign_phase_ids(events["timestamp_utc"], phases)
    excluded = events[events["phase_id"].isna()].copy()
    assigned = events[events["phase_id"].notna()].copy()
    assigned["phase_id"] = assigned["phase_id"].astype(int)
    assigned = assigned.merge(
        phases[["phase_id", "phase_label"]], on="phase_id", how="left"
    )

    snapshots, node_metrics, graphs = phase_analysis.compute_snapshot_outputs(
        assigned, phases, nodes, top_share_pct=0.10
    )
    no_self = assigned[assigned["user_id"] != assigned["parent_user_id"]].copy()
    no_self_snapshots, _, _ = phase_analysis.compute_snapshot_outputs(
        no_self, phases, nodes, top_share_pct=0.10
    )
    cohort, cohort_activity = phase_analysis.compute_first_phase_cohort_activity(
        all_events=all_events,
        retweet_events=assigned,
        phase_df=phases,
    )
    cohort_comparison = (
        phase_analysis.compute_phase1_vs_later_author_retweet_comparison(
            all_events=all_events,
            retweet_events=assigned,
            phase_df=phases,
            followup_hours=12.0,
        )
    )
    post_outcomes = phase_analysis.compute_phase1_vs_later_author_post_outcomes(
        all_events=all_events,
        retweet_events=assigned,
        phase_df=phases,
        followup_hours=12.0,
    )
    cohort_summary = phase_analysis.compute_cluster_bootstrap_relative_likelihood(
        post_outcomes,
        bootstrap_reps=10_000,
        seed=42,
    )

    phases.to_csv(outdir / "phase_boundaries.csv", index=False)
    snapshots.to_csv(outdir / "phase_snapshot_metrics.csv", index=False)
    no_self_snapshots.to_csv(
        outdir / "phase_snapshot_metrics_no_self_loops.csv", index=False
    )
    node_metrics.to_csv(outdir / "phase_node_metrics.csv", index=False)
    cohort.to_csv(outdir / "first_phase_author_cohort.csv", index=False)
    cohort_activity.to_csv(
        outdir / "first_phase_author_cohort_activity.csv", index=False
    )
    cohort_comparison.to_csv(
        outdir / "phase1_vs_later_author_retweet_comparison.csv", index=False
    )
    post_outcomes.to_csv(
        outdir / "phase1_vs_later_author_post_outcomes.csv", index=False
    )
    cohort_summary.to_csv(
        outdir / "phase1_vs_later_author_retweet_summary.csv", index=False
    )
    excluded[["post_id", "timestamp_utc"]].to_csv(
        outdir / "excluded_retweet_events.csv", index=False
    )

    qa.update(
        {
            "assigned_retweet_events": int(len(assigned)),
            "excluded_retweet_events": int(len(excluded)),
            "duplicate_retweet_event_ids": int(events["post_id"].duplicated().sum()),
            "self_retweet_events": int(
                (events["user_id"] == events["parent_user_id"]).sum()
            ),
            "assigned_self_retweet_events": int(
                (assigned["user_id"] == assigned["parent_user_id"]).sum()
            ),
            "unique_directed_dyads": int(
                assigned[["user_id", "parent_user_id"]].drop_duplicates().shape[0]
            ),
        }
    )
    return snapshots, node_metrics, graphs, qa


def load_gemma() -> pd.DataFrame:
    rows = []
    path = (
        ROOT
        / "results"
        / "emotions_gemma"
        / "emotions_gemma4:31b_temp1.0_all_tweets.jsonl"
    )
    with path.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows.append(
                    {
                        "target_id_str": str(record["id"]),
                        "emotion": EMOTION_TRANSLATION.get(
                            record["emotion"], record["emotion"]
                        ),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("target_id_str", keep="last")


def prepare_emotion_outputs(phases: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    tweets = pd.read_csv(ROOT / "data" / "np_without_duplicates.csv", dtype=str)
    tweets["phase_id"] = assign_phase_ids(tweets["time"], phases)
    tweets = tweets.merge(load_gemma(), on="target_id_str", how="left")
    emotion_qa = (
        tweets.assign(
            matched_label=tweets["emotion"].notna(),
            assigned=tweets["phase_id"].notna(),
            assigned_and_labeled=tweets["phase_id"].notna() & tweets["emotion"].notna(),
        )
        .groupby("post_type")
        .agg(
            input_records=("id_str", "size"),
            matched_label_records=("matched_label", "sum"),
            assigned_records=("assigned", "sum"),
            assigned_labeled_records=("assigned_and_labeled", "sum"),
        )
        .reset_index()
    )
    emotion_qa.to_csv(outdir / "emotion_assignment_qa.csv", index=False)
    labeled = tweets[tweets["phase_id"].notna() & tweets["emotion"].notna()].copy()
    labeled["phase_id"] = labeled["phase_id"].astype(int)
    counts = (
        labeled.groupby(["phase_id", "post_type", "emotion"])
        .size()
        .rename("n")
        .reset_index()
    )
    counts["phase_post_type_total"] = counts.groupby(["phase_id", "post_type"])[
        "n"
    ].transform("sum")
    counts["share"] = counts["n"] / counts["phase_post_type_total"]
    counts.to_csv(outdir / "emotion_distribution_by_phase.csv", index=False)

    summary = (
        counts.sort_values(
            ["phase_id", "post_type", "n"], ascending=[True, True, False]
        )
        .groupby(["phase_id", "post_type"], as_index=False)
        .first()
        .rename(columns={"emotion": "dominant_emotion", "share": "dominant_share"})
    )
    summary.to_csv(outdir / "emotion_phase_summary.csv", index=False)
    return counts


def prepare_reason_outputs(
    phases: pd.DataFrame, outdir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        ROOT / "analysis" / "manual_coding_motives" / "manual_motive_label_table.csv",
        dtype={"tweet_id": str},
    )
    retweets = pd.read_csv(
        ROOT
        / "analysis"
        / "manual_coding_motives"
        / "manual_motive_retweet_label_table.csv",
        dtype={"tweet_id": str, "retweet_id": str},
    )
    source["phase_id"] = assign_phase_ids(source["post_time"], phases)
    retweets["phase_id"] = assign_phase_ids(retweets["retweet_time"], phases)
    source = source[source["phase_id"].notna()].copy()
    retweets = retweets[retweets["phase_id"].notna()].copy()
    source["phase_id"] = source["phase_id"].astype(int)
    retweets["phase_id"] = retweets["phase_id"].astype(int)
    source["lifecycle_group"] = source["phase_id"].map(LIFECYCLE_GROUPS)
    retweets["lifecycle_group"] = retweets["phase_id"].map(LIFECYCLE_GROUPS)
    source["post_phase_group"] = source["lifecycle_group"]

    source_summary = (
        source.groupby(["phase_id", "motive_family", "motive_label"])
        .agg(
            label_assignments=("tweet_id", "size"),
            unique_tweets=("tweet_id", "nunique"),
            fractional_source_weight=("fractional_tweet_weight", "sum"),
        )
        .reset_index()
    )
    source_summary["phase_fractional_total"] = source_summary.groupby("phase_id")[
        "fractional_source_weight"
    ].transform("sum")
    source_summary["phase_share"] = (
        source_summary["fractional_source_weight"]
        / source_summary["phase_fractional_total"]
    )
    source_summary.to_csv(
        outdir / "reason_source_distribution_by_phase.csv", index=False
    )

    label_distribution = pd.read_csv(
        ROOT / "analysis" / "manual_coding_motives" / "label_distribution.csv"
    )
    motive_analysis.PHASE_GROUP_ORDER = [
        "emergence",
        "viral amplification",
        "contraction",
        "later afterlife",
    ]
    visibility, global_test = motive_analysis.reason_visibility_fractional(
        source, label_distribution
    )
    visibility.to_csv(outdir / "reason_visibility_fractional.csv", index=False)
    global_test.to_csv(outdir / "reason_visibility_global_test.csv", index=False)

    retweet_summary = (
        retweets.groupby(["phase_id", "motive_family", "motive_label"])
        .agg(
            retweet_label_events=("retweet_id", "size"),
            unique_retweets=("retweet_id", "nunique"),
            fractional_retweet_weight=("fractional_retweet_event_weight", "sum"),
        )
        .reset_index()
    )
    retweet_summary["phase_fractional_total"] = retweet_summary.groupby("phase_id")[
        "fractional_retweet_weight"
    ].transform("sum")
    retweet_summary["phase_share"] = (
        retweet_summary["fractional_retweet_weight"]
        / retweet_summary["phase_fractional_total"]
    )
    retweet_summary.to_csv(
        outdir / "reason_retweet_distribution_by_phase.csv", index=False
    )

    lifecycle = (
        retweets.groupby(["lifecycle_group", "motive_family", "motive_label"])
        .agg(
            unique_retweets=("retweet_id", "nunique"),
            fractional_retweet_weight=("fractional_retweet_event_weight", "sum"),
        )
        .reset_index()
    )
    lifecycle["group_total"] = lifecycle.groupby("lifecycle_group")[
        "fractional_retweet_weight"
    ].transform("sum")
    lifecycle["group_share"] = (
        lifecycle["fractional_retweet_weight"] / lifecycle["group_total"]
    )
    lifecycle.to_csv(
        outdir / "reason_retweet_composition_lifecycle_groups.csv", index=False
    )

    top_families = set(
        retweets.groupby("motive_family")["fractional_retweet_event_weight"]
        .sum()
        .nlargest(6)
        .index
    )
    retweets["reason_group"] = retweets["motive_family"].where(
        retweets["motive_family"].isin(top_families), "Other"
    )
    collapsed = (
        retweets.groupby(["lifecycle_group", "reason_group"])
        .agg(fractional_retweet_count=("fractional_retweet_event_weight", "sum"))
        .reset_index()
    )
    lifecycle_totals = (
        retweets.groupby("lifecycle_group")
        .agg(
            phase_total_fractional=("fractional_retweet_event_weight", "sum"),
            phase_unique_retweets=("retweet_id", "nunique"),
        )
        .reset_index()
    )
    collapsed = collapsed.merge(lifecycle_totals, on="lifecycle_group", how="left")
    collapsed["phase_share"] = (
        collapsed["fractional_retweet_count"] / collapsed["phase_total_fractional"]
    )
    display_lookup = (
        retweets.drop_duplicates("motive_family")
        .set_index("motive_family")["motive_label"]
        .to_dict()
    )
    display_lookup["Other"] = "Other"
    collapsed["display_label"] = collapsed["reason_group"].map(display_lookup)
    collapsed.to_csv(outdir / "reason_phase_retweet_composition.csv", index=False)

    reason_emotion = (
        source.groupby(["phase_id", "motive_family", "motive_label", "emotion_gemma"])
        .agg(tweets=("tweet_id", "nunique"))
        .reset_index()
    )
    reason_emotion.to_csv(outdir / "reason_emotion_by_phase.csv", index=False)
    return source_summary, retweet_summary


def strict_percentile(value: float, population: np.ndarray) -> float:
    return float(100 * np.mean(population < value)) if population.size else np.nan


def markdown_table(frame: pd.DataFrame, decimals: int = 3) -> str:
    def format_value(value: object) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.{decimals}f}"
        return str(value)

    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *rows,
        ]
    )


def prepare_institutional_outputs(
    phases: pd.DataFrame,
    nodes: pd.DataFrame,
    graphs: dict[int, nx.DiGraph],
    outdir: Path,
) -> pd.DataFrame:
    registry = pd.read_csv(
        ROOT / "analysis" / "institutional_accounts" / "account_registry.csv"
    )
    institutions = set(
        registry.loc[registry["in_dataset"].eq("yes"), "username"].dropna()
    )
    cohort_ids = set(
        pd.read_csv(outdir / "first_phase_author_cohort.csv", dtype=str)["user_id"]
    )
    rows = []
    phase_summary = []
    for phase in phases.itertuples(index=False):
        phase_nodes = nodes[nodes["phase_id"].eq(phase.phase_id)].copy()
        graph = graphs[int(phase.phase_id)]
        pagerank = nx.pagerank(graph, weight="weight", max_iter=500, tol=1e-8)
        phase_nodes["pagerank"] = (
            phase_nodes["user_id"].astype(str).map(pagerank).fillna(0)
        )
        phase_nodes["is_institutional"] = phase_nodes["username"].isin(institutions)
        phase_nodes["is_phase1_author"] = (
            phase_nodes["user_id"].astype(str).isin(cohort_ids)
        )
        total_in_strength = phase_nodes["in_strength"].sum()
        inst_nodes = phase_nodes[phase_nodes["is_institutional"]]
        phase_summary.append(
            {
                "phase_id": int(phase.phase_id),
                "n_active_institutional": int(len(inst_nodes)),
                "institutional_in_strength": float(inst_nodes["in_strength"].sum()),
                "institutional_in_strength_share": float(
                    inst_nodes["in_strength"].sum() / total_in_strength
                )
                if total_in_strength
                else np.nan,
            }
        )
        if int(phase.phase_id) == 1:
            continue
        group_masks = {
            "institutional": phase_nodes["is_institutional"],
            "regular": ~phase_nodes["is_institutional"],
            "phase1_authors": phase_nodes["is_phase1_author"],
        }
        for metric in ["in_strength", "out_strength", "in_degree", "pagerank"]:
            population = phase_nodes[metric].to_numpy(dtype=float)
            for group, mask in group_masks.items():
                values = phase_nodes.loc[mask, metric].to_numpy(dtype=float)
                rows.append(
                    {
                        "phase_id": int(phase.phase_id),
                        "metric": metric,
                        "group": group,
                        "n": int(values.size),
                        "median": float(np.median(values)) if values.size else np.nan,
                        "mean": float(np.mean(values)) if values.size else np.nan,
                        "median_percentile": strict_percentile(
                            float(np.median(values)), population
                        )
                        if values.size
                        else np.nan,
                    }
                )
    pd.DataFrame(phase_summary).to_csv(
        outdir / "institutional_visibility_by_phase.csv", index=False
    )
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "institutional_group_metrics_by_phase.csv", index=False)
    return result


def write_report(
    phases: pd.DataFrame,
    snapshots: pd.DataFrame,
    emotions: pd.DataFrame,
    reasons: pd.DataFrame,
    institutions: pd.DataFrame,
    qa: dict[str, int],
    outdir: Path,
) -> None:
    post_emotions = emotions[emotions["post_type"].eq("post")]
    focus = post_emotions[post_emotions["emotion"].isin(["sadness", "anger", "fear"])]
    pivot = focus.pivot_table(
        index="phase_id", columns="emotion", values="share", fill_value=0
    )
    top_reasons = (
        reasons.sort_values(["phase_id", "phase_share"], ascending=[True, False])
        .groupby("phase_id")
        .head(3)
    )
    inst = institutions[
        institutions["group"].eq("institutional")
        & institutions["metric"].eq("in_strength")
    ]
    lines = [
        "# Six-phase counterfactual results",
        "",
        "_Re-estimated phase-dependent manuscript results · 11 July 2026_",
        "",
        "---",
        "",
        "## 📋 Phase definitions",
        "",
        markdown_table(
            phases[
                [
                    "phase_id",
                    "start_utc",
                    "end_utc",
                    "duration_hours",
                    "n_events",
                    "mean_retweets_per_hour",
                ]
            ]
        ),
        "",
        "## 📊 Network snapshots",
        "",
        markdown_table(snapshots),
        "",
        "## 📊 Emotion shares in source posts",
        "",
        markdown_table(pivot.reset_index()),
        "",
        "## 📊 Three largest reason shares within each phase",
        "",
        markdown_table(
            top_reasons[["phase_id", "motive_label", "unique_tweets", "phase_share"]]
        ),
        "",
        "## 📊 Institutional in-strength percentiles",
        "",
        markdown_table(inst[["phase_id", "n", "median", "mean", "median_percentile"]]),
        "",
        "## ✅ QA",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in qa.items())
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    phases_path = DEFAULT_PHASES
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    phases = load_phases(phases_path)
    snapshots, nodes, graphs, qa = prepare_network_outputs(phases, outdir)
    emotions = prepare_emotion_outputs(phases, outdir)
    reasons, _ = prepare_reason_outputs(phases, outdir)
    institutions = prepare_institutional_outputs(phases, nodes, graphs, outdir)
    write_report(phases, snapshots, emotions, reasons, institutions, qa, outdir)
    manifest = {
        "phase_source": str(phases_path.relative_to(ROOT)),
        "phase_source_sha256": sha256(phases_path),
        "output_directory": str(outdir.relative_to(ROOT)),
        "retweet_events_assigned": qa["assigned_retweet_events"],
        "retweet_events_excluded": qa["excluded_retweet_events"],
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
