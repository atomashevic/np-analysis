#!/usr/bin/env python3
"""Audit first-person disclosure labels and summarize them by campaign phase."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pandas as pd

from manual_coding_reconciliation import normalize_text, reconcile_tweet_ids

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = (
    ROOT / "data/manual_coding/np_anonymized_originals_kodiranje_motiva_2026-07-17.xlsx"
)
SOURCE_ORIGINALS = ROOT / "data/np_anonymized_originals.xlsx"
TWEETS = ROOT / "data/np_without_duplicates.csv"
PHASES = ROOT / "analysis/retweet_network_phases/phase_boundaries.csv"
REVIEW_DECISIONS_CSV = ROOT / "data/manual_coding/x_owner_review_decisions.csv"
OUTDIR = ROOT / "analysis/manual_coding_motives"

CODE_SPLIT_RE = re.compile(r"[,\n]+")
EXPECTED_COUNTS = {
    "coded_rows": 4386,
    "analytic_tweets": 824,
    "analytic_detailed_assignments": 1266,
    "analytic_parent_assignments": 1238,
    "x_rows": 64,
    "x_owner_conflicts": 28,
}


def parse_codes(value: object) -> list[str]:
    return [
        part.strip() for part in CODE_SPLIT_RE.split(str(value or "")) if part.strip()
    ]


def motive_family(value: str) -> str:
    value = value.strip().casefold()
    if value == "x":
        return value
    return value.split(".", 1)[0]


def load_workbook() -> tuple[pd.DataFrame, set[str]]:
    manual = pd.read_excel(WORKBOOK, sheet_name="KODIRANJE MOTIVA", dtype=str).fillna(
        ""
    )
    columns = list(manual.columns)
    manual = manual.rename(
        columns={
            columns[0]: "workbook_tweet_id",
            columns[1]: "text",
            columns[2]: "reporting_status",
            columns[3]: "motives_raw",
            columns[4]: "confession_flag",
            columns[5]: "experience_owner",
        }
    )
    manual["workbook_row"] = manual.index + 2
    manual["text_normalized"] = manual["text"].map(normalize_text)
    manual["parsed_codes"] = manual["motives_raw"].map(parse_codes)

    codebook = pd.read_excel(WORKBOOK, sheet_name="Sheet1", dtype=str).fillna("")
    codebook_families = {
        motive_family(value) for value in codebook.iloc[:, 0] if str(value).strip()
    }
    return manual, codebook_families


def validate_coding_counts(
    manual: pd.DataFrame, valid_families: set[str]
) -> dict[str, int]:
    analytic_families = valid_families - {"x", "0"}
    parsed_families = manual["parsed_codes"].map(
        lambda values: [motive_family(value) for value in values]
    )
    has_analytic = parsed_families.map(
        lambda values: any(value in analytic_families for value in values)
    )
    counts = {
        "coded_rows": len(manual),
        "analytic_tweets": int(has_analytic.sum()),
        "analytic_detailed_assignments": sum(
            value in analytic_families for values in parsed_families for value in values
        ),
        "analytic_parent_assignments": sum(
            len({value for value in values if value in analytic_families})
            for values in parsed_families
        ),
        "x_rows": int(parsed_families.map(lambda values: "x" in values).sum()),
        "x_owner_conflicts": int(
            (
                parsed_families.map(lambda values: "x" in values)
                & manual["experience_owner"].str.strip().eq("1")
            ).sum()
        ),
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected reconciled counts: {counts}")
    return counts


def resolve_tweet_ids(manual: pd.DataFrame) -> pd.DataFrame:
    return reconcile_tweet_ids(
        manual,
        SOURCE_ORIGINALS,
        workbook_id_column="workbook_tweet_id",
        text_column="text",
    )


def assign_phases(manual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tweets = pd.read_csv(TWEETS, dtype=str).fillna("")
    tweets = tweets[tweets["post_type"] != "retweet"].drop_duplicates("id_str")
    tweets["timestamp"] = pd.to_datetime(tweets["time"], utc=True, errors="coerce")
    metadata = tweets.set_index("id_str")[["timestamp", "post_type"]]

    phases = pd.read_csv(PHASES)
    phases["start_utc"] = pd.to_datetime(phases["start_utc"], utc=True)
    phases["end_utc"] = pd.to_datetime(phases["end_utc"], utc=True)

    result = manual.join(metadata, on="resolved_tweet_id")
    if result["timestamp"].isna().any():
        raise ValueError("Resolved tweet IDs are missing timestamp metadata")
    result["phase_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["phase_label"] = "outside_phase_window"
    for phase in phases.itertuples(index=False):
        mask = (result["timestamp"] >= phase.start_utc) & (
            result["timestamp"] < phase.end_utc
        )
        result.loc[mask, "phase_id"] = int(phase.phase_id)
        result.loc[mask, "phase_label"] = str(phase.phase_label)

    workbook_timestamps = result["workbook_tweet_id"].map(metadata["timestamp"])
    result["workbook_phase_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for phase in phases.itertuples(index=False):
        mask = (workbook_timestamps >= phase.start_utc) & (
            workbook_timestamps < phase.end_utc
        )
        result.loc[mask, "workbook_phase_id"] = int(phase.phase_id)
    result["phase_changed_after_id_repair"] = (
        result["phase_id"] != result["workbook_phase_id"]
    ).fillna(result["phase_id"].notna() | result["workbook_phase_id"].notna())
    return result, phases


def apply_review(manual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = manual.copy()
    result["is_x"] = result["parsed_codes"].map(
        lambda values: any(motive_family(value) == "x" for value in values)
    )
    result["first_person_raw"] = result["experience_owner"].str.strip().eq("1")
    result["first_person_x_excluded"] = result["first_person_raw"] & ~result["is_x"]
    result["first_person_adjudicated"] = result["first_person_raw"]

    conflicts = result[result["is_x"] & result["first_person_raw"]].copy()
    decisions = pd.read_csv(
        REVIEW_DECISIONS_CSV,
        dtype={"resolved_tweet_id": str, "first_person_decision": int},
    ).set_index("resolved_tweet_id")
    conflict_ids = set(conflicts["resolved_tweet_id"])
    if conflict_ids != set(decisions.index):
        raise ValueError("Review decisions do not match the 28 ownership conflicts")

    review_rows: list[dict[str, object]] = []
    for order, row in enumerate(conflicts.itertuples(index=True), start=1):
        decision = decisions.loc[row.resolved_tweet_id]
        first_person = bool(decision["first_person_decision"])
        result.at[row.Index, "first_person_adjudicated"] = first_person
        review_rows.append(
            {
                "review_order": order,
                "workbook_row": row.workbook_row,
                "workbook_tweet_id": row.workbook_tweet_id,
                "resolved_tweet_id": row.resolved_tweet_id,
                "timestamp_utc": row.timestamp.isoformat(),
                "phase_id": row.phase_id,
                "post_type": row.post_type,
                "first_person_decision": first_person,
                "category": decision["category"],
                "rationale": decision["rationale"],
                "text_sha256": hashlib.sha256(row.text_normalized.encode()).hexdigest(),
                "text_excerpt": row.text_normalized[:180],
            }
        )
    return result, pd.DataFrame(review_rows)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


def build_phase_summary(manual: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    in_window = manual.dropna(subset=["phase_id"]).copy()
    summary = (
        in_window.groupby(["phase_id", "phase_label"])
        .agg(
            original_coding_units=("resolved_tweet_id", "size"),
            first_person_raw=("first_person_raw", "sum"),
            first_person_x_excluded=("first_person_x_excluded", "sum"),
            first_person_adjudicated=("first_person_adjudicated", "sum"),
        )
        .reset_index()
    )
    summary = summary.merge(
        phases[["phase_id", "start_utc", "end_utc", "duration_hours"]],
        on="phase_id",
        how="left",
    )
    for prefix, column in [
        ("raw", "first_person_raw"),
        ("x_excluded", "first_person_x_excluded"),
        ("adjudicated", "first_person_adjudicated"),
    ]:
        summary[f"{prefix}_share"] = summary[column] / summary["original_coding_units"]
    intervals = [
        wilson_interval(
            int(row.first_person_adjudicated), int(row.original_coding_units)
        )
        for row in summary.itertuples(index=False)
    ]
    summary["adjudicated_wilson_low"] = [interval[0] for interval in intervals]
    summary["adjudicated_wilson_high"] = [interval[1] for interval in intervals]
    return summary.sort_values("phase_id")


def markdown_table(summary: pd.DataFrame) -> str:
    rows = [
        "| Phase | Original coding units | First-person | Share | 95% Wilson interval |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.phase_id} | {row.original_coding_units:,} | "
            f"{row.first_person_adjudicated:,} | {row.adjudicated_share:.1%} | "
            f"{row.adjudicated_wilson_low:.1%}-{row.adjudicated_wilson_high:.1%} |"
        )
    return "\n".join(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manual, codebook_families = load_workbook()
    validate_coding_counts(manual, codebook_families)
    manual = resolve_tweet_ids(manual)
    manual, phases = assign_phases(manual)
    manual, review = apply_review(manual)
    summary = build_phase_summary(manual, phases)

    summary.to_csv(OUTDIR / "first_person_share_by_phase.csv", index=False)
    review.to_csv(OUTDIR / "x_owner_conflict_review.csv", index=False)
    manual[
        [
            "workbook_row",
            "workbook_tweet_id",
            "resolved_tweet_id",
            "resolution_method",
            "id_reassigned",
            "timestamp",
            "post_type",
            "phase_id",
            "phase_label",
            "workbook_phase_id",
            "phase_changed_after_id_repair",
            "reporting_status",
            "motives_raw",
            "confession_flag",
            "experience_owner",
            "is_x",
            "first_person_raw",
            "first_person_x_excluded",
            "first_person_adjudicated",
        ]
    ].to_csv(OUTDIR / "manual_coding_id_resolution.csv", index=False)
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
