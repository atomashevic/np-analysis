#!/usr/bin/env python3
"""Analyze manually coded non-reporting motives for #NisamPrijavila."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "analysis" / "manual_coding_motives"

CODED_TWEETS = ROOT / "data" / "manual_coding" / "coded_tweets.csv"
MOTIVE_LONG = ROOT / "data" / "manual_coding" / "motive_annotations_long.csv"
MOTIVE_CODEBOOK = ROOT / "data" / "manual_coding" / "motive_codebook.csv"
TWEETS = ROOT / "data" / "np_without_duplicates.csv"
PHASES = ROOT / "analysis" / "retweet_network_phases" / "phase_boundaries.csv"
INSTITUTIONS = ROOT / "analysis" / "institutional_accounts" / "account_registry.csv"
GPT_EMOTIONS = (
    ROOT / "results" / "emotions_gpt" / "emotions_gpt-5.4_temp1.0_all_tweets.jsonl"
)
GEMMA_EMOTIONS = (
    ROOT / "results" / "emotions_gemma" / "emotions_gemma4:31b_temp1.0_all_tweets.jsonl"
)
XLM_EMOTIONS = ROOT / "results" / "emotions_first_run" / "np_text_features.csv"

EXPECTED_CODED_UNITS = 4386
EXPECTED_ANALYTIC_TWEETS = 824
EXPECTED_DETAILED_REASON_ASSIGNMENTS = 1266
EXPECTED_PARENT_CATEGORY_ASSIGNMENTS = 1238
WINDOWS_HOURS = [1, 6, 12, 24, 72]
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260612
PERMUTATION_REPS = 10000
PERMUTATION_SEED = 20260624
MAIN_REASON_MIN_TWEETS = 30
PHASE_GROUPS = {
    1: "phase 1",
    2: "phase 2",
    3: "phase 3",
    4: "phases 4-6",
    5: "phases 4-6",
    6: "phases 4-6",
}
EMOTION_LABELS = {
    "bes": "anger",
    "gadjenje": "disgust",
    "ga\u0111enje": "disgust",
    "iscekkivanje": "anticipation",
    "i\u0161\u010dekivanje": "anticipation",
    "iznenadjenje": "surprise",
    "iznena\u0111enje": "surprise",
    "nepoznato": "unknown",
    "neutralno": "neutral",
    "poverenje": "trust",
    "radost": "joy",
    "strah": "fear",
    "tuga": "sadness",
}
MOTIVE_DISPLAY_LABELS = {
    "1": "Discouraged by prior reporting outcomes",
    "2": "Distrust in institutions",
    "3": "Others minimized the experience",
    "4": "Experience treated as normal",
    "5": "Lack of knowledge about violence",
    "6": "Fear",
    "7": "Self-blame",
    "8": "Discouraged by others",
    "9": "Escaped or defended oneself",
    "10": "Gaslighting or self-doubt",
    "11": "Shame or embarrassment",
    "12": "Lack of support",
    "14": "Perpetrator status or power",
    "15": "Invisible violence or lack of evidence",
    "16": "No one would believe",
    "17": "Victim blaming",
    "18": "Perpetrator was family",
    "19": "Family protection",
    "20": "Belief perpetrator would change",
    "22": "Dependence on perpetrator",
    "23": "Belief that one could deal with it",
    "24": "Other",
}
MOTIVE_SUBCODE_LABELS = {
    "3.1": "Others did not believe the victim",
    "3.2": "Others minimized the severity",
    "5.1": "Did not know the experience should be reported",
    "5.2": "Did not recognize the experience as violence",
    "5.3": "Did not know whom to report to",
    "5.4": "Was a child or young person",
    "5.5": "Gendered barriers affecting men",
    "6.1": "Fear of admitting what happened",
    "6.2": "Fear of perpetrator retaliation",
    "6.3": "Fear of others' reactions",
    "12.1": "Lack of support after disclosure",
    "12.2": "No one available to tell",
    "19.1": "Protecting family from shame or pain",
    "19.2": "Protecting family from retaliatory consequences",
}
ANALYTIC_MOTIVE_FAMILIES = set(MOTIVE_DISPLAY_LABELS)
PHASE_GROUP_ORDER = ["phase 1", "phase 2", "phase 3", "phases 4-6"]
EMOTION_GROUP_ORDER = ["sadness", "anger", "fear", "disgust", "Other"]
REASON_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#999999",
]


def ensure_outdir() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)


def read_jsonl_emotions(path: Path, id_key: str, label_name: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            tweet_id = str(obj.get(id_key, "")).strip()
            emotion = str(obj.get("emotion", "")).strip()
            if tweet_id:
                rows.append({"tweet_id": tweet_id, label_name: emotion})
    return pd.DataFrame(rows).drop_duplicates("tweet_id", keep="last")


def assign_phases(
    times: pd.Series, phases: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    labels: list[str] = []
    ids: list[float] = []
    for timestamp in times:
        if pd.isna(timestamp):
            labels.append("")
            ids.append(math.nan)
            continue
        match = phases[
            (phases["start_utc"] <= timestamp) & (timestamp < phases["end_utc"])
        ]
        if match.empty and timestamp == phases["end_utc"].max():
            match = phases[phases["end_utc"] == timestamp]
        if match.empty:
            labels.append("")
            ids.append(math.nan)
        else:
            row = match.iloc[0]
            labels.append(str(row["phase_label"]))
            ids.append(float(row["phase_id"]))
    return pd.Series(labels, index=times.index), pd.Series(ids, index=times.index)


def phase_group(phase_id: object) -> str:
    if pd.isna(phase_id):
        return ""
    try:
        return PHASE_GROUPS.get(int(phase_id), "")
    except (TypeError, ValueError):
        return ""


def display_motive_label(row: pd.Series) -> str:
    family = str(row.get("motive_family", "")).strip()
    return MOTIVE_DISPLAY_LABELS.get(family, str(row.get("motive_label", "")).strip())


def display_emotion_label(value: object) -> str:
    key = str(value or "").strip()
    return EMOTION_LABELS.get(key, key)


def collapsed_emotion_label(value: object) -> str:
    label = display_emotion_label(value)
    return label if label in {"sadness", "anger", "fear", "disgust"} else "Other"


def first_line(value: str) -> str:
    value = str(value or "").strip()
    return value.splitlines()[0].strip() if value else ""


def load_inputs() -> dict[str, pd.DataFrame]:
    coded = pd.read_csv(CODED_TWEETS, dtype=str).fillna("")
    motive_long_all = pd.read_csv(MOTIVE_LONG, dtype=str).fillna("")
    codebook = pd.read_csv(MOTIVE_CODEBOOK, dtype=str).fillna("")
    tweets = pd.read_csv(TWEETS, dtype=str).fillna("")
    phases = pd.read_csv(PHASES, dtype=str).fillna("")
    institutions = pd.read_csv(INSTITUTIONS, dtype=str).fillna("")

    tweets["time_dt"] = pd.to_datetime(tweets["time"], utc=True, errors="coerce")
    phases["phase_id"] = pd.to_numeric(phases["phase_id"], errors="coerce").astype(
        "Int64"
    )
    phases["start_utc"] = pd.to_datetime(phases["start_utc"], utc=True, errors="coerce")
    phases["end_utc"] = pd.to_datetime(phases["end_utc"], utc=True, errors="coerce")

    gpt = read_jsonl_emotions(GPT_EMOTIONS, "id_str", "emotion_gpt")
    gemma = read_jsonl_emotions(GEMMA_EMOTIONS, "id", "emotion_gemma")
    xlm = pd.read_csv(
        XLM_EMOTIONS, dtype=str, usecols=["id_str", "label_xlm_emotions"]
    ).rename(columns={"id_str": "tweet_id", "label_xlm_emotions": "emotion_xlm"})
    xlm = xlm.drop_duplicates("tweet_id", keep="last")

    return {
        "coded": coded,
        "motive_long_all": motive_long_all,
        "codebook": codebook,
        "tweets": tweets,
        "phases": phases,
        "institutions": institutions,
        "gpt": gpt,
        "gemma": gemma,
        "xlm": xlm,
    }


def build_codebook_lookup(codebook: pd.DataFrame) -> pd.DataFrame:
    lookup = codebook.copy()
    lookup["motive_label"] = lookup["name_with_subcodes"].map(first_line)
    return lookup[
        ["motive_family", "code", "motive_label", "name_with_subcodes", "frequency"]
    ]


def collapse_to_parent_categories(
    detailed_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Return one auditable row per tweet and parent motive category."""
    parent_labels = (
        detailed_labels.groupby(
            ["tweet_id", "motive_family", "motive_label"],
            as_index=False,
            sort=False,
        )
        .agg(
            resolved_tweet_id=("resolved_tweet_id", "first"),
            workbook_tweet_id=("workbook_tweet_id", "first"),
            workbook_row=("workbook_row", "first"),
            nonreporting_motives_raw=("nonreporting_motives_raw", "first"),
            motive_codes=(
                "motive_code",
                lambda values: ";".join(sorted(set(values))),
            ),
            detailed_assignment_n=("motive_code", "size"),
        )
    )
    return parent_labels.sort_values(["tweet_id", "motive_family"]).reset_index(
        drop=True
    )


def validate_inputs(
    coded: pd.DataFrame, tweets: pd.DataFrame, phases: pd.DataFrame
) -> None:
    if len(coded) != EXPECTED_CODED_UNITS:
        raise ValueError(
            f"Expected {EXPECTED_CODED_UNITS} coded units, found {len(coded)}"
        )
    coded_ids = set(coded["tweet_id"])
    tweet_ids = set(tweets["id_str"])
    missing = coded_ids - tweet_ids
    if missing:
        sample = ", ".join(sorted(list(missing))[:5])
        raise ValueError(
            f"{len(missing)} coded tweet IDs are missing from {TWEETS}: {sample}"
        )
    resolved_missing = set(coded["resolved_tweet_id"]) - tweet_ids
    if resolved_missing:
        sample = ", ".join(sorted(resolved_missing)[:5])
        raise ValueError(
            f"{len(resolved_missing)} resolved IDs are missing from {TWEETS}: {sample}"
        )
    if phases["start_utc"].isna().any() or phases["end_utc"].isna().any():
        raise ValueError("Phase boundaries contain unparsable timestamps")


def build_tables(inputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    coded = inputs["coded"].copy()
    motive_long_all = inputs["motive_long_all"].copy()
    codebook = inputs["codebook"].copy()
    tweets = inputs["tweets"].copy()
    phases = inputs["phases"].copy()
    institutions = inputs["institutions"].copy()

    validate_inputs(coded, tweets, phases)

    code_lookup = build_codebook_lookup(codebook)
    motive_long_all = motive_long_all.merge(
        code_lookup[["motive_family", "motive_label"]].drop_duplicates("motive_family"),
        on="motive_family",
        how="left",
    )
    x_or_zero_ids = set(
        motive_long_all.loc[
            motive_long_all["motive_family"].isin({"x", "0"}), "tweet_id"
        ]
    )
    motive_long_all["is_analytic_motive"] = motive_long_all["motive_family"].isin(
        ANALYTIC_MOTIVE_FAMILIES
    ) & ~motive_long_all["tweet_id"].isin(x_or_zero_ids)
    detailed_label_table = motive_long_all[
        motive_long_all["is_analytic_motive"]
    ].copy()
    observed_counts = (
        detailed_label_table["tweet_id"].nunique(),
        len(detailed_label_table),
    )
    expected_counts = (
        EXPECTED_ANALYTIC_TWEETS,
        EXPECTED_DETAILED_REASON_ASSIGNMENTS,
    )
    if observed_counts != expected_counts:
        raise ValueError(
            "Expected reconciled analytic counts "
            f"{expected_counts}, found {observed_counts}"
        )
    label_table = collapse_to_parent_categories(detailed_label_table)
    parent_counts = (label_table["tweet_id"].nunique(), len(label_table))
    expected_parent_counts = (
        EXPECTED_ANALYTIC_TWEETS,
        EXPECTED_PARENT_CATEGORY_ASSIGNMENTS,
    )
    if parent_counts != expected_parent_counts:
        raise ValueError(
            "Expected deduplicated parent-category counts "
            f"{expected_parent_counts}, found {parent_counts}"
        )

    coded_items = tweets[
        [
            "id_str",
            "from_user",
            "from_user_id_str",
            "time",
            "time_dt",
            "post_type",
            "parent_id_str",
        ]
    ].rename(
        columns={
            "id_str": "tweet_id",
            "from_user": "author_username",
            "from_user_id_str": "author_user_id",
            "time": "post_time",
            "time_dt": "post_time_dt",
            "post_type": "coded_post_type",
            "parent_id_str": "coded_parent_id",
        }
    )
    resolved_items = tweets[["id_str", "time", "time_dt", "post_type"]].rename(
        columns={
            "id_str": "resolved_tweet_id",
            "time": "resolved_post_time",
            "time_dt": "resolved_post_time_dt",
            "post_type": "resolved_post_type",
        }
    )
    retweets = tweets[tweets["post_type"] == "retweet"][
        ["id_str", "from_user", "from_user_id_str", "time", "time_dt", "parent_id_str"]
    ].rename(
        columns={
            "id_str": "retweet_id",
            "from_user": "retweeter_username",
            "from_user_id_str": "retweeter_user_id",
            "time": "retweet_time",
            "time_dt": "retweet_time_dt",
            "parent_id_str": "tweet_id",
        }
    )

    post_table = coded.merge(coded_items, on="tweet_id", how="left")
    post_table = post_table.merge(
        resolved_items.drop_duplicates("resolved_tweet_id"),
        on="resolved_tweet_id",
        how="left",
    )
    post_table["post_phase"], post_table["post_phase_id"] = assign_phases(
        post_table["resolved_post_time_dt"], phases
    )
    post_table["post_phase_group"] = post_table["post_phase_id"].map(phase_group)

    retweets_coded = retweets[retweets["tweet_id"].isin(post_table["tweet_id"])].copy()
    retweets_coded["retweet_phase"], retweets_coded["retweet_phase_id"] = assign_phases(
        retweets_coded["retweet_time_dt"], phases
    )
    retweets_coded["retweet_phase_group"] = retweets_coded["retweet_phase_id"].map(
        phase_group
    )

    institutional_usernames = set(
        institutions.loc[institutions["in_dataset"].str.lower().eq("yes"), "username"]
    )
    post_table["author_is_institutional"] = post_table["author_username"].isin(
        institutional_usernames
    )
    retweets_coded["retweeter_is_institutional"] = retweets_coded[
        "retweeter_username"
    ].isin(institutional_usernames)
    retweets_coded = retweets_coded.merge(
        institutions[["username", "category", "description"]].rename(
            columns={
                "username": "retweeter_username",
                "category": "retweeter_institution_category",
                "description": "retweeter_institution_description",
            }
        ),
        on="retweeter_username",
        how="left",
    )

    post_table = post_table.merge(inputs["gpt"], on="tweet_id", how="left")
    post_table = post_table.merge(inputs["gemma"], on="tweet_id", how="left")
    post_table = post_table.merge(inputs["xlm"], on="tweet_id", how="left")

    retweet_counts = retweets_coded.groupby("tweet_id").size().rename("total_retweets")
    post_table = post_table.merge(retweet_counts, on="tweet_id", how="left")
    post_table["total_retweets"] = post_table["total_retweets"].fillna(0).astype(int)

    for hours in WINDOWS_HOURS:
        cutoff = post_table[["tweet_id", "post_time_dt"]].copy()
        tmp = retweets_coded.merge(cutoff, on="tweet_id", how="left")
        tmp[f"retweets_within_{hours}h"] = (
            tmp["retweet_time_dt"] >= tmp["post_time_dt"]
        ) & (tmp["retweet_time_dt"] <= tmp["post_time_dt"] + pd.Timedelta(hours=hours))
        counts = (
            tmp[tmp[f"retweets_within_{hours}h"]]
            .groupby("tweet_id")
            .size()
            .rename(f"retweets_within_{hours}h")
        )
        post_table = post_table.merge(counts, on="tweet_id", how="left")
        post_table[f"retweets_within_{hours}h"] = (
            post_table[f"retweets_within_{hours}h"].fillna(0).astype(int)
        )

    survival = retweets_coded.groupby("tweet_id")["retweet_time_dt"].agg(
        first_retweet_time="min", last_retweet_time="max"
    )
    post_table = post_table.merge(survival, on="tweet_id", how="left")
    post_table["time_to_first_retweet_hours"] = (
        post_table["first_retweet_time"] - post_table["post_time_dt"]
    ).dt.total_seconds() / 3600
    post_table["retweet_survival_span_hours"] = (
        post_table["last_retweet_time"] - post_table["post_time_dt"]
    ).dt.total_seconds() / 3600
    post_table["has_late_retweet_after_12h"] = (
        post_table["retweet_survival_span_hours"] > 12
    )
    post_table["has_late_retweet_after_24h"] = (
        post_table["retweet_survival_span_hours"] > 24
    )

    detailed_counts_per_tweet = (
        detailed_label_table.groupby("tweet_id")
        .size()
        .rename("n_detailed_reason_codes")
    )
    parent_counts_per_tweet = (
        label_table.groupby("tweet_id").size().rename("n_parent_categories")
    )
    post_table = post_table.merge(
        detailed_counts_per_tweet, on="tweet_id", how="left"
    )
    post_table = post_table.merge(parent_counts_per_tweet, on="tweet_id", how="left")
    post_table["n_detailed_reason_codes"] = (
        post_table["n_detailed_reason_codes"].fillna(0).astype(int)
    )
    post_table["n_parent_categories"] = (
        post_table["n_parent_categories"].fillna(0).astype(int)
    )

    metadata_columns = [
        "tweet_id",
        "post_time",
        "post_phase",
        "post_phase_id",
        "post_phase_group",
        "coded_post_type",
        "author_username",
        "total_retweets",
        "emotion_gpt",
        "emotion_gemma",
        "emotion_xlm",
        "n_detailed_reason_codes",
        "n_parent_categories",
    ]

    detailed_label_table = detailed_label_table.merge(
        post_table[metadata_columns],
        on="tweet_id",
        how="left",
    )

    label_table = label_table.merge(
        post_table[metadata_columns],
        on="tweet_id",
        how="left",
    )
    label_table["fractional_tweet_weight"] = np.where(
        label_table["n_parent_categories"] > 0,
        1 / label_table["n_parent_categories"],
        0.0,
    )
    label_table["fractional_retweet_weight"] = (
        label_table["total_retweets"] * label_table["fractional_tweet_weight"]
    )

    retweet_label_table = retweets_coded.merge(
        label_table[
            [
                "tweet_id",
                "motive_family",
                "motive_label",
                "motive_codes",
                "detailed_assignment_n",
                "n_parent_categories",
                "emotion_gemma",
                "emotion_gpt",
                "emotion_xlm",
            ]
        ],
        on="tweet_id",
        how="inner",
    )
    retweet_label_table["fractional_retweet_event_weight"] = np.where(
        retweet_label_table["n_parent_categories"] > 0,
        1 / retweet_label_table["n_parent_categories"],
        0.0,
    )
    retweet_label_table["emotion_gemma_display"] = retweet_label_table[
        "emotion_gemma"
    ].map(display_emotion_label)
    retweet_label_table["emotion_gemma_collapsed"] = retweet_label_table[
        "emotion_gemma"
    ].map(collapsed_emotion_label)

    retweet_events = retweets_coded.merge(
        label_table.groupby("tweet_id")["motive_family"]
        .apply(lambda values: ";".join(sorted(set(values))))
        .rename("motive_families"),
        on="tweet_id",
        how="left",
    )

    return {
        "code_lookup": code_lookup,
        "post_table": post_table,
        "label_table": label_table,
        "detailed_label_table": detailed_label_table,
        "label_table_all_codes": motive_long_all,
        "retweet_events": retweet_events,
        "retweet_label_table": retweet_label_table,
        "institutions": institutions,
    }


def iqr(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return ""
    return f"{clean.quantile(0.25):.2f}-{clean.quantile(0.75):.2f}"


def bootstrap_ci(values: np.ndarray, statistic=np.mean) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + len(values))
    stats = [
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(BOOTSTRAP_REPS)
    ]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def fdr_bh(p_values: list[float]) -> list[float]:
    p_array = np.asarray(
        [1.0 if pd.isna(p) else float(p) for p in p_values], dtype=float
    )
    order = np.argsort(p_array)
    q_values = np.empty_like(p_array)
    previous = 1.0
    total = len(p_array)
    for rank, index in reversed(list(enumerate(order, start=1))):
        current = min(previous, p_array[index] * total / rank)
        q_values[index] = current
        previous = current
    return q_values.tolist()


def normalize_shares(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def js_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = normalize_shares(left)
    right = normalize_shares(right)
    mask = (left > 0) | (right > 0)
    left = left[mask]
    right = right[mask]
    if left.size == 0:
        return 0.0
    midpoint = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    kl_left = np.sum(left[left_mask] * np.log2(left[left_mask] / midpoint[left_mask]))
    kl_right = np.sum(
        right[right_mask] * np.log2(right[right_mask] / midpoint[right_mask])
    )
    return float(math.sqrt(0.5 * (kl_left + kl_right)))


def reason_weight_matrix(
    label_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = (
        label_table[
            [
                "tweet_id",
                "post_phase_group",
                "total_retweets",
                "motive_family",
                "fractional_tweet_weight",
            ]
        ]
        .copy()
        .sort_values(["tweet_id", "motive_family"])
    )
    matrix = rows.pivot_table(
        index="tweet_id",
        columns="motive_family",
        values="fractional_tweet_weight",
        aggfunc="sum",
        fill_value=0.0,
    )
    metadata = (
        label_table[
            [
                "tweet_id",
                "post_phase_group",
                "total_retweets",
                "n_parent_categories",
            ]
        ]
        .drop_duplicates("tweet_id")
        .set_index("tweet_id")
        .loc[matrix.index]
    )
    return matrix, metadata


def reason_visibility_fractional(
    label_table: pd.DataFrame, label_distribution: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, metadata = reason_weight_matrix(label_table)
    weights = matrix.to_numpy(dtype=float)
    retweets = metadata["total_retweets"].to_numpy(dtype=float)
    families = list(matrix.columns)

    source_counts = weights.sum(axis=0)
    retweet_counts = (weights * retweets[:, None]).sum(axis=0)
    source_shares = normalize_shares(source_counts)
    retweet_shares = normalize_shares(retweet_counts)
    epsilon = 1e-12
    observed_log2 = np.log2((retweet_shares + epsilon) / (source_shares + epsilon))
    observed_pp = retweet_shares - source_shares
    observed_js = js_distance(source_shares, retweet_shares)

    rng = np.random.default_rng(PERMUTATION_SEED)
    group_indexes = [
        np.flatnonzero(metadata["post_phase_group"].to_numpy() == group)
        for group in PHASE_GROUP_ORDER
    ]
    perm_js = np.zeros(PERMUTATION_REPS)
    perm_log2 = np.zeros((PERMUTATION_REPS, len(families)))
    for rep in range(PERMUTATION_REPS):
        shuffled = retweets.copy()
        for indexes in group_indexes:
            if len(indexes) > 1:
                shuffled[indexes] = rng.permutation(shuffled[indexes])
        null_counts = (weights * shuffled[:, None]).sum(axis=0)
        null_shares = normalize_shares(null_counts)
        perm_js[rep] = js_distance(source_shares, null_shares)
        perm_log2[rep] = np.log2((null_shares + epsilon) / (source_shares + epsilon))

    bootstrap_log2 = np.zeros((BOOTSTRAP_REPS, len(families)))
    bootstrap_pp = np.zeros((BOOTSTRAP_REPS, len(families)))
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    row_indexes = np.arange(len(matrix))
    for rep in range(BOOTSTRAP_REPS):
        sample = bootstrap_rng.choice(row_indexes, size=len(row_indexes), replace=True)
        sample_source = weights[sample].sum(axis=0)
        sample_retweets = retweets[sample]
        sample_retweet_counts = (weights[sample] * sample_retweets[:, None]).sum(axis=0)
        sample_source_shares = normalize_shares(sample_source)
        sample_retweet_shares = normalize_shares(sample_retweet_counts)
        bootstrap_log2[rep] = np.log2(
            (sample_retweet_shares + epsilon) / (sample_source_shares + epsilon)
        )
        bootstrap_pp[rep] = sample_retweet_shares - sample_source_shares

    distribution = label_distribution.set_index("motive_family")
    rows = []
    p_values = []
    for col, family in enumerate(families):
        row = distribution.loc[family]
        eligible = int(row["unique_tweets"]) >= MAIN_REASON_MIN_TWEETS
        if eligible:
            p_value = (
                1 + np.sum(np.abs(perm_log2[:, col]) >= abs(observed_log2[col]))
            ) / (PERMUTATION_REPS + 1)
        else:
            p_value = math.nan
        p_values.append(p_value)
        rows.append(
            {
                "motive_family": family,
                "display_label": row["display_label"],
                "unique_tweets": int(row["unique_tweets"]),
                "label_assignments": int(row["label_assignments"]),
                "source_fractional_count": source_counts[col],
                "retweet_fractional_count": retweet_counts[col],
                "source_share": source_shares[col],
                "retweet_share": retweet_shares[col],
                "percentage_point_change": observed_pp[col],
                "log2_multiplier": observed_log2[col],
                "amplification_multiplier": (retweet_shares[col] + epsilon)
                / (source_shares[col] + epsilon),
                "log2_ci_low": np.percentile(bootstrap_log2[:, col], 2.5),
                "log2_ci_high": np.percentile(bootstrap_log2[:, col], 97.5),
                "percentage_point_ci_low": np.percentile(bootstrap_pp[:, col], 2.5),
                "percentage_point_ci_high": np.percentile(bootstrap_pp[:, col], 97.5),
                "permutation_p": p_value,
                "eligible_main": eligible,
                "global_js_observed": observed_js,
                "global_permutation_p": (1 + np.sum(perm_js >= observed_js))
                / (PERMUTATION_REPS + 1),
            }
        )
    result = pd.DataFrame(rows)
    eligible_mask = result["eligible_main"]
    result["fdr_q"] = math.nan
    if eligible_mask.any():
        result.loc[eligible_mask, "fdr_q"] = fdr_bh(
            result.loc[eligible_mask, "permutation_p"].tolist()
        )
    result = result.sort_values("log2_multiplier", ascending=False)
    global_test = pd.DataFrame(
        [
            {
                "test": "phase-stratified permutation",
                "unit": "source tweet",
                "reason_tweets": int(len(matrix)),
                "permutations": PERMUTATION_REPS,
                "statistic": "Jensen-Shannon distance",
                "observed": observed_js,
                "p_value": (1 + np.sum(perm_js >= observed_js))
                / (PERMUTATION_REPS + 1),
            }
        ]
    )
    return result, global_test


def reason_phase_retweet_composition(retweet_label: pd.DataFrame) -> pd.DataFrame:
    reason_totals = (
        retweet_label.groupby("motive_family")["fractional_retweet_event_weight"]
        .sum()
        .sort_values(ascending=False)
    )
    top_families = set(reason_totals.head(6).index)
    rows = retweet_label.copy()
    rows["reason_group"] = np.where(
        rows["motive_family"].isin(top_families), rows["motive_family"], "Other"
    )
    display_lookup = (
        rows.drop_duplicates("motive_family")
        .assign(display_label=lambda df: df.apply(display_motive_label, axis=1))
        .set_index("motive_family")["display_label"]
        .to_dict()
    )
    display_lookup["Other"] = "Other"
    summary = (
        rows.groupby(["retweet_phase_group", "reason_group"], dropna=False)
        .agg(fractional_retweet_count=("fractional_retweet_event_weight", "sum"))
        .reset_index()
    )
    phase_totals = (
        rows.groupby("retweet_phase_group")
        .agg(
            phase_total_fractional=("fractional_retweet_event_weight", "sum"),
            phase_unique_retweets=("retweet_id", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(phase_totals, on="retweet_phase_group", how="left")
    summary["phase_share"] = (
        summary["fractional_retweet_count"] / summary["phase_total_fractional"]
    )
    summary["display_label"] = summary["reason_group"].map(display_lookup)
    summary["phase_order"] = summary["retweet_phase_group"].map(
        {value: index for index, value in enumerate(PHASE_GROUP_ORDER)}
    )
    return summary.sort_values(
        ["phase_order", "fractional_retweet_count"], ascending=[True, False]
    )


def reason_emotion_retweet_distribution(retweet_label: pd.DataFrame) -> pd.DataFrame:
    result = (
        retweet_label.groupby(
            ["motive_family", "motive_label", "emotion_gemma_collapsed"], dropna=False
        )
        .agg(
            fractional_retweet_count=("fractional_retweet_event_weight", "sum"),
            unique_retweets=("retweet_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"emotion_gemma_collapsed": "emotion_display"})
    )
    result["reason_total_fractional"] = result.groupby("motive_family")[
        "fractional_retweet_count"
    ].transform("sum")
    result["reason_emotion_retweet_share"] = (
        result["fractional_retweet_count"] / result["reason_total_fractional"]
    )
    result["display_label"] = result.apply(display_motive_label, axis=1)
    return result.sort_values(
        ["reason_total_fractional", "fractional_retweet_count"], ascending=False
    )


def supplementary_reason_aggregate(
    label_table: pd.DataFrame, detailed_label_table: pd.DataFrame
) -> pd.DataFrame:
    """Build Supplementary Table S3 at parent and defined-subcode levels."""
    rows: list[dict[str, object]] = []
    for parent_code, parent_label in MOTIVE_DISPLAY_LABELS.items():
        parent = label_table[label_table["motive_family"].eq(parent_code)]
        rows.append(
            {
                "row_type": "parent",
                "parent_code": parent_code,
                "code": parent_code,
                "label": parent_label,
                "assignment_n": len(parent),
                "assignment_unit": "parent category",
                "assignment_denominator": EXPECTED_PARENT_CATEGORY_ASSIGNMENTS,
                "unique_tweets_n": parent["tweet_id"].nunique(),
            }
        )
        for code, label in MOTIVE_SUBCODE_LABELS.items():
            if code.split(".", 1)[0] != parent_code:
                continue
            subcode = detailed_label_table[
                detailed_label_table["motive_code"].eq(code)
            ]
            rows.append(
                {
                    "row_type": "subcode",
                    "parent_code": parent_code,
                    "code": code,
                    "label": label,
                    "assignment_n": len(subcode),
                    "assignment_unit": "detailed code",
                    "assignment_denominator": EXPECTED_DETAILED_REASON_ASSIGNMENTS,
                    "unique_tweets_n": subcode["tweet_id"].nunique(),
                }
            )
    result = pd.DataFrame(rows)
    result["assignment_pct"] = (
        result["assignment_n"] / result["assignment_denominator"]
    )
    result["unique_tweets_denominator"] = EXPECTED_ANALYTIC_TWEETS
    result["unique_tweets_pct"] = result["unique_tweets_n"] / EXPECTED_ANALYTIC_TWEETS
    parent_total = result.loc[result["row_type"].eq("parent"), "assignment_n"].sum()
    if parent_total != EXPECTED_PARENT_CATEGORY_ASSIGNMENTS:
        raise ValueError(f"Supplementary Table S3 parent total is {parent_total}")
    return result[
        [
            "row_type",
            "parent_code",
            "code",
            "label",
            "assignment_n",
            "assignment_unit",
            "assignment_denominator",
            "assignment_pct",
            "unique_tweets_n",
            "unique_tweets_denominator",
            "unique_tweets_pct",
        ]
    ]


def reason_exclusion_summary(
    coded: pd.DataFrame, label_all: pd.DataFrame
) -> pd.DataFrame:
    """Report mutually interpretable exclusions from the analytic reason layer."""
    families = label_all["motive_family"].astype(str).str.casefold()
    analytic = ANALYTIC_MOTIVE_FAMILIES | {"x", "0", "?"}
    definitions = [
        ("x", families.eq("x"), "Another person's experience"),
        ("0", families.eq("0"), "No reason stated"),
        ("unresolved", families.eq("?"), "Unresolved code"),
        ("unlisted", ~families.isin(analytic), "Code absent from final codebook"),
    ]
    rows: list[dict[str, object]] = []
    for exclusion_type, mask, definition in definitions:
        subset = label_all[mask]
        rows.append(
            {
                "exclusion_type": exclusion_type,
                "codes": ";".join(sorted(set(subset["motive_code"]))),
                "coding_units_n": subset["tweet_id"].nunique(),
                "parsed_assignments_n": len(subset),
                "definition": definition,
            }
        )
    rows.append(
        {
            "exclusion_type": "blank",
            "codes": "",
            "coding_units_n": int(coded["nonreporting_motives_raw"].eq("").sum()),
            "parsed_assignments_n": 0,
            "definition": "Blank reason field",
        }
    )
    return pd.DataFrame(rows)


def summarize(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    post_table = tables["post_table"]
    label_table = tables["label_table"]
    detailed_label_table = tables["detailed_label_table"]
    label_all = tables["label_table_all_codes"]
    retweet_label = tables["retweet_label_table"]

    total_label_assignments = len(label_table)
    total_label_tweets = label_table["tweet_id"].nunique()
    label_distribution = (
        label_table.groupby(["motive_family", "motive_label"], dropna=False)
        .agg(
            label_assignments=("tweet_id", "size"),
            unique_tweets=("tweet_id", "nunique"),
            full_count_retweets=("total_retweets", "sum"),
            fractional_tweet_count=("fractional_tweet_weight", "sum"),
            fractional_retweet_count=("fractional_retweet_weight", "sum"),
        )
        .reset_index()
    )
    label_distribution["assignment_share"] = (
        label_distribution["label_assignments"] / total_label_assignments
    )
    label_distribution["tweet_share"] = (
        label_distribution["unique_tweets"] / total_label_tweets
    )
    label_distribution = label_distribution.sort_values(
        ["label_assignments", "unique_tweets"], ascending=False
    )
    label_distribution["display_label"] = label_distribution.apply(
        display_motive_label, axis=1
    )

    labels_per_tweet = (
        post_table.loc[
            post_table["n_parent_categories"] > 0,
            ["tweet_id", "n_parent_categories"],
        ]
        .drop_duplicates()
        .copy()
    )
    labels_per_tweet["label_count_bucket"] = np.select(
        [
            labels_per_tweet["n_parent_categories"] == 1,
            labels_per_tweet["n_parent_categories"] == 2,
            labels_per_tweet["n_parent_categories"] >= 3,
        ],
        ["1", "2", "3+"],
        default="0",
    )
    multilabel_distribution = (
        labels_per_tweet.groupby("label_count_bucket")
        .agg(tweets=("tweet_id", "nunique"))
        .reset_index()
    )
    multilabel_distribution["share"] = (
        multilabel_distribution["tweets"] / multilabel_distribution["tweets"].sum()
    )

    label_retweet_distribution = (
        label_table.groupby(["motive_family", "motive_label"], dropna=False)
        .agg(
            labeled_tweets=("tweet_id", "nunique"),
            total_retweets=("total_retweets", "sum"),
            retweeted_tweets=("total_retweets", lambda s: int((s > 0).sum())),
            median_retweets=("total_retweets", "median"),
            mean_retweets=("total_retweets", "mean"),
            max_retweets=("total_retweets", "max"),
            iqr_retweets=("total_retweets", iqr),
            fractional_retweets=("fractional_retweet_weight", "sum"),
        )
        .reset_index()
    )
    label_retweet_distribution["share_retweeted"] = (
        label_retweet_distribution["retweeted_tweets"]
        / label_retweet_distribution["labeled_tweets"]
    )
    label_retweet_distribution = label_retweet_distribution.sort_values(
        "total_retweets", ascending=False
    )
    label_retweet_distribution["display_label"] = label_retweet_distribution.apply(
        display_motive_label, axis=1
    )

    label_by_post_phase = (
        label_table.groupby(
            ["post_phase_id", "post_phase", "motive_family", "motive_label"]
        )
        .agg(
            label_assignments=("tweet_id", "size"),
            unique_tweets=("tweet_id", "nunique"),
        )
        .reset_index()
    )
    label_by_post_phase["phase_total_assignments"] = label_by_post_phase.groupby(
        "post_phase"
    )["label_assignments"].transform("sum")
    label_by_post_phase["phase_share"] = (
        label_by_post_phase["label_assignments"]
        / label_by_post_phase["phase_total_assignments"]
    )
    label_by_post_phase["display_label"] = label_by_post_phase.apply(
        display_motive_label, axis=1
    )

    reason_phase_distribution = (
        label_table.groupby(
            [
                "post_phase_id",
                "post_phase",
                "post_phase_group",
                "coded_post_type",
                "motive_family",
                "motive_label",
            ],
            dropna=False,
        )
        .agg(
            label_assignments=("tweet_id", "size"),
            unique_tweets=("tweet_id", "nunique"),
        )
        .reset_index()
    )
    reason_phase_distribution["phase_total_assignments"] = (
        reason_phase_distribution.groupby("post_phase")["label_assignments"].transform(
            "sum"
        )
    )
    reason_phase_distribution["phase_assignment_share"] = (
        reason_phase_distribution["label_assignments"]
        / reason_phase_distribution["phase_total_assignments"]
    )
    reason_phase_distribution["display_label"] = reason_phase_distribution.apply(
        display_motive_label, axis=1
    )

    reason_phase_group_distribution = (
        label_table.groupby(
            ["post_phase_group", "motive_family", "motive_label"], dropna=False
        )
        .agg(
            label_assignments=("tweet_id", "size"),
            unique_tweets=("tweet_id", "nunique"),
        )
        .reset_index()
    )
    reason_phase_group_distribution["group_total_assignments"] = (
        reason_phase_group_distribution.groupby("post_phase_group")[
            "label_assignments"
        ].transform("sum")
    )
    reason_phase_group_distribution["group_assignment_share"] = (
        reason_phase_group_distribution["label_assignments"]
        / reason_phase_group_distribution["group_total_assignments"]
    )
    reason_phase_group_distribution["display_label"] = (
        reason_phase_group_distribution.apply(display_motive_label, axis=1)
    )

    label_retweets_by_phase = (
        retweet_label.groupby(
            ["retweet_phase_id", "retweet_phase", "motive_family", "motive_label"]
        )
        .agg(
            retweet_label_events=("retweet_id", "size"),
            unique_retweets=("retweet_id", "nunique"),
        )
        .reset_index()
    )
    if not label_retweets_by_phase.empty:
        label_retweets_by_phase["phase_total_events"] = label_retweets_by_phase.groupby(
            "retweet_phase"
        )["retweet_label_events"].transform("sum")
        label_retweets_by_phase["phase_share"] = (
            label_retweets_by_phase["retweet_label_events"]
            / label_retweets_by_phase["phase_total_events"]
        )
    label_retweets_by_phase["display_label"] = label_retweets_by_phase.apply(
        display_motive_label, axis=1
    )

    reason_amplification_by_phase = (
        retweet_label.groupby(
            [
                "retweet_phase_id",
                "retweet_phase",
                "retweet_phase_group",
                "motive_family",
                "motive_label",
            ],
            dropna=False,
        )
        .agg(
            retweet_label_events=("retweet_id", "size"),
            unique_retweets=("retweet_id", "nunique"),
        )
        .reset_index()
    )
    if not reason_amplification_by_phase.empty:
        reason_amplification_by_phase["phase_total_events"] = (
            reason_amplification_by_phase.groupby("retweet_phase")[
                "retweet_label_events"
            ].transform("sum")
        )
        reason_amplification_by_phase["phase_share"] = (
            reason_amplification_by_phase["retweet_label_events"]
            / reason_amplification_by_phase["phase_total_events"]
        )
    reason_amplification_by_phase["display_label"] = (
        reason_amplification_by_phase.apply(display_motive_label, axis=1)
    )

    total_posts_with_labels = label_table["tweet_id"].nunique()
    total_retweets_with_labels = label_table.drop_duplicates("tweet_id")[
        "total_retweets"
    ].sum()
    amplification = label_distribution[
        ["motive_family", "motive_label", "unique_tweets", "full_count_retweets"]
    ].copy()
    amplification["post_share"] = (
        amplification["unique_tweets"] / total_posts_with_labels
    )
    amplification["retweet_share"] = (
        amplification["full_count_retweets"] / total_retweets_with_labels
    )
    amplification["amplification_multiplier"] = (
        amplification["retweet_share"] / amplification["post_share"]
    )
    amplification = amplification.sort_values(
        "amplification_multiplier", ascending=False
    )
    amplification["display_label"] = amplification.apply(display_motive_label, axis=1)
    visibility_fractional, visibility_global_test = reason_visibility_fractional(
        label_table, label_distribution
    )
    phase_retweet_composition = reason_phase_retweet_composition(retweet_label)

    survival_rows = []
    for (family, label), group in label_table.groupby(
        ["motive_family", "motive_label"]
    ):
        tweets = post_table[post_table["tweet_id"].isin(group["tweet_id"])]
        total_retweets = tweets["total_retweets"].astype(float).to_numpy()
        span = tweets["retweet_survival_span_hours"].fillna(0).astype(float).to_numpy()
        retweet_lo, retweet_hi = bootstrap_ci(total_retweets, np.mean)
        span_lo, span_hi = bootstrap_ci(span, np.mean)
        survival_rows.append(
            {
                "motive_family": family,
                "motive_label": label,
                "tweets": int(tweets["tweet_id"].nunique()),
                "mean_total_retweets": float(np.mean(total_retweets))
                if len(total_retweets)
                else 0,
                "mean_total_retweets_ci_low": retweet_lo,
                "mean_total_retweets_ci_high": retweet_hi,
                "median_total_retweets": float(np.median(total_retweets))
                if len(total_retweets)
                else 0,
                "share_retweeted": float(np.mean(total_retweets > 0))
                if len(total_retweets)
                else 0,
                "mean_survival_span_hours": float(np.mean(span)) if len(span) else 0,
                "mean_survival_span_ci_low": span_lo,
                "mean_survival_span_ci_high": span_hi,
                "median_survival_span_hours": float(np.median(span))
                if len(span)
                else 0,
                "share_late_after_12h": float(
                    tweets["has_late_retweet_after_12h"].mean()
                ),
                "share_late_after_24h": float(
                    tweets["has_late_retweet_after_24h"].mean()
                ),
            }
        )
    label_retweet_survival = pd.DataFrame(survival_rows).sort_values(
        "mean_survival_span_hours", ascending=False
    )
    label_retweet_survival["display_label"] = label_retweet_survival.apply(
        display_motive_label, axis=1
    )

    institutional = retweet_label[retweet_label["retweeter_is_institutional"]].copy()

    def reason_emotion_summary(emotion_col: str) -> pd.DataFrame:
        result = (
            label_table.groupby(
                ["motive_family", "motive_label", emotion_col], dropna=False
            )
            .agg(tweets=("tweet_id", "nunique"))
            .reset_index()
            .rename(columns={emotion_col: "emotion"})
        )
        result["reason_total"] = result.groupby("motive_family")["tweets"].transform(
            "sum"
        )
        result["reason_emotion_share"] = result["tweets"] / result["reason_total"]
        result["display_label"] = result.apply(display_motive_label, axis=1)
        result["emotion_display"] = result["emotion"].map(display_emotion_label)
        return result

    def reason_emotion_phase_summary(emotion_col: str) -> pd.DataFrame:
        result = (
            label_table.groupby(
                [
                    "post_phase_id",
                    "post_phase",
                    "post_phase_group",
                    "motive_family",
                    "motive_label",
                    emotion_col,
                ],
                dropna=False,
            )
            .agg(tweets=("tweet_id", "nunique"))
            .reset_index()
            .rename(columns={emotion_col: "emotion"})
        )
        result["phase_reason_total"] = result.groupby(["post_phase", "motive_family"])[
            "tweets"
        ].transform("sum")
        result["phase_reason_emotion_share"] = (
            result["tweets"] / result["phase_reason_total"]
        )
        result["display_label"] = result.apply(display_motive_label, axis=1)
        result["emotion_display"] = result["emotion"].map(display_emotion_label)
        return result

    reason_emotion_crosstab_gemma = reason_emotion_summary("emotion_gemma")
    reason_emotion_crosstab_gpt = reason_emotion_summary("emotion_gpt")
    reason_emotion_crosstab_xlm = reason_emotion_summary("emotion_xlm")
    reason_emotion_by_phase_gemma = reason_emotion_phase_summary("emotion_gemma")
    reason_emotion_by_phase_gpt = reason_emotion_phase_summary("emotion_gpt")
    reason_emotion_by_phase_xlm = reason_emotion_phase_summary("emotion_xlm")
    reason_emotion_retweets = reason_emotion_retweet_distribution(retweet_label)
    supplementary_table_s3 = supplementary_reason_aggregate(
        label_table, detailed_label_table
    )
    exclusions = reason_exclusion_summary(post_table, label_all)

    qc_special_codes = label_all[~label_all["is_analytic_motive"]].copy()
    qc_code_counts = (
        label_all.groupby(
            ["motive_family", "motive_code", "motive_label", "is_analytic_motive"]
        )
        .agg(rows=("tweet_id", "size"), unique_tweets=("tweet_id", "nunique"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    return {
        "label_distribution": label_distribution,
        "multilabel_distribution": multilabel_distribution,
        "label_retweet_distribution": label_retweet_distribution,
        "label_by_post_phase": label_by_post_phase,
        "label_retweets_by_phase": label_retweets_by_phase,
        "reason_phase_distribution": reason_phase_distribution,
        "reason_phase_group_distribution": reason_phase_group_distribution,
        "reason_amplification_by_phase": reason_amplification_by_phase,
        "label_amplification_multipliers": amplification,
        "reason_visibility_fractional": visibility_fractional,
        "reason_visibility_global_test": visibility_global_test,
        "reason_phase_retweet_composition": phase_retweet_composition,
        "label_retweet_survival": label_retweet_survival,
        "institutional_retweet_events": institutional,
        "reason_emotion_crosstab": reason_emotion_crosstab_gemma,
        "reason_emotion_by_phase": reason_emotion_by_phase_gemma,
        "reason_emotion_retweet_distribution": reason_emotion_retweets,
        "reason_emotion_crosstab_gemma": reason_emotion_crosstab_gemma,
        "reason_emotion_crosstab_gpt": reason_emotion_crosstab_gpt,
        "reason_emotion_crosstab_xlm": reason_emotion_crosstab_xlm,
        "reason_emotion_by_phase_gemma": reason_emotion_by_phase_gemma,
        "reason_emotion_by_phase_gpt": reason_emotion_by_phase_gpt,
        "reason_emotion_by_phase_xlm": reason_emotion_by_phase_xlm,
        "supplementary_table_s3_reason_codes": supplementary_table_s3,
        "supplementary_table_s3_exclusions": exclusions,
        "qc_special_codes": qc_special_codes,
        "qc_code_counts": qc_code_counts,
    }


def save_tables(
    tables: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> None:
    tables["post_table"].drop(columns=["post_time_dt"], errors="ignore").to_csv(
        OUTDIR / "manual_motive_post_table.csv", index=False
    )
    tables["label_table"].to_csv(OUTDIR / "manual_motive_label_table.csv", index=False)
    tables["detailed_label_table"].to_csv(
        OUTDIR / "manual_motive_detailed_label_table.csv", index=False
    )
    tables["label_table_all_codes"].to_csv(
        OUTDIR / "manual_motive_label_table_all_codes_qc.csv", index=False
    )
    tables["retweet_events"].drop(columns=["retweet_time_dt"], errors="ignore").to_csv(
        OUTDIR / "manual_motive_retweet_table.csv", index=False
    )
    tables["retweet_label_table"].drop(
        columns=["retweet_time_dt"], errors="ignore"
    ).to_csv(OUTDIR / "manual_motive_retweet_label_table.csv", index=False)
    for name, df in summaries.items():
        df.to_csv(OUTDIR / f"{name}.csv", index=False)


def top_n(df: pd.DataFrame, by: str, n: int = 12) -> pd.DataFrame:
    return df.sort_values(by, ascending=False).head(n).copy()


def save_barh(
    df: pd.DataFrame,
    path: Path,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    n: int = 12,
) -> None:
    plot_df = top_n(df, value_col, n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(plot_df))))
    ax.barh(plot_df[label_col], plot_df[value_col], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_phase_stacked(
    df: pd.DataFrame,
    path: Path,
    phase_col: str,
    value_col: str,
    title: str,
    top_families: list[str],
) -> None:
    plot_df = df[df["motive_family"].isin(top_families)].copy()
    pivot = plot_df.pivot_table(
        index=phase_col,
        columns="motive_family",
        values=value_col,
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.sort_index()
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.82)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Count")
    ax.legend(title="Motive family", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(
    df: pd.DataFrame,
    path: Path,
    index_col: str,
    columns_col: str,
    values_col: str,
    title: str,
    max_rows: int = 12,
) -> None:
    top_rows = (
        df.groupby(index_col)[values_col]
        .sum()
        .sort_values(ascending=False)
        .head(max_rows)
        .index
    )
    pivot = df[df[index_col].isin(top_rows)].pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="sum",
        fill_value=0,
    )
    if pivot.empty:
        return
    pivot = pivot.loc[top_rows]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(pivot))))
    image = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
    ax.set_xticks(
        range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=values_col)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reason_main_figure(summaries: dict[str, pd.DataFrame]) -> None:
    visibility = summaries["reason_visibility_fractional"].copy()
    phase_comp = summaries["reason_phase_retweet_composition"].copy()

    eligible = visibility[visibility["eligible_main"]].copy()
    eligible = eligible.sort_values("log2_multiplier", ascending=True)
    labels = eligible["display_label"].tolist()

    fig = plt.figure(figsize=(11.2, 6.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, :])

    y_positions = np.arange(len(eligible))
    source_pct = eligible["source_share"] * 100
    retweet_pct = eligible["retweet_share"] * 100
    ax_a.hlines(
        y_positions, source_pct, retweet_pct, color="0.72", linewidth=1.4, zorder=1
    )
    ax_a.scatter(
        source_pct,
        y_positions,
        facecolors="white",
        edgecolors="0.35",
        linewidths=1.0,
        s=42,
        label="Source tweets",
        zorder=3,
    )
    ax_a.scatter(
        retweet_pct,
        y_positions,
        color="#0072B2",
        s=42,
        label="Retweet circulation",
        zorder=4,
    )
    for y, value, multiplier in zip(
        y_positions,
        np.maximum(source_pct, retweet_pct),
        eligible["amplification_multiplier"],
    ):
        ax_a.text(value + 0.35, y, f"{multiplier:.2f}x", va="center", fontsize=7)
    ax_a.set_yticks(y_positions, labels=labels)
    ax_a.set_xlabel("Share of reason layer (%)")
    ax_a.set_title(
        "A  Source-post share and retweet-weighted share", loc="left", fontweight="bold"
    )
    ax_a.legend(frameon=False, loc="lower right")
    ax_a.grid(axis="x", alpha=0.18)

    phase_plot = phase_comp[
        phase_comp["retweet_phase_group"].isin(PHASE_GROUP_ORDER)
    ].copy()
    reason_order = (
        phase_plot.groupby("display_label")["fractional_retweet_count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if "Other" in reason_order:
        reason_order = [label for label in reason_order if label != "Other"] + ["Other"]
    pivot = phase_plot.pivot_table(
        index="retweet_phase_group",
        columns="display_label",
        values="phase_share",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(PHASE_GROUP_ORDER)
    pivot = pivot[[label for label in reason_order if label in pivot.columns]]
    left = np.zeros(len(pivot))
    colors = {
        label: REASON_PALETTE[index % len(REASON_PALETTE)]
        for index, label in enumerate(pivot.columns)
    }
    for label in pivot.columns:
        values = pivot[label].to_numpy() * 100
        ax_b.barh(
            pivot.index,
            values,
            left=left,
            color=colors[label],
            label=label,
            height=0.72,
        )
        left += values
    phase_counts = (
        phase_plot.groupby("retweet_phase_group")["phase_unique_retweets"]
        .max()
        .reindex(PHASE_GROUP_ORDER)
        .fillna(0)
        .astype(int)
    )
    for y, count in enumerate(phase_counts):
        ax_b.text(101, y, f"n={count}", va="center", fontsize=7)
    ax_b.set_xlim(0, 112)
    ax_b.set_xlabel("Share of retweet circulation (%)")
    ax_b.set_title(
        "B  Retweet reason composition by phase group", loc="left", fontweight="bold"
    )
    ax_b.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax_b.grid(axis="x", alpha=0.16)

    for ax in [ax_a, ax_b]:
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    publication_paths = [
        OUTDIR / "figure_8_reason_visibility.png",
        OUTDIR / "fig_reason_main.png",
    ]
    for path in publication_paths:
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTDIR / "fig_reason_main_doc.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    for path in [
        OUTDIR / "figure_8_reason_visibility.pdf",
        OUTDIR / "fig_reason_main.pdf",
    ]:
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTDIR / "fig_reason_main.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_reason_emotion_retweet_figure(summaries: dict[str, pd.DataFrame]) -> None:
    emotion = summaries["reason_emotion_retweet_distribution"].copy()
    visibility = summaries["reason_visibility_fractional"].copy()
    order = visibility.sort_values("retweet_share", ascending=False).head(14)[
        "display_label"
    ]
    plot_df = emotion[emotion["display_label"].isin(order)].copy()
    pivot = plot_df.pivot_table(
        index="display_label",
        columns="emotion_display",
        values="fractional_retweet_count",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(order[::-1])
    pivot = pivot.reindex(columns=EMOTION_GROUP_ORDER, fill_value=0.0)

    colors = {
        "sadness": "#4C78A8",
        "anger": "#E45756",
        "fear": "#F58518",
        "disgust": "#72B7B2",
        "Other": "#A6A1A3",
    }
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.42 * len(pivot))))
    left = np.zeros(len(pivot))
    for emotion_label in EMOTION_GROUP_ORDER:
        values = pivot[emotion_label].to_numpy()
        ax.barh(
            pivot.index,
            values,
            left=left,
            color=colors[emotion_label],
            label=emotion_label,
            height=0.72,
        )
        left += values
    totals = pivot.sum(axis=1)
    for y, total in enumerate(totals):
        ax.text(
            total + max(totals.max() * 0.01, 1),
            y,
            f"n={total:.0f}",
            va="center",
            fontsize=7,
        )
    ax.set_xlabel("Retweet-weighted reason observations")
    ax.set_ylabel("")
    ax.set_title(
        "Gemma-labeled tone of retweet-circulated reasons",
        loc="left",
        fontweight="bold",
    )
    ax.legend(
        title="Annotated tone",
        frameon=False,
        ncol=5,
        bbox_to_anchor=(0, 1.02),
        loc="lower left",
    )
    ax.grid(axis="x", alpha=0.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_reason_emotion_retweets.png", dpi=600)
    fig.savefig(OUTDIR / "fig_reason_emotion_retweets.pdf")
    plt.close(fig)


def make_figures(summaries: dict[str, pd.DataFrame]) -> None:
    label_dist = summaries["label_distribution"].copy()
    label_dist["label_for_plot"] = (
        label_dist["motive_family"] + ": " + label_dist["motive_label"]
    )
    save_barh(
        label_dist,
        OUTDIR / "fig_label_distribution.png",
        "label_for_plot",
        "label_assignments",
        "Manual non-reporting motive label distribution",
        "Label assignments",
    )

    retweet_dist = summaries["label_retweet_distribution"].copy()
    retweet_dist["label_for_plot"] = (
        retweet_dist["motive_family"] + ": " + retweet_dist["motive_label"]
    )
    save_barh(
        retweet_dist,
        OUTDIR / "fig_label_retweet_distribution.png",
        "label_for_plot",
        "total_retweets",
        "Retweets received by manually coded motive labels",
        "Retweet-label events",
    )

    top_families = label_dist.head(8)["motive_family"].tolist()
    save_phase_stacked(
        summaries["label_by_post_phase"],
        OUTDIR / "fig_label_by_post_phase.png",
        "post_phase",
        "label_assignments",
        "Manual motive labels by original-post phase",
        top_families,
    )
    save_phase_stacked(
        summaries["label_retweets_by_phase"],
        OUTDIR / "fig_label_retweets_by_phase.png",
        "retweet_phase",
        "retweet_label_events",
        "Retweet amplification of motive labels by phase",
        top_families,
    )

    survival = summaries["label_retweet_survival"].copy()
    survival["label_for_plot"] = (
        survival["motive_family"] + ": " + survival["motive_label"]
    )
    save_barh(
        survival,
        OUTDIR / "fig_label_survival.png",
        "label_for_plot",
        "mean_survival_span_hours",
        "Average retweet survival span by motive label",
        "Mean hours from post to last retweet",
    )

    save_heatmap(
        summaries["reason_emotion_crosstab"],
        OUTDIR / "fig_reason_emotion_heatmap.png",
        "motive_family",
        "emotion_display",
        "tweets",
        "Reason-emotion intersections, Gemma primary",
    )
    save_heatmap(
        summaries["reason_emotion_by_phase"],
        OUTDIR / "fig_reason_emotion_by_phase.png",
        "motive_family",
        "post_phase",
        "tweets",
        "Reason labels across phases",
    )
    save_reason_main_figure(summaries)
    save_reason_emotion_retweet_figure(summaries)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * value:.1f}%"


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    subset = df[columns].head(n).copy()
    headers = list(subset.columns)
    rows = []
    for _, row in subset.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    tables: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]
) -> dict[str, object]:
    post_table = tables["post_table"]
    label_table = tables["label_table"]
    detailed_label_table = tables["detailed_label_table"]
    retweet_events = tables["retweet_events"]
    institutions = tables["institutions"]
    institutional = summaries["institutional_retweet_events"]
    institutional_all_coded = retweet_events[
        retweet_events["retweeter_is_institutional"]
    ]
    label_distribution = summaries["label_distribution"].copy()
    top_label = label_distribution.iloc[0]
    top_retweets = summaries["label_retweet_distribution"].iloc[0]
    top_amp = summaries["label_amplification_multipliers"].iloc[0]
    visibility = summaries["reason_visibility_fractional"].copy()
    global_visibility = summaries["reason_visibility_global_test"].iloc[0]
    institutional_all_coded_posts = institutional_all_coded.merge(
        post_table[["tweet_id", "confession_flag"]],
        on="tweet_id",
        how="left",
    )
    institutional_non_confession = institutional_all_coded_posts[
        institutional_all_coded_posts["confession_flag"].fillna(0).astype(int).eq(0)
    ]
    emotion = summaries["reason_emotion_crosstab"].copy()
    dominant_emotions = (
        emotion.sort_values(["motive_family", "tweets"], ascending=[True, False])
        .groupby("motive_family")
        .head(1)
        .sort_values("tweets", ascending=False)
    )
    exclusion_lookup = summaries["supplementary_table_s3_exclusions"].set_index(
        "exclusion_type"
    )

    metrics = {
        "coded_tweets": int(len(post_table)),
        "analytic_reason_tweets": int(label_table["tweet_id"].nunique()),
        "analytic_label_assignments": int(len(label_table)),
        "analytic_parent_category_assignments": int(len(label_table)),
        "analytic_detailed_reason_assignments": int(len(detailed_label_table)),
        "excluded_x_rows": int(exclusion_lookup.loc["x", "coding_units_n"]),
        "excluded_zero_rows": int(exclusion_lookup.loc["0", "coding_units_n"]),
        "excluded_unresolved_assignments": int(
            exclusion_lookup.loc["unresolved", "parsed_assignments_n"]
        ),
        "excluded_unlisted_assignments": int(
            exclusion_lookup.loc["unlisted", "parsed_assignments_n"]
        ),
        "coded_retweets": int(retweet_events["retweet_id"].nunique()),
        "reason_coded_unique_retweets": int(
            retweet_events.loc[
                retweet_events["motive_families"].notna(), "retweet_id"
            ].nunique()
        ),
        "reason_coded_retweets": int(
            summaries["label_retweet_distribution"]["total_retweets"].sum()
        ),
        "institutional_registry_accounts": int(
            institutions["in_dataset"].str.lower().eq("yes").sum()
        ),
        "institutional_coded_retweets": int(
            institutional_all_coded["retweet_id"].nunique()
        ),
        "institutional_coded_retweeters": int(
            institutional_all_coded["retweeter_username"].nunique()
        ),
        "institutional_coded_source_tweets": int(
            institutional_all_coded["tweet_id"].nunique()
        ),
        "institutional_non_confession_retweets": int(
            institutional_non_confession["retweet_id"].nunique()
        ),
        "institutional_non_confession_retweeters": int(
            institutional_non_confession["retweeter_username"].nunique()
        ),
        "reason_tweets_missing_phase": int(
            label_table.drop_duplicates("tweet_id")["post_phase"].eq("").sum()
        ),
        "institutional_unique_retweets": int(institutional["retweet_id"].nunique()),
        "institutional_accounts": int(institutional["retweeter_username"].nunique()),
        "gpt_emotion_coverage": int(post_table["emotion_gpt"].notna().sum()),
        "gemma_emotion_coverage": int(post_table["emotion_gemma"].notna().sum()),
        "xlm_emotion_coverage": int(post_table["emotion_xlm"].notna().sum()),
        "top_label_family": str(top_label["motive_family"]),
        "top_label_name": str(top_label["display_label"]),
        "top_label_assignments": int(top_label["label_assignments"]),
        "top_retweet_family": str(top_retweets["motive_family"]),
        "top_retweet_name": str(top_retweets["display_label"]),
        "top_retweets": int(top_retweets["total_retweets"]),
        "top_amp_family": str(top_amp["motive_family"]),
        "top_amp_name": str(top_amp["display_label"]),
        "top_amp_multiplier": float(top_amp["amplification_multiplier"]),
        "visibility_global_js": float(global_visibility["observed"]),
        "visibility_global_p": float(global_visibility["p_value"]),
        "fractional_top_amp_family": str(visibility.iloc[0]["motive_family"]),
        "fractional_top_amp_name": str(visibility.iloc[0]["display_label"]),
        "fractional_top_amp_multiplier": float(
            visibility.iloc[0]["amplification_multiplier"]
        ),
    }

    figure_8_caption = (
        "Figure 8. Selective visibility of manually coded reasons in retweet circulation. "
        "(A) Fractional reason shares among coded source tweets and retweet-weighted "
        "circulation. Each multi-label source tweet contributes total weight 1 "
        "across its unique parent categories; each retweet contributes total weight "
        "1 across the source tweet's unique parent categories. (B) Retweet reason "
        "composition across collapsed "
        "campaign phase groups. Bars sum to 100% within phase group, and labels "
        "report the number of unique reason-coded retweets contributing to each "
        "phase group. Full reason tables, low-count reason families and emotion "
        "sensitivity outputs are reported in the Supplement."
    )
    supplementary_table_s3_caption = (
        "Supplementary Table S3. Reconciled manual reasons for non-reporting. "
        "Parent-code rows report one assignment per tweet and parent category as a "
        "percentage of 1,238 parent-category assignments. Subcode rows report detailed "
        "codes as a percentage of 1,266 detailed-code assignments. Unique coding units "
        "use 824 analytic reason tweets as the denominator. Rows coded x or 0 are "
        "excluded from all reason analyses; unresolved and unlisted codes are reported "
        "separately."
    )
    supplement_caption = (
        "Supplementary Figure S7. Gemma-labeled textual tone of retweet-circulated "
        "reasons. Bars show retweet-weighted reason observations for the most visible "
        "reason families, with each retweet of a multi-label source tweet split "
        "fractionally across that tweet's unique parent categories. Emotion labels "
        "describe the "
        "model-annotated tone of source tweet text, not retweeter sentiment or private "
        "emotional state."
    )

    report = f"""# Manual motive analysis

## Source and scope

- Coded tweet rows: `{metrics["coded_tweets"]}`.
- Tweets with analytic reason labels after excluding all rows coded `x` or `0` and excluding unresolved or unlisted code assignments: `{metrics["analytic_reason_tweets"]}`.
- Detailed reason-code assignments retained from the workbook: `{metrics["analytic_detailed_reason_assignments"]}`.
- Deduplicated tweet-parent assignments used for parent-category analyses: `{metrics["analytic_parent_category_assignments"]}`.
- Excluded `x` rows: `{metrics["excluded_x_rows"]}`; excluded `0` rows: `{metrics["excluded_zero_rows"]}`.
- Excluded unresolved assignments: `{metrics["excluded_unresolved_assignments"]}`; excluded unlisted assignments: `{metrics["excluded_unlisted_assignments"]}`.
- Retweets of coded tweets: `{metrics["coded_retweets"]}` unique retweets.
- Unique retweets of reason-coded tweets: `{metrics["reason_coded_unique_retweets"]}`.
- Full-count retweet-parent events after parent-category expansion: `{metrics["reason_coded_retweets"]}`.
- Institutional-related accounts present in the dataset: `{metrics["institutional_registry_accounts"]}`.
- Institutional-related retweeters of any manually coded tweet: `{metrics["institutional_coded_retweeters"]}` accounts, `{metrics["institutional_coded_retweets"]}` unique retweets of `{metrics["institutional_coded_source_tweets"]}` source tweets.
- Institutional-related retweeters of reason-coded tweets: `{metrics["institutional_accounts"]}` accounts, `{metrics["institutional_unique_retweets"]}` unique retweets.
- Institutional-related retweets of non-confession rows: `{metrics["institutional_non_confession_retweets"]}` unique retweets from `{metrics["institutional_non_confession_retweeters"]}` accounts.
- Gemma emotion coverage for coded tweets: `{metrics["gemma_emotion_coverage"]}` rows.
- Reason-coded tweets without phase assignment: `{metrics["reason_tweets_missing_phase"]}`.

This analysis treats motive coding as final consensus manual coding, not coder-specific reliability evidence. Main summaries count each tweet once per parent category, even when multiple detailed codes map to that parent.

## Main findings

1. The largest manual reason family is `{metrics["top_label_family"]}` ({metrics["top_label_name"]}) with `{metrics["top_label_assignments"]}` parent-category assignments.
2. Retweet attention differs from posting volume. `{metrics["top_retweet_family"]}` ({metrics["top_retweet_name"]}) receives the most full-count retweet-parent events: `{metrics["top_retweets"]}`.
3. The highest amplification multiplier is `{metrics["top_amp_family"]}` ({metrics["top_amp_name"]}), at `{metrics["top_amp_multiplier"]:.2f}x` its share of labeled posts.
4. Institutional-account retweets of reason-coded posts are sparse: the dataset contains `{metrics["institutional_registry_accounts"]}` institutional-related accounts, `{metrics["institutional_coded_retweeters"]}` of them retweeted at least one manually coded tweet, and `{metrics["institutional_accounts"]}` retweeted reason-coded tweets (`{metrics["institutional_unique_retweets"]}` unique retweets). Use this as descriptive context only because the reason-coded institutional layer is too sparse for statistical comparison.
5. Fractional multi-label visibility test: global Jensen-Shannon distance `{metrics["visibility_global_js"]:.4f}`, phase-stratified permutation `p = {metrics["visibility_global_p"]:.4f}`.

## Label distribution

{markdown_table(label_distribution, ["motive_family", "display_label", "label_assignments", "unique_tweets", "assignment_share"], 12)}

## Retweet distribution

{markdown_table(summaries["label_retweet_distribution"], ["motive_family", "display_label", "labeled_tweets", "total_retweets", "share_retweeted", "median_retweets", "mean_retweets", "max_retweets"], 12)}

## Amplification multipliers

{markdown_table(summaries["label_amplification_multipliers"], ["motive_family", "display_label", "unique_tweets", "full_count_retweets", "post_share", "retweet_share", "amplification_multiplier"], 12)}

## Fractional visibility test

{markdown_table(summaries["reason_visibility_fractional"], ["motive_family", "display_label", "unique_tweets", "source_share", "retweet_share", "percentage_point_change", "log2_multiplier", "permutation_p", "fdr_q"], 12)}

## Retweet composition by phase group

{markdown_table(summaries["reason_phase_retweet_composition"], ["retweet_phase_group", "display_label", "fractional_retweet_count", "phase_share", "phase_unique_retweets"], 16)}

## Retweet survival

{markdown_table(summaries["label_retweet_survival"], ["motive_family", "display_label", "tweets", "mean_total_retweets", "mean_survival_span_hours", "share_late_after_12h", "share_late_after_24h"], 12)}

## Reason-emotion intersections

{markdown_table(dominant_emotions, ["motive_family", "display_label", "emotion_display", "tweets", "reason_emotion_share"], 12)}

## Caveats for manuscript decisions

- The strongest manuscript-ready use is descriptive: motive frequencies, phase distributions, amplification differences, and reason-emotion intersections.
- Late phases have low counts for reason-coded tweets and should not carry strong inferential claims.
- Institutional retweeting of reason-coded posts is too sparse for a statistical test; use only as bounded descriptive context.
- Emotion interactions depend on model labels. Use Gemma as the manuscript primary model and GPT/XLM as sensitivity, not as human-coded emotion ground truth.

## Manuscript-ready captions

{figure_8_caption}

{supplementary_table_s3_caption}

{supplement_caption}
"""
    (OUTDIR / "report.md").write_text(report, encoding="utf-8")
    (OUTDIR / "figure_captions.md").write_text(
        f"{figure_8_caption}\n\n{supplementary_table_s3_caption}\n\n"
        f"{supplement_caption}\n",
        encoding="utf-8",
    )
    (OUTDIR / "analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    ensure_outdir()
    inputs = load_inputs()
    tables = build_tables(inputs)
    summaries = summarize(tables)
    save_tables(tables, summaries)
    make_figures(summaries)
    metrics = write_report(tables, summaries)
    print("Manual motive analysis complete")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
