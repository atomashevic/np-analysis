#!/usr/bin/env python3
"""Track model-derived emotion labels in first-person disclosures by phase."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
RESOLUTION = ROOT / "analysis/manual_coding_motives/manual_coding_id_resolution.csv"
GEMMA = ROOT / "results/emotions_gemma/emotions_gemma4:31b_temp1.0_all_tweets.jsonl"
GEMMA_REPEAT = (
    ROOT / "results/emotions_gemma/emotions_gemma4:31b_temp1.0_all_tweets_1.jsonl"
)
GPT = ROOT / "results/emotions_gpt/emotions_gpt-5.4_temp1.0_all_tweets.jsonl"
XLM = ROOT / "results/emotions_first_run/np_text_features.csv"
OUTDIR = ROOT / "analysis/manual_coding_motives"

TRANSLATION = {
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
PHASE_GROUPS = {
    1: "phase 1",
    2: "phase 2",
    3: "phase 3",
    4: "phases 4-6",
    5: "phases 4-6",
    6: "phases 4-6",
}
GROUP_ORDER = ["phase 1", "phase 2", "phase 3", "phases 4-6"]
FULL_EMOTION_ORDER = [
    "sadness",
    "anger",
    "fear",
    "disgust",
    "joy",
    "neutral",
    "trust",
    "anticipation",
    "surprise",
    "unknown",
]
COLLAPSED_ORDER = ["sadness", "anger", "fear", "disgust", "Other"]
XLM_ORDER = ["sadness", "anger", "fear", "joy"]
EXPECTED_PHASE_COUNTS = {1: 45, 2: 1002, 3: 145, 4: 15, 5: 1, 6: 2}
PERMUTATION_REPS = 50_000
BOOTSTRAP_REPS = 5_000
SEED = 20260721


def read_jsonl(path: Path, id_key: str, label_name: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records = record if isinstance(record, list) else [record]
            for item in records:
                rows.append(
                    {
                        "resolved_tweet_id": str(item[id_key]),
                        label_name: TRANSLATION.get(item["emotion"], item["emotion"]),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("resolved_tweet_id", keep="last")


def load_first_person() -> pd.DataFrame:
    manual = pd.read_csv(RESOLUTION, dtype={"resolved_tweet_id": str})
    first_person = manual[manual["first_person_adjudicated"]].copy()
    first_person = first_person[first_person["phase_id"].notna()].copy()
    first_person["phase_id"] = first_person["phase_id"].astype(int)
    first_person["phase_group"] = first_person["phase_id"].map(PHASE_GROUPS)
    if (
        first_person["phase_id"].value_counts().sort_index().to_dict()
        != EXPECTED_PHASE_COUNTS
    ):
        raise ValueError("First-person phase counts differ from the adjudicated audit")

    sources = [
        read_jsonl(GEMMA, "id", "gemma"),
        read_jsonl(GEMMA_REPEAT, "id", "gemma_repeat"),
        read_jsonl(GPT, "id_str", "gpt"),
    ]
    for source in sources:
        first_person = first_person.merge(source, on="resolved_tweet_id", how="left")
    xlm = pd.read_csv(
        XLM,
        usecols=["id_str", "label_xlm_emotions"],
        dtype=str,
    ).rename(columns={"id_str": "resolved_tweet_id", "label_xlm_emotions": "xlm"})
    first_person = first_person.merge(
        xlm.drop_duplicates("resolved_tweet_id", keep="last"),
        on="resolved_tweet_id",
        how="left",
    )
    if first_person[["gemma", "gpt", "xlm"]].isna().any().any():
        raise ValueError("Primary or sensitivity emotion labels are missing")
    return first_person


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


def summarize_phase_labels(first_person: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [range(1, 7), FULL_EMOTION_ORDER], names=["phase_id", "emotion"]
    )
    counts = (
        first_person.groupby(["phase_id", "gemma"]).size().reindex(index, fill_value=0)
    )
    result = counts.rename("n").reset_index()
    result["phase_n"] = result["phase_id"].map(EXPECTED_PHASE_COUNTS)
    result["share"] = result["n"] / result["phase_n"]
    intervals = [
        wilson_interval(int(row.n), int(row.phase_n))
        for row in result.itertuples(index=False)
    ]
    result["wilson_low"] = [interval[0] for interval in intervals]
    result["wilson_high"] = [interval[1] for interval in intervals]
    return result


def collapsed_labels(values: pd.Series) -> pd.Series:
    return values.where(values.isin(COLLAPSED_ORDER[:-1]), "Other")


def summarize_groups(first_person: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for model, order in [
        ("gemma", COLLAPSED_ORDER),
        ("gpt", COLLAPSED_ORDER),
        ("xlm", XLM_ORDER),
    ]:
        labels = (
            collapsed_labels(first_person[model])
            if model != "xlm"
            else first_person[model]
        )
        frame = first_person.assign(emotion=labels)
        index = pd.MultiIndex.from_product(
            [GROUP_ORDER, order], names=["phase_group", "emotion"]
        )
        counts = (
            frame.groupby(["phase_group", "emotion"])
            .size()
            .reindex(index, fill_value=0)
        )
        result = counts.rename("n").reset_index()
        result["model"] = model
        result["group_n"] = result.groupby("phase_group")["n"].transform("sum")
        result["share"] = result["n"] / result["group_n"]
        intervals = [
            wilson_interval(int(row.n), int(row.group_n))
            for row in result.itertuples(index=False)
        ]
        result["wilson_low"] = [interval[0] for interval in intervals]
        result["wilson_high"] = [interval[1] for interval in intervals]
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def corrected_cramers_v(statistic: float, table: np.ndarray) -> float:
    total = table.sum()
    rows, columns = table.shape
    phi_squared = statistic / total
    corrected_phi = max(0.0, phi_squared - (columns - 1) * (rows - 1) / (total - 1))
    corrected_rows = rows - (rows - 1) ** 2 / (total - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (total - 1)
    return math.sqrt(corrected_phi / min(corrected_rows - 1, corrected_columns - 1))


def permutation_p_value(
    group_codes: np.ndarray,
    label_codes: np.ndarray,
    shape: tuple[int, int],
    observed_statistic: float,
    rng: np.random.Generator,
) -> float:
    row_totals = np.bincount(group_codes, minlength=shape[0])
    column_totals = np.bincount(label_codes, minlength=shape[1])
    expected = np.outer(row_totals, column_totals) / len(group_codes)
    exceedances = 0
    for _ in range(PERMUTATION_REPS):
        permuted = rng.permutation(label_codes)
        table = np.bincount(
            group_codes * shape[1] + permuted,
            minlength=shape[0] * shape[1],
        ).reshape(shape)
        statistic = np.sum((table - expected) ** 2 / expected)
        exceedances += statistic >= observed_statistic - 1e-12
    return (exceedances + 1) / (PERMUTATION_REPS + 1)


def bootstrap_v_interval(
    group_codes: np.ndarray,
    label_codes: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[float, float]:
    group_labels = [label_codes[group_codes == group] for group in range(shape[0])]
    values = np.empty(BOOTSTRAP_REPS)
    for iteration in range(BOOTSTRAP_REPS):
        table = np.vstack(
            [
                np.bincount(
                    rng.choice(labels, size=len(labels), replace=True),
                    minlength=shape[1],
                )
                for labels in group_labels
            ]
        )
        statistic = chi2_contingency(table, correction=False)[0]
        values[iteration] = corrected_cramers_v(statistic, table)
    return tuple(np.quantile(values, [0.025, 0.975]))


def distribution_tests(first_person: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for model, order in [
        ("gemma", COLLAPSED_ORDER),
        ("gpt", COLLAPSED_ORDER),
        ("xlm", XLM_ORDER),
    ]:
        labels = (
            collapsed_labels(first_person[model])
            if model != "xlm"
            else first_person[model]
        )
        group_codes = pd.Categorical(
            first_person["phase_group"], categories=GROUP_ORDER
        ).codes
        label_codes = pd.Categorical(labels, categories=order).codes
        table = np.bincount(
            group_codes * len(order) + label_codes,
            minlength=len(GROUP_ORDER) * len(order),
        ).reshape(len(GROUP_ORDER), len(order))
        statistic, asymptotic_p, dof, expected = chi2_contingency(
            table, correction=False
        )
        uncorrected_v = math.sqrt(
            statistic / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))
        )
        corrected_v = corrected_cramers_v(statistic, table)
        low, high = bootstrap_v_interval(group_codes, label_codes, table.shape, rng)
        rows.append(
            {
                "model": model,
                "n": int(table.sum()),
                "table_rows": table.shape[0],
                "table_columns": table.shape[1],
                "chi_square": statistic,
                "degrees_of_freedom": dof,
                "asymptotic_p": asymptotic_p,
                "permutation_p": permutation_p_value(
                    group_codes, label_codes, table.shape, statistic, rng
                ),
                "cramers_v": uncorrected_v,
                "bias_corrected_cramers_v": corrected_v,
                "bias_corrected_v_bootstrap_low": low,
                "bias_corrected_v_bootstrap_high": high,
                "expected_cells_below_5": int((expected < 5).sum()),
                "minimum_expected_count": float(expected.min()),
                "permutation_reps": PERMUTATION_REPS,
                "bootstrap_reps": BOOTSTRAP_REPS,
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows)


def cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    labels = sorted(set(left) | set(right))
    observed = (left.to_numpy() == right.to_numpy()).mean()
    left_share = left.value_counts(normalize=True).reindex(labels, fill_value=0)
    right_share = right.value_counts(normalize=True).reindex(labels, fill_value=0)
    expected = float((left_share * right_share).sum())
    return (observed - expected) / (1 - expected)


def agreement_summary(first_person: pd.DataFrame) -> pd.DataFrame:
    repeat = first_person.dropna(subset=["gemma_repeat"])
    return pd.DataFrame(
        [
            {
                "comparison": "Gemma 4 31B vs GPT-5.4",
                "n": len(first_person),
                "observed_agreement": (
                    first_person["gemma"] == first_person["gpt"]
                ).mean(),
                "cohen_kappa": cohen_kappa(first_person["gemma"], first_person["gpt"]),
                "coverage_note": "all phases",
            },
            {
                "comparison": "Gemma primary vs partial repeat",
                "n": len(repeat),
                "observed_agreement": (
                    repeat["gemma"] == repeat["gemma_repeat"]
                ).mean(),
                "cohen_kappa": cohen_kappa(repeat["gemma"], repeat["gemma_repeat"]),
                "coverage_note": "phase 1 and early phase 2 only",
            },
        ]
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    first_person = load_first_person()
    phase_summary = summarize_phase_labels(first_person)
    group_summary = summarize_groups(first_person)
    tests = distribution_tests(first_person)
    agreement = agreement_summary(first_person)

    phase_summary.to_csv(OUTDIR / "first_person_emotion_by_phase.csv", index=False)
    group_summary.to_csv(
        OUTDIR / "first_person_emotion_by_lifecycle_group.csv", index=False
    )
    tests.to_csv(OUTDIR / "first_person_emotion_tests.csv", index=False)
    agreement.to_csv(OUTDIR / "first_person_emotion_model_agreement.csv", index=False)
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
