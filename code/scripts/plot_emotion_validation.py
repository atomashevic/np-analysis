#!/usr/bin/env python3
"""Reproduce Supplementary Figure S1 and emotion-validation statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "emotion_validation"
HUMAN_PATH = (
    ROOT / "results" / "emotions_ff" / "ff_samples_18_25_annotations_with_majority.csv"
)
MODEL_PATHS = {
    "Gemma 4 31B": (
        ROOT
        / "results"
        / "emotions_gemma"
        / "emotions_gemma4:31b_temp1.0_all_tweets.jsonl",
        "id",
    ),
    "GPT-5": (
        ROOT / "results" / "emotions_gpt" / "emotions_gpt-5_temp1.0_validation_1.jsonl",
        "id_str",
    ),
    "GPT-5.4": (
        ROOT
        / "results"
        / "emotions_gpt"
        / "emotions_gpt-5.4_temp1.0_validation_1.jsonl",
        "id_str",
    ),
}
LABEL_TRANSLATION = {
    "radost": "joy",
    "tuga": "sadness",
    "poverenje": "trust",
    "gadjenje": "disgust",
    "gađenje": "disgust",
    "strah": "fear",
    "straх": "fear",
    "bes": "anger",
    "iznenadjenje": "surprise",
    "iznenađenje": "surprise",
    "anticipacija": "anticipation",
    "iščekivanje": "anticipation",
    "neutralno": "neutral",
    "Emocionalno neutralno": "neutral",
    "nepoznato": "unknown",
    "Nepoznato": "unknown",
    "Ne mogu da razumem": "unknown",
}
MODEL_COLOURS = {
    "Gemma 4 31B": "#009E73",
    "GPT-5": "#0072B2",
    "GPT-5.4": "#D55E00",
}
MODEL_MARKERS = {"Gemma 4 31B": "^", "GPT-5": "o", "GPT-5.4": "s"}
MODEL_ORDER = ["Gemma 4 31B", "GPT-5", "GPT-5.4"]


def translate_label(label: str) -> str:
    """Translate an archived Serbian label into the manuscript label set."""
    try:
        return LABEL_TRANSLATION[label.strip()]
    except KeyError as error:
        raise ValueError(f"Unrecognised emotion label: {label!r}") from error


def load_human_annotations(
    path: Path = HUMAN_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return tweet-level modal labels and the underlying student annotations."""
    annotations = pd.read_csv(path, dtype={"tweet_id": "string"})
    required = {"tweet_id", "annotator_code", "label"}
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Human annotation file lacks columns: {sorted(missing)}")
    if annotations.duplicated(["tweet_id", "annotator_code"]).any():
        raise ValueError("Duplicate student annotation for the same tweet")

    annotations = annotations.loc[:, ["tweet_id", "annotator_code", "label"]].copy()
    annotations["label_en"] = annotations["label"].map(translate_label)
    grouped = annotations.groupby("tweet_id", sort=False)["label_en"].agg(list)
    human = grouped.rename("student_labels").reset_index()
    human["human_modal"] = human["student_labels"].map(
        lambda labels: Counter(labels).most_common(1)[0][0]
    )
    human["human_modal_votes"] = human.apply(
        lambda row: row["student_labels"].count(row["human_modal"]), axis=1
    )
    human["student_raters"] = human["student_labels"].map(len)
    return human, annotations


def load_model_labels(path: Path, id_field: str, model: str) -> pd.DataFrame:
    """Read one archived JSONL model output and validate identifier uniqueness."""
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if id_field not in record or "emotion" not in record:
                raise ValueError(f"Malformed record at {path}:{line_number}")
            records.append(
                {
                    "tweet_id": str(record[id_field]),
                    model: translate_label(str(record["emotion"])),
                }
            )
    labels = pd.DataFrame.from_records(records)
    if labels["tweet_id"].duplicated().any():
        raise ValueError(f"Duplicate tweet identifiers in {path}")
    return labels


def build_validation_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join modal human labels to the primary Gemma and archived GPT labels."""
    validation, annotations = load_human_annotations()
    for model, (path, id_field) in MODEL_PATHS.items():
        labels = load_model_labels(path, id_field, model)
        validation = validation.merge(labels, on="tweet_id", how="left", validate="1:1")
    if len(validation) != 400:
        raise ValueError(f"Expected 400 validation tweets, found {len(validation)}")
    missing = validation[MODEL_ORDER].isna().sum()
    if missing.any():
        raise ValueError(f"Missing model labels in validation set: {missing.to_dict()}")
    return validation, annotations


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Calculate unweighted Cohen's kappa for two nominal label vectors."""
    if len(labels_a) != len(labels_b) or not len(labels_a):
        raise ValueError("Cohen's kappa requires two non-empty, equally sized vectors")
    observed = float(labels_a.eq(labels_b).mean())
    labels = set(labels_a).union(labels_b)
    expected = sum(
        float(labels_a.eq(label).mean() * labels_b.eq(label).mean()) for label in labels
    )
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else np.nan
    return (observed - expected) / (1.0 - expected)


def nominal_krippendorff_alpha(annotations: pd.DataFrame) -> float:
    """Calculate nominal Krippendorff's alpha with varying raters per tweet."""
    labels = sorted(annotations["label_en"].unique())
    label_index = {label: index for index, label in enumerate(labels)}
    coincidences = np.zeros((len(labels), len(labels)), dtype=float)

    for unit_labels in annotations.groupby("tweet_id", sort=False)["label_en"]:
        counts = Counter(unit_labels[1])
        denominator = sum(counts.values()) - 1
        for label_a, count_a in counts.items():
            for label_b, count_b in counts.items():
                ordered_pairs = count_a * (count_b - int(label_a == label_b))
                coincidences[label_index[label_a], label_index[label_b]] += (
                    ordered_pairs / denominator
                )

    total = coincidences.sum()
    observed_disagreement = (total - np.trace(coincidences)) / total
    marginals = coincidences.sum(axis=1)
    expected_agreement = sum(count * (count - 1) for count in marginals)
    expected_disagreement = 1.0 - expected_agreement / (total * (total - 1))
    return 1.0 - observed_disagreement / expected_disagreement


def model_metrics(validation: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Calculate accuracy and kappa against the modal human label."""
    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "analysis": "model_vs_modal_human",
                "subset": subset,
                "n_tweets": len(validation),
                "comparison": model,
                "accuracy": validation[model].eq(validation["human_modal"]).mean(),
                "cohen_kappa": cohen_kappa(
                    validation[model], validation["human_modal"]
                ),
                "krippendorff_alpha": np.nan,
            }
        )
    return pd.DataFrame(rows)


def cumulative_metrics(validation: pd.DataFrame) -> pd.DataFrame:
    """Calculate model-human agreement at each human-consensus threshold."""
    rows = []
    for minimum_votes in range(1, int(validation["human_modal_votes"].max()) + 1):
        subset = validation.loc[validation["human_modal_votes"] >= minimum_votes]
        for model in MODEL_ORDER:
            rows.append(
                {
                    "minimum_modal_votes": minimum_votes,
                    "n_tweets": len(subset),
                    "model": model,
                    "accuracy": subset[model].eq(subset["human_modal"]).mean(),
                    "cohen_kappa": cohen_kappa(subset[model], subset["human_modal"]),
                }
            )
    return pd.DataFrame(rows)


def pairwise_kappa(validation: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise kappa matrix for models and modal human labels."""
    columns = [*MODEL_ORDER, "Human modal"]
    labels = validation.rename(columns={"human_modal": "Human modal"})
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for row_name in columns:
        for column_name in columns:
            matrix.loc[row_name, column_name] = cohen_kappa(
                labels[row_name], labels[column_name]
            )
    return matrix


def write_tables(
    validation: pd.DataFrame,
    annotations: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Write the manuscript statistics and return data needed for the figure."""
    high_consensus = validation.loc[validation["human_modal_votes"] >= 4].copy()
    student_alpha = nominal_krippendorff_alpha(annotations)
    metrics = pd.concat(
        [
            model_metrics(validation, "all_400"),
            model_metrics(high_consensus, "strict_majority"),
            pd.DataFrame(
                [
                    {
                        "analysis": "student_interrater_reliability",
                        "subset": "all_400",
                        "n_tweets": len(validation),
                        "comparison": "student annotators",
                        "accuracy": np.nan,
                        "cohen_kappa": np.nan,
                        "krippendorff_alpha": student_alpha,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    cumulative = cumulative_metrics(validation)
    pairwise = pairwise_kappa(high_consensus)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "emotion_validation_metrics.csv", index=False)
    cumulative.to_csv(output_dir / "emotion_validation_cumulative.csv", index=False)
    pairwise.to_csv(output_dir / "emotion_validation_pairwise_kappa.csv")
    return metrics, cumulative, pairwise, student_alpha


def annotate_heatmap(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    """Add values to a kappa heatmap with contrast-aware text."""
    for row_index, row in enumerate(matrix.itertuples(index=False, name=None)):
        for column_index, value in enumerate(row):
            colour = "white" if value >= 0.72 else "#202020"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=colour,
                fontsize=7.5,
            )


def plot_figure(
    cumulative: pd.DataFrame,
    pairwise: pd.DataFrame,
    student_alpha: float,
    output_dir: Path,
) -> None:
    """Create and export the two-panel Supplementary Figure S1."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, (agreement_ax, heatmap_ax) = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.35),
            gridspec_kw={"width_ratios": [1.12, 1]},
            constrained_layout=True,
        )
        plot_cumulative_agreement(agreement_ax, cumulative)
        plot_pairwise_heatmap(figure, heatmap_ax, pairwise)
        figure.text(
            0.5,
            -0.025,
            f"Student nominal Krippendorff's α = {student_alpha:.3f} (400 tweets)",
            ha="center",
            va="top",
            fontsize=7.5,
        )
        figure.savefig(
            output_dir / "figure_s1.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            output_dir / "figure_s1.pdf",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def plot_cumulative_agreement(ax: plt.Axes, cumulative: pd.DataFrame) -> None:
    """Plot kappa against the minimum number of votes for the modal label."""
    thresholds = sorted(cumulative["minimum_modal_votes"].unique())
    for model in MODEL_ORDER:
        model_data = cumulative.loc[cumulative["model"] == model]
        ax.plot(
            model_data["minimum_modal_votes"],
            model_data["cohen_kappa"],
            color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            markersize=4.5,
            linewidth=1.5,
            label=model,
        )
    for level in (0.4, 0.6, 0.8):
        ax.axhline(level, color="#9E9E9E", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(4, color="#6F6F6F", linewidth=0.9, linestyle=":", zorder=0)
    ax.set(
        xlabel="Minimum human votes for modal label",
        ylabel="Cohen's κ (model vs modal human label)",
        xlim=(0.75, 7.25),
        ylim=(0.37, 1.03),
        xticks=thresholds,
        title="Model agreement across human-consensus thresholds",
    )
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    sample_sizes = (
        cumulative.drop_duplicates("minimum_modal_votes")
        .set_index("minimum_modal_votes")
        .loc[thresholds, "n_tweets"]
    )
    sample_axis = ax.secondary_xaxis("top")
    sample_axis.set_xticks(thresholds, labels=sample_sizes.astype(str).tolist())
    sample_axis.set_xlabel("Validation tweets (n)", labelpad=4)
    sample_axis.tick_params(length=0, pad=2)
    ax.text(-0.12, 1.27, "A", transform=ax.transAxes, fontweight="bold", fontsize=10)


def plot_pairwise_heatmap(
    figure: plt.Figure,
    ax: plt.Axes,
    pairwise: pd.DataFrame,
) -> None:
    """Plot pairwise kappa in the strict-majority subset."""
    image = ax.imshow(pairwise.to_numpy(), cmap="viridis", vmin=0.5, vmax=1.0)
    labels = ["Gemma 4\n31B", "GPT-5", "GPT-5.4", "Human\nmodal"]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="x", rotation=0)
    ax.set_title("Pairwise agreement, strict-majority subset (n = 224)")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate_heatmap(ax, pairwise)
    colourbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.set_label("Cohen's κ")
    colourbar.outline.set_visible(False)
    ax.text(-0.14, 1.27, "B", transform=ax.transAxes, fontweight="bold", fontsize=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figure and metric outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation, annotations = build_validation_table()
    _, cumulative, pairwise, student_alpha = write_tables(
        validation,
        annotations,
        args.output_dir,
    )
    plot_figure(cumulative, pairwise, student_alpha, args.output_dir)


if __name__ == "__main__":
    main()
