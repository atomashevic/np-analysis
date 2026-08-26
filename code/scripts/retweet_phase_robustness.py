#!/usr/bin/env python3
"""Negative-binomial and temporal-aggregation sensitivity checks for retweet phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, xlogy

REQUIRED_HOURLY_COLUMNS = {"hour_idx", "hour_start_utc", "hour_end_utc", "n_retweets"}
REQUIRED_BASELINE_COLUMNS = {"phase_id", "start_utc", "end_utc"}
TRANSITION_LABELS = [
    "emergence to viral peak",
    "viral peak to declining amplification",
    "declining amplification to sharp contraction",
    "sharp contraction to lower-volume contraction",
    "contraction to dormant period",
    "dormant period to late reactivation",
    "late reactivation to residual afterlife",
]


@dataclass(frozen=True)
class SegmentationFit:
    k: int
    score: float
    bic: float
    dispersion_size: float
    segments: tuple[tuple[int, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shared-dispersion negative-binomial sensitivity analysis for retweet phases"
    )
    parser.add_argument(
        "--hourly",
        default="analysis/retweet_network_phases/hourly_retweet_counts.csv",
        help="Hourly retweet-count CSV from the primary phase analysis",
    )
    parser.add_argument(
        "--baseline-boundaries",
        default=(
            "analysis/eight_phase_sensitivity/retweet_network_phases/"
            "phase_boundaries.csv"
        ),
        help="Archived hourly Poisson sensitivity boundary CSV",
    )
    parser.add_argument(
        "--outdir",
        default="analysis/retweet_phase_robustness",
        help="Output directory",
    )
    parser.add_argument("--max-phases", type=int, default=12)
    parser.add_argument("--retained-phases", type=int, default=8)
    parser.add_argument("--min-phase-hours", type=int, default=12)
    parser.add_argument("--min-phase-events", type=int, default=300)
    parser.add_argument("--aggregate-hours", type=int, default=3)
    parser.add_argument("--profile-grid-points", type=int, default=61)
    parser.add_argument("--log-size-min", type=float, default=-8.0)
    parser.add_argument("--log-size-max", type=float, default=16.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_phases < 1:
        raise ValueError("--max-phases must be at least 1")
    if not 1 <= args.retained_phases <= args.max_phases:
        raise ValueError("--retained-phases must be between 1 and --max-phases")
    if args.min_phase_hours < 1 or args.min_phase_events < 0:
        raise ValueError(
            "Phase constraints must be non-negative and duration must be positive"
        )
    if args.aggregate_hours < 2:
        raise ValueError("--aggregate-hours must be at least 2")
    if args.profile_grid_points < 5:
        raise ValueError("--profile-grid-points must be at least 5")
    if args.log_size_min >= args.log_size_max:
        raise ValueError("--log-size-min must be smaller than --log-size-max")


def load_hourly_counts(path: Path) -> pd.DataFrame:
    hourly = pd.read_csv(path)
    missing = sorted(REQUIRED_HOURLY_COLUMNS - set(hourly.columns))
    if missing:
        raise ValueError(f"Hourly count file is missing columns: {', '.join(missing)}")
    hourly = hourly.copy()
    hourly["hour_start_utc"] = pd.to_datetime(hourly["hour_start_utc"], utc=True)
    hourly["hour_end_utc"] = pd.to_datetime(hourly["hour_end_utc"], utc=True)
    hourly["n_retweets"] = hourly["n_retweets"].astype(int)
    hourly = hourly.sort_values("hour_start_utc").reset_index(drop=True)
    expected = pd.date_range(
        hourly["hour_start_utc"].iloc[0],
        hourly["hour_start_utc"].iloc[-1],
        freq="1h",
        tz="UTC",
    )
    if not hourly["hour_start_utc"].reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError("Hourly count file must contain contiguous one-hour bins")
    return hourly


def load_baseline_boundaries(path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(path)
    missing = sorted(REQUIRED_BASELINE_COLUMNS - set(baseline.columns))
    if missing:
        raise ValueError(
            f"Baseline boundary file is missing columns: {', '.join(missing)}"
        )
    baseline = baseline.sort_values("phase_id").reset_index(drop=True)
    baseline["start_utc"] = pd.to_datetime(baseline["start_utc"], utc=True)
    baseline["end_utc"] = pd.to_datetime(baseline["end_utc"], utc=True)
    return baseline


def aggregate_complete_bins(
    hourly: pd.DataFrame, bin_hours: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    complete_hours = (len(hourly) // bin_hours) * bin_hours
    used = hourly.iloc[:complete_hours].copy()
    dropped = hourly.iloc[complete_hours:].copy()
    used["bin_id"] = np.arange(complete_hours) // bin_hours
    aggregated = (
        used.groupby("bin_id", sort=True)
        .agg(
            bin_start_utc=("hour_start_utc", "first"),
            bin_end_utc=("hour_end_utc", "last"),
            n_retweets=("n_retweets", "sum"),
        )
        .reset_index(drop=True)
    )
    aggregated["duration_hours"] = bin_hours
    qa = {
        "input_hours": int(len(hourly)),
        "complete_hours": int(complete_hours),
        "dropped_trailing_hours": int(len(dropped)),
        "input_events": int(hourly["n_retweets"].sum()),
        "retained_events": int(aggregated["n_retweets"].sum()),
        "dropped_trailing_events": int(dropped["n_retweets"].sum()),
    }
    return aggregated, qa


def hourly_as_bins(hourly: pd.DataFrame) -> pd.DataFrame:
    bins = hourly.rename(
        columns={"hour_start_utc": "bin_start_utc", "hour_end_utc": "bin_end_utc"}
    )[["bin_start_utc", "bin_end_utc", "n_retweets"]].copy()
    bins["duration_hours"] = 1
    return bins


def negative_binomial_cost_matrix(
    counts: np.ndarray,
    dispersion_size: float,
    min_bins: int,
    min_events: int,
) -> np.ndarray:
    if dispersion_size <= 0 or not math.isfinite(dispersion_size):
        raise ValueError("dispersion_size must be finite and positive")
    n_obs = len(counts)
    prefix_events = np.r_[0.0, np.cumsum(counts, dtype=float)]
    base = (
        gammaln(counts + dispersion_size)
        - gammaln(dispersion_size)
        - gammaln(counts + 1)
    )
    prefix_base = np.r_[0.0, np.cumsum(base, dtype=float)]
    costs = np.full((n_obs + 1, n_obs + 1), np.inf, dtype=float)
    log_size = math.log(dispersion_size)

    for start in range(n_obs):
        ends = np.arange(start + min_bins, n_obs + 1)
        if ends.size == 0:
            continue
        lengths = ends - start
        totals = prefix_events[ends] - prefix_events[start]
        valid = totals >= min_events
        if not np.any(valid):
            continue
        valid_ends = ends[valid]
        valid_lengths = lengths[valid].astype(float)
        valid_totals = totals[valid]
        means = valid_totals / valid_lengths
        log_denom = np.log(dispersion_size + means)
        log_likelihood = (
            prefix_base[valid_ends]
            - prefix_base[start]
            + valid_lengths * dispersion_size * (log_size - log_denom)
            + xlogy(valid_totals, means / (dispersion_size + means))
        )
        costs[start, valid_ends] = -2.0 * log_likelihood
    return costs


def dynamic_program(
    costs: np.ndarray, max_phases: int, min_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    n_obs = costs.shape[0] - 1
    scores = np.full((max_phases + 1, n_obs + 1), np.inf, dtype=float)
    previous = np.full((max_phases + 1, n_obs + 1), -1, dtype=int)
    scores[0, 0] = 0.0
    for k in range(1, max_phases + 1):
        for end in range(k * min_bins, n_obs + 1):
            starts = np.arange((k - 1) * min_bins, end - min_bins + 1)
            candidates = scores[k - 1, starts] + costs[starts, end]
            best_position = int(np.argmin(candidates))
            best_score = float(candidates[best_position])
            if math.isfinite(best_score):
                scores[k, end] = best_score
                previous[k, end] = int(starts[best_position])
    return scores, previous


def backtrack_segments(
    previous: np.ndarray, k: int, n_obs: int
) -> tuple[tuple[int, int], ...]:
    segments: list[tuple[int, int]] = []
    end = n_obs
    for phase in range(k, 0, -1):
        start = int(previous[phase, end])
        if start < 0:
            raise ValueError(f"Backtracking failed for K={k}")
        segments.append((start, end))
        end = start
    if end != 0:
        raise ValueError(f"Backtracking did not reach the start for K={k}")
    return tuple(reversed(segments))


class ProfileEvaluator:
    def __init__(
        self, counts: np.ndarray, max_phases: int, min_bins: int, min_events: int
    ) -> None:
        self.counts = counts
        self.max_phases = max_phases
        self.min_bins = min_bins
        self.min_events = min_events
        self.cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(self, log_size: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(log_size), 12)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        size = math.exp(float(log_size))
        costs = negative_binomial_cost_matrix(
            self.counts, size, self.min_bins, self.min_events
        )
        result = dynamic_program(costs, self.max_phases, self.min_bins)
        self.cache[key] = result
        return result

    def score(self, log_size: float, k: int) -> float:
        scores, _ = self.evaluate(log_size)
        return float(scores[k, len(self.counts)])


def fit_negative_binomial_models(
    counts: np.ndarray,
    max_phases: int,
    min_bins: int,
    min_events: int,
    grid_points: int,
    log_size_bounds: tuple[float, float],
) -> list[SegmentationFit]:
    evaluator = ProfileEvaluator(counts, max_phases, min_bins, min_events)
    grid = np.linspace(log_size_bounds[0], log_size_bounds[1], grid_points)
    grid_scores = np.full((grid_points, max_phases + 1), np.inf, dtype=float)
    for index, log_size in enumerate(grid):
        scores, _ = evaluator.evaluate(float(log_size))
        grid_scores[index, :] = scores[:, len(counts)]

    fits: list[SegmentationFit] = []
    for k in range(1, max_phases + 1):
        column = grid_scores[:, k]
        if not np.any(np.isfinite(column)):
            continue
        best_grid = int(np.nanargmin(column))
        left = grid[max(0, best_grid - 1)]
        right = grid[min(grid_points - 1, best_grid + 1)]
        if left == right:
            optimum_log_size = float(grid[best_grid])
        else:
            result = minimize_scalar(
                lambda value: evaluator.score(float(value), k),
                bounds=(float(left), float(right)),
                method="bounded",
                options={"xatol": 1e-7},
            )
            optimum_log_size = float(result.x)
        score = evaluator.score(optimum_log_size, k)
        _, previous = evaluator.evaluate(optimum_log_size)
        segments = backtrack_segments(previous, k, len(counts))
        parameter_count = 2 * k
        bic = score + parameter_count * math.log(max(len(counts), 2))
        fits.append(
            SegmentationFit(
                k=k,
                score=score,
                bic=float(bic),
                dispersion_size=math.exp(optimum_log_size),
                segments=segments,
            )
        )
    return fits


def fits_to_frame(fits: list[SegmentationFit]) -> pd.DataFrame:
    if not fits:
        raise ValueError("No feasible negative-binomial models")
    minimum_bic = min(fit.bic for fit in fits)
    rows = []
    for fit in fits:
        rows.append(
            {
                "k": fit.k,
                "negative_2_log_likelihood": fit.score,
                "bic": fit.bic,
                "delta_bic": fit.bic - minimum_bic,
                "dispersion_size": fit.dispersion_size,
                "dispersion_alpha": 1.0 / fit.dispersion_size,
                "bic_minimum": fit.bic == minimum_bic,
            }
        )
    return pd.DataFrame(rows)


def segments_to_frame(fit: SegmentationFit, bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase_id, (start, end) in enumerate(fit.segments, start=1):
        segment = bins.iloc[start:end]
        events = int(segment["n_retweets"].sum())
        duration = int(segment["duration_hours"].sum())
        rows.append(
            {
                "phase_id": phase_id,
                "start_bin_idx": start,
                "end_bin_idx_exclusive": end,
                "start_utc": segment["bin_start_utc"].iloc[0],
                "end_utc": segment["bin_end_utc"].iloc[-1],
                "duration_hours": duration,
                "n_events": events,
                "mean_retweets_per_hour": events / duration,
            }
        )
    return pd.DataFrame(rows)


def internal_boundaries(phases: pd.DataFrame) -> list[pd.Timestamp]:
    return list(pd.to_datetime(phases["end_utc"].iloc[:-1], utc=True))


def compare_equal_k_boundaries(
    baseline: pd.DataFrame,
    hourly_nb: pd.DataFrame,
    aggregated_nb: pd.DataFrame,
) -> pd.DataFrame:
    baseline_boundaries = internal_boundaries(baseline)
    hourly_boundaries = internal_boundaries(hourly_nb)
    aggregated_boundaries = internal_boundaries(aggregated_nb)
    if not (
        len(baseline_boundaries) == len(hourly_boundaries) == len(aggregated_boundaries)
    ):
        raise ValueError(
            "Equal-K boundary comparison requires the same number of phases"
        )
    rows = []
    for index, (base, hourly, aggregated) in enumerate(
        zip(baseline_boundaries, hourly_boundaries, aggregated_boundaries, strict=True),
        start=1,
    ):
        label = (
            TRANSITION_LABELS[index - 1]
            if index <= len(TRANSITION_LABELS)
            else f"boundary {index}"
        )
        rows.append(
            {
                "boundary_id": index,
                "transition": label,
                "poisson_hourly_utc": base,
                "nb_hourly_utc": hourly,
                "nb_hourly_difference_hours": (hourly - base).total_seconds() / 3600.0,
                "nb_hourly_absolute_difference_hours": abs(
                    (hourly - base).total_seconds()
                )
                / 3600.0,
                "nb_aggregated_utc": aggregated,
                "nb_aggregated_difference_hours": (aggregated - base).total_seconds()
                / 3600.0,
                "nb_aggregated_absolute_difference_hours": abs(
                    (aggregated - base).total_seconds()
                )
                / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def nearest_boundary_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_name: str,
) -> pd.DataFrame:
    candidate_boundaries = internal_boundaries(candidate)
    rows = []
    for index, baseline_boundary in enumerate(internal_boundaries(baseline), start=1):
        nearest = min(
            candidate_boundaries, key=lambda value: abs(value - baseline_boundary)
        )
        difference = (nearest - baseline_boundary).total_seconds() / 3600.0
        rows.append(
            {
                "boundary_id": index,
                "transition": TRANSITION_LABELS[index - 1],
                "poisson_hourly_utc": baseline_boundary,
                "candidate": candidate_name,
                "nearest_candidate_utc": nearest,
                "difference_hours": difference,
                "absolute_difference_hours": abs(difference),
            }
        )
    return pd.DataFrame(rows)


def lag_one_correlation(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def phase_diagnostics(
    phases: pd.DataFrame, bins: pd.DataFrame, model: str
) -> pd.DataFrame:
    rows = []
    for phase in phases.itertuples(index=False):
        segment = bins.iloc[int(phase.start_bin_idx) : int(phase.end_bin_idx_exclusive)]
        counts = segment["n_retweets"].to_numpy(dtype=float)
        mean = float(np.mean(counts))
        variance = float(np.var(counts, ddof=1)) if len(counts) > 1 else float("nan")
        rows.append(
            {
                "model": model,
                "phase_id": int(phase.phase_id),
                "n_bins": int(len(counts)),
                "bin_hours": int(segment["duration_hours"].iloc[0]),
                "mean_count_per_bin": mean,
                "variance_count_per_bin": variance,
                "variance_to_mean": variance / mean if mean > 0 else float("nan"),
                "lag1_correlation": lag_one_correlation(counts),
            }
        )
    return pd.DataFrame(rows)


def boundary_summary(comparison: pd.DataFrame, column: str) -> dict[str, float | int]:
    values = comparison[column].to_numpy(dtype=float)
    return {
        "median_absolute_difference_hours": float(np.median(values)),
        "maximum_absolute_difference_hours": float(np.max(values)),
        "within_3_hours": int(np.sum(values <= 3)),
        "within_6_hours": int(np.sum(values <= 6)),
        "within_12_hours": int(np.sum(values <= 12)),
        "n_boundaries": int(len(values)),
    }


def plot_boundary_sensitivity(
    hourly: pd.DataFrame,
    phase_frames: list[tuple[str, pd.DataFrame, str]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(phase_frames), 1, figsize=(13, 8.5), sharex=True, constrained_layout=True
    )
    if len(phase_frames) == 1:
        axes = [axes]
    for ax, (label, phases, color) in zip(axes, phase_frames, strict=True):
        ax.plot(
            hourly["hour_start_utc"],
            hourly["n_retweets"] + 1,
            color="#7f8c8d",
            linewidth=0.8,
        )
        for boundary in internal_boundaries(phases):
            ax.axvline(boundary, color=color, linewidth=1.3, alpha=0.9)
        ax.set_yscale("log")
        ax.set_ylabel("Retweets/h + 1")
        ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Time (UTC)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    out_path: Path,
    args: argparse.Namespace,
    hourly_path: Path,
    baseline_path: Path,
    aggregation_qa: dict[str, int],
) -> None:
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "hourly_counts": str(hourly_path),
            "hourly_counts_sha256": sha256(hourly_path),
            "baseline_boundaries": str(baseline_path),
            "baseline_boundaries_sha256": sha256(baseline_path),
        },
        "parameters": {
            "max_phases": args.max_phases,
            "retained_phases": args.retained_phases,
            "min_phase_hours": args.min_phase_hours,
            "min_phase_events": args.min_phase_events,
            "aggregate_hours": args.aggregate_hours,
            "shared_dispersion": True,
            "negative_binomial_parameterization": "variance = mean + mean^2 / dispersion_size",
            "profile_log_size_bounds": [args.log_size_min, args.log_size_max],
            "profile_grid_points": args.profile_grid_points,
        },
        "aggregation_qa": aggregation_qa,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 2) -> str:
    def format_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(format_cell(value) for value in row) + " |"
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    out_path: Path,
    hourly_models: pd.DataFrame,
    aggregated_models: pd.DataFrame,
    hourly_k8: pd.DataFrame,
    aggregated_k8: pd.DataFrame,
    equal_k_comparison: pd.DataFrame,
    optimal_comparison: pd.DataFrame,
    aggregation_qa: dict[str, int],
    aggregate_hours: int,
) -> None:
    hourly_optimum = int(hourly_models.loc[hourly_models["bic"].idxmin(), "k"])
    aggregated_optimum = int(
        aggregated_models.loc[aggregated_models["bic"].idxmin(), "k"]
    )
    hourly_summary = boundary_summary(
        equal_k_comparison, "nb_hourly_absolute_difference_hours"
    )
    aggregated_summary = boundary_summary(
        equal_k_comparison, "nb_aggregated_absolute_difference_hours"
    )
    aggregated_optimum_rows = optimal_comparison[
        optimal_comparison["candidate"].str.startswith(f"NB {aggregate_hours}-hour")
    ]
    aggregated_optimum_summary = boundary_summary(
        aggregated_optimum_rows,
        "absolute_difference_hours",
    )
    unsupported_transitions = aggregated_optimum_rows.loc[
        aggregated_optimum_rows["absolute_difference_hours"] > 12,
        "transition",
    ].tolist()
    ordered_aggregated = aggregated_models.sort_values("bic").reset_index(drop=True)
    second_best_delta = float(ordered_aggregated.loc[1, "delta_bic"])
    lines = [
        "# Retweet-phase robustness analysis",
        "",
        "## Design",
        "",
        "The primary analysis uses the BIC-minimizing piecewise-constant negative-binomial model with a shared dispersion parameter fitted to complete non-overlapping three-hour bins. Hourly negative-binomial and archived hourly Poisson segmentations provide temporal-resolution and distributional sensitivity checks. All models use the same minimum phase-duration and event-count constraints. The incomplete trailing three-hour bin is excluded and reported below.",
        "",
        "## Model-selection summary",
        "",
        f"- Hourly negative-binomial BIC minimum: K={hourly_optimum}.",
        f"- Three-hour negative-binomial BIC minimum: K={aggregated_optimum}.",
        f"- The second-best three-hour solution differed from the minimum by only {second_best_delta:.3f} BIC units.",
        f"- Three-hour aggregation retained {aggregation_qa['retained_events']} of {aggregation_qa['input_events']} retweets and excluded {aggregation_qa['dropped_trailing_events']} events in {aggregation_qa['dropped_trailing_hours']} incomplete trailing hours.",
        "",
        "### Hourly negative-binomial candidates",
        "",
        markdown_table(
            hourly_models,
            [
                "k",
                "bic",
                "delta_bic",
                "dispersion_size",
                "dispersion_alpha",
                "bic_minimum",
            ],
            digits=3,
        ),
        "",
        "### Three-hour negative-binomial candidates",
        "",
        markdown_table(
            aggregated_models,
            [
                "k",
                "bic",
                "delta_bic",
                "dispersion_size",
                "dispersion_alpha",
                "bic_minimum",
            ],
            digits=3,
        ),
        "",
        "## Fixed eight-phase sensitivity comparison",
        "",
        f"The hourly negative-binomial boundaries differed from the archived hourly Poisson boundaries by a median of {hourly_summary['median_absolute_difference_hours']:.1f} hours and a maximum of {hourly_summary['maximum_absolute_difference_hours']:.1f} hours. {hourly_summary['within_12_hours']} of {hourly_summary['n_boundaries']} boundaries were within 12 hours.",
        "",
        f"The three-hour negative-binomial boundaries differed by a median of {aggregated_summary['median_absolute_difference_hours']:.1f} hours and a maximum of {aggregated_summary['maximum_absolute_difference_hours']:.1f} hours. {aggregated_summary['within_12_hours']} of {aggregated_summary['n_boundaries']} boundaries were within 12 hours.",
        "",
        markdown_table(
            equal_k_comparison,
            [
                "boundary_id",
                "transition",
                "nb_hourly_difference_hours",
                "nb_aggregated_difference_hours",
            ],
            digits=1,
        ),
        "",
        "### Hourly negative-binomial eight-phase sensitivity",
        "",
        markdown_table(
            hourly_k8,
            [
                "phase_id",
                "start_utc",
                "end_utc",
                "duration_hours",
                "n_events",
                "mean_retweets_per_hour",
            ],
            digits=2,
        ),
        "",
        "### Three-hour negative-binomial eight-phase sensitivity",
        "",
        markdown_table(
            aggregated_k8,
            [
                "phase_id",
                "start_utc",
                "end_utc",
                "duration_hours",
                "n_events",
                "mean_retweets_per_hour",
            ],
            digits=2,
        ),
        "",
        "## Numerical-optimum boundary sensitivity",
        "",
        "Each archived hourly Poisson boundary is paired with the nearest boundary from the BIC-minimizing negative-binomial solution. This comparison assesses whether the principal lifecycle transitions remain represented when the phase count varies.",
        "",
        markdown_table(
            optimal_comparison,
            [
                "boundary_id",
                "transition",
                "candidate",
                "difference_hours",
                "absolute_difference_hours",
            ],
            digits=1,
        ),
        "",
        f"The BIC-minimizing three-hour solution retained a boundary within 12 hours for {aggregated_optimum_summary['within_12_hours']} of {aggregated_optimum_summary['n_boundaries']} transitions. It did not retain separate boundaries for: {', '.join(unsupported_transitions)}.",
        "",
        "## Interpretation rule",
        "",
        "The six-phase three-hour negative-binomial solution is the primary partition. The broad lifecycle interpretation is considered robust when emergence, the viral peak, contraction, dormancy, late reactivation, and residual afterlife remain represented across the hourly sensitivity models. Differences in exact boundary placement and the effective tie between six and seven three-hour phases limit claims about a unique fine-grained partition.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_fit(fits: list[SegmentationFit], k: int) -> SegmentationFit:
    for fit in fits:
        if fit.k == k:
            return fit
    raise ValueError(f"No feasible model for K={k}")


def format_timestamp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in [name for name in output.columns if name.endswith("_utc")]:
        output[column] = pd.to_datetime(output[column], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    return output


def main() -> None:
    args = parse_args()
    validate_args(args)
    hourly_path = Path(args.hourly)
    baseline_path = Path(args.baseline_boundaries)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hourly = load_hourly_counts(hourly_path)
    baseline = load_baseline_boundaries(baseline_path)
    if len(baseline) != args.retained_phases:
        raise ValueError("Baseline boundary count does not match --retained-phases")
    hourly_bins = hourly_as_bins(hourly)
    aggregated_bins, aggregation_qa = aggregate_complete_bins(
        hourly, args.aggregate_hours
    )
    log_size_bounds = (args.log_size_min, args.log_size_max)

    hourly_fits = fit_negative_binomial_models(
        hourly_bins["n_retweets"].to_numpy(dtype=int),
        args.max_phases,
        args.min_phase_hours,
        args.min_phase_events,
        args.profile_grid_points,
        log_size_bounds,
    )
    aggregated_fits = fit_negative_binomial_models(
        aggregated_bins["n_retweets"].to_numpy(dtype=int),
        args.max_phases,
        math.ceil(args.min_phase_hours / args.aggregate_hours),
        args.min_phase_events,
        args.profile_grid_points,
        log_size_bounds,
    )

    hourly_models = fits_to_frame(hourly_fits)
    aggregated_models = fits_to_frame(aggregated_fits)
    hourly_k8 = segments_to_frame(
        select_fit(hourly_fits, args.retained_phases), hourly_bins
    )
    aggregated_k8 = segments_to_frame(
        select_fit(aggregated_fits, args.retained_phases), aggregated_bins
    )
    hourly_optimum_fit = min(hourly_fits, key=lambda fit: fit.bic)
    aggregated_optimum_fit = min(aggregated_fits, key=lambda fit: fit.bic)
    hourly_optimum = segments_to_frame(hourly_optimum_fit, hourly_bins)
    aggregated_optimum = segments_to_frame(aggregated_optimum_fit, aggregated_bins)

    equal_k_comparison = compare_equal_k_boundaries(baseline, hourly_k8, aggregated_k8)
    optimal_comparison = pd.concat(
        [
            nearest_boundary_comparison(
                baseline, hourly_optimum, f"NB hourly K={hourly_optimum_fit.k}"
            ),
            nearest_boundary_comparison(
                baseline,
                aggregated_optimum,
                f"NB {args.aggregate_hours}-hour K={aggregated_optimum_fit.k}",
            ),
        ],
        ignore_index=True,
    )
    diagnostics = pd.concat(
        [
            phase_diagnostics(hourly_k8, hourly_bins, "NB hourly K=8"),
            phase_diagnostics(
                aggregated_k8, aggregated_bins, f"NB {args.aggregate_hours}-hour K=8"
            ),
        ],
        ignore_index=True,
    )

    hourly_models.to_csv(outdir / "nb_hourly_model_selection.csv", index=False)
    aggregated_models.to_csv(outdir / "nb_3hour_model_selection.csv", index=False)
    format_timestamp_columns(hourly_k8).to_csv(
        outdir / "nb_hourly_phase_boundaries_k8.csv", index=False
    )
    format_timestamp_columns(aggregated_k8).to_csv(
        outdir / "nb_3hour_phase_boundaries_k8.csv", index=False
    )
    format_timestamp_columns(hourly_optimum).to_csv(
        outdir / "nb_hourly_phase_boundaries_bic_minimum.csv", index=False
    )
    format_timestamp_columns(aggregated_optimum).to_csv(
        outdir / "nb_3hour_phase_boundaries_bic_minimum.csv", index=False
    )
    format_timestamp_columns(equal_k_comparison).to_csv(
        outdir / "boundary_comparison_k8.csv", index=False
    )
    format_timestamp_columns(optimal_comparison).to_csv(
        outdir / "principal_boundary_comparison_bic_minimum.csv", index=False
    )
    diagnostics.to_csv(outdir / "phase_diagnostics.csv", index=False)

    plot_boundary_sensitivity(
        hourly,
        [
            ("Archived hourly Poisson sensitivity (K=8)", baseline, "#355070"),
            ("Hourly negative-binomial segmentation (K=8)", hourly_k8, "#b56576"),
            (
                f"{args.aggregate_hours}-hour negative-binomial segmentation (K=8)",
                aggregated_k8,
                "#2a9d8f",
            ),
            (
                f"{args.aggregate_hours}-hour negative-binomial BIC minimum (K={aggregated_optimum_fit.k})",
                aggregated_optimum,
                "#e76f51",
            ),
        ],
        outdir / "fig_boundary_sensitivity.png",
    )
    write_manifest(
        outdir / "run_manifest.json", args, hourly_path, baseline_path, aggregation_qa
    )
    write_report(
        outdir / "report.md",
        hourly_models,
        aggregated_models,
        hourly_k8,
        aggregated_k8,
        equal_k_comparison,
        optimal_comparison,
        aggregation_qa,
        args.aggregate_hours,
    )


if __name__ == "__main__":
    main()
