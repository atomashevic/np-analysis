"""Compare institutional accounts to regular users and phase 1 authors by phase.

This script refocuses the institutional analysis on one primary question:
for each phase from phase_2 onward, how do pooled institutional accounts
differ from (a) all regular users and (b) the phase 1 author cohort on a
small set of network metrics?

Institutional accounts are loaded from the combined classification file
(results/users_institution_final.jsonl) which merges the hand-curated
registry, GPT-based classification, and web-search verification.

Primary outputs:
  analysis/institutional_accounts/
    account_registry.csv          - institutional account registry
    phase_metric_comparison.csv   - one canonical phase x metric comparison table
    report.md                     - short human-readable summary of the table

Legacy comparison-heavy outputs from earlier iterations are removed on each
run so the directory stays focused.
"""

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PHASE_DIR = ROOT / "analysis" / "retweet_network_phases"
CLASSIFICATION_FILE = (
    ROOT / "results" / "users_institution" / "users_institution_final.jsonl"
)
OUT_DIR = ROOT / "analysis" / "institutional_accounts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_OUTPUTS = [
    "comparison_inst_vs_phase1_cohort.csv",
    "comparison_inst_vs_regular.csv",
    "phase_instrength_share.csv",
    "phase_node_positions.csv",
    "phase_retweet_share.csv",
    "phase_strength_share.csv",
    "phase_summary.csv",
]

METRICS = [
    ("in_strength", "Retweets received", 1),
    ("pagerank", "PageRank", 2),
    ("out_strength", "Retweets made", 3),
    ("in_degree", "Unique retweeters", 4),
]


def load_institutional_accounts():
    """Load institutional accounts from the combined classification file.

    Returns a dict: username -> (category, reason) for non-personal accounts.
    """
    accounts = {}
    with open(CLASSIFICATION_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["institution_type"] != "personal":
                accounts[record["user"]] = (
                    record["institution_type"],
                    record.get("reason", ""),
                )
    return accounts


def load_nodes():
    nodes = {}
    with open(DATA / "np_nodes.csv") as handle:
        for row in csv.DictReader(handle):
            nodes[row["username"]] = row
    return nodes


def load_phase_boundaries():
    phases = []
    with open(PHASE_DIR / "phase_boundaries.csv") as handle:
        for row in csv.DictReader(handle):
            phases.append(
                {
                    "phase_id": int(row["phase_id"]),
                    "label": row["phase_label"],
                    "start": datetime.fromisoformat(row["start_utc"]),
                    "end": datetime.fromisoformat(row["end_utc"]),
                    "n_events": int(row["n_events"]),
                }
            )
    return phases


def load_tweets():
    tweets = []
    with open(DATA / "np_without_duplicates.csv") as handle:
        for row in csv.DictReader(handle):
            tweets.append(
                {
                    "id": row["id_str"],
                    "username": row["from_user"],
                    "time": row["time"],
                    "post_type": row["post_type"],
                    "parent_id": row.get("parent_id_str", ""),
                }
            )
    return tweets


def load_phase_node_metrics():
    metrics = {}
    with open(PHASE_DIR / "phase_node_metrics.csv") as handle:
        for row in csv.DictReader(handle):
            metrics[(int(row["phase_id"]), row["username"])] = {
                "user_id": str(row.get("user_id", "")).strip(),
                "in_degree": int(row["in_degree"]),
                "out_degree": int(row["out_degree"]),
                "in_strength": float(row["in_strength"]),
                "out_strength": float(row["out_strength"]),
                "total_strength": float(row["total_strength"]),
            }
    return metrics


def load_phase1_cohort_usernames(phase_node):
    cohort_user_ids = set()
    with open(PHASE_DIR / "first_phase_author_cohort.csv") as handle:
        for row in csv.DictReader(handle):
            cohort_user_ids.add(str(row["user_id"]).strip())

    cohort_usernames = {
        username
        for (phase_id, username), metrics in phase_node.items()
        if metrics.get("user_id", "") in cohort_user_ids
    }
    return cohort_user_ids, cohort_usernames


def assign_phase(time_str, phases):
    try:
        timestamp = datetime.fromisoformat(time_str).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

    for phase in phases:
        if phase["start"] <= timestamp < phase["end"]:
            return phase["phase_id"]
    return None


def build_phase_graphs(tweets, phases):
    post_author = {
        tweet["id"]: tweet["username"]
        for tweet in tweets
        if tweet["post_type"] == "post"
    }

    phase_edge_weights = defaultdict(lambda: defaultdict(int))
    for tweet in tweets:
        if tweet["post_type"] != "retweet":
            continue
        phase_id = assign_phase(tweet["time"], phases)
        if phase_id is None:
            continue
        retweeted = post_author.get(tweet["parent_id"])
        if retweeted and retweeted != tweet["username"]:
            phase_edge_weights[phase_id][(tweet["username"], retweeted)] += 1

    graphs = {}
    for phase_id, edge_weights in phase_edge_weights.items():
        graph = nx.DiGraph()
        for (source, target), weight in edge_weights.items():
            graph.add_edge(source, target, weight=weight)
        graphs[phase_id] = graph
    return graphs


def compute_phase_pageranks(graphs):
    pageranks = {}
    for phase_id in sorted(graphs):
        graph = graphs[phase_id]
        try:
            pageranks[phase_id] = nx.pagerank(
                graph, weight="weight", max_iter=500, tol=1e-6
            )
        except nx.PowerIterationFailedConvergence:
            pageranks[phase_id] = nx.pagerank(
                graph, weight=None, max_iter=200, tol=1e-4
            )
    return pageranks


def compute_percentile(value, all_values):
    if not all_values:
        return float("nan")
    return 100.0 * sum(1 for current in all_values if current < value) / len(all_values)


def safe_round(value, digits):
    return round(value, digits) if value == value else float("nan")


def safe_ratio(numerator, denominator):
    if numerator != numerator or denominator != denominator or denominator == 0:
        return float("nan")
    return numerator / denominator


def summarize_group(values, all_values):
    clean_values = [float(value) for value in values if value == value]
    if not clean_values:
        return {
            "n": 0,
            "median_value": float("nan"),
            "mean_value": float("nan"),
            "median_percentile": float("nan"),
        }

    percentiles = [compute_percentile(value, all_values) for value in clean_values]
    return {
        "n": len(clean_values),
        "median_value": float(np.median(clean_values)),
        "mean_value": float(np.mean(clean_values)),
        "median_percentile": float(np.median(percentiles)),
    }


def format_number(value, digits=2):
    if value != value:
        return "N/A"
    return f"{value:.{digits}f}"


def prune_legacy_outputs():
    for filename in LEGACY_OUTPUTS:
        path = OUT_DIR / filename
        if path.exists():
            path.unlink()


def build_registry(nodes, accounts):
    rows = []
    present_usernames = set()
    for username, (category, description) in accounts.items():
        node = nodes.get(username, {})
        in_dataset = "yes" if username in nodes else "no"
        rows.append(
            {
                "username": username,
                "category": category,
                "description": description,
                "user_id": node.get("user_id", ""),
                "in_dataset": in_dataset,
                "followers_count": node.get("followers_count", ""),
                "n_posts": node.get("n_posts", ""),
                "n_retweets": node.get("n_retweets", ""),
                "n_total": node.get("n_total", ""),
            }
        )
        if in_dataset == "yes":
            present_usernames.add(username)

    with open(OUT_DIR / "account_registry.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda row: (row["category"], row["username"]))
        )

    return present_usernames, rows


def build_phase_metric_rows(
    phases, phase_node, phase_pr, institutional_usernames, phase1_usernames
):
    rows = []

    for phase_info in sorted(phases, key=lambda phase: phase["phase_id"]):
        phase_id = phase_info["phase_id"]
        if phase_id == 1:
            continue

        phase_users = {}
        for (current_phase_id, username), metrics in phase_node.items():
            if current_phase_id != phase_id:
                continue
            phase_users[username] = {
                "in_strength": metrics["in_strength"],
                "out_strength": metrics["out_strength"],
                "in_degree": metrics["in_degree"],
                "pagerank": phase_pr.get(phase_id, {}).get(username, float("nan")),
            }

        if not phase_users:
            continue

        regular_usernames = {
            username
            for username in phase_users
            if username not in institutional_usernames
        }
        phase1_present = {
            username for username in phase_users if username in phase1_usernames
        }

        for metric_key, metric_label, metric_order in METRICS:
            all_values = [
                metrics[metric_key]
                for metrics in phase_users.values()
                if metrics[metric_key] == metrics[metric_key]
            ]

            institutional_values = [
                phase_users[username][metric_key]
                for username in phase_users
                if username in institutional_usernames
            ]
            regular_values = [
                phase_users[username][metric_key]
                for username in phase_users
                if username in regular_usernames
            ]
            phase1_values = [
                phase_users[username][metric_key]
                for username in phase_users
                if username in phase1_present
            ]

            institutional_summary = summarize_group(institutional_values, all_values)
            regular_summary = summarize_group(regular_values, all_values)
            phase1_summary = summarize_group(phase1_values, all_values)

            rows.append(
                {
                    "phase_id": phase_id,
                    "phase_label": phase_info["label"],
                    "metric_order": metric_order,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "n_institutional": institutional_summary["n"],
                    "institutional_median_value": safe_round(
                        institutional_summary["median_value"], 6
                    ),
                    "institutional_mean_value": safe_round(
                        institutional_summary["mean_value"], 6
                    ),
                    "institutional_median_percentile": safe_round(
                        institutional_summary["median_percentile"], 2
                    ),
                    "n_regular": regular_summary["n"],
                    "regular_median_value": safe_round(
                        regular_summary["median_value"], 6
                    ),
                    "regular_mean_value": safe_round(regular_summary["mean_value"], 6),
                    "regular_median_percentile": safe_round(
                        regular_summary["median_percentile"], 2
                    ),
                    "n_phase1_authors": phase1_summary["n"],
                    "phase1_median_value": safe_round(
                        phase1_summary["median_value"], 6
                    ),
                    "phase1_mean_value": safe_round(phase1_summary["mean_value"], 6),
                    "phase1_median_percentile": safe_round(
                        phase1_summary["median_percentile"], 2
                    ),
                    "institutional_minus_regular_percentile": safe_round(
                        institutional_summary["median_percentile"]
                        - regular_summary["median_percentile"],
                        2,
                    ),
                    "institutional_minus_phase1_percentile": safe_round(
                        institutional_summary["median_percentile"]
                        - phase1_summary["median_percentile"],
                        2,
                    ),
                    "institutional_vs_regular_mean_ratio": safe_round(
                        safe_ratio(
                            institutional_summary["mean_value"],
                            regular_summary["mean_value"],
                        ),
                        4,
                    ),
                    "institutional_vs_phase1_mean_ratio": safe_round(
                        safe_ratio(
                            institutional_summary["mean_value"],
                            phase1_summary["mean_value"],
                        ),
                        4,
                    ),
                }
            )

    return rows


def build_report(registry_rows, comparison_rows):
    category_counts = defaultdict(int)
    in_dataset_count = 0
    for row in registry_rows:
        category_counts[row["category"]] += 1
        if row["in_dataset"] == "yes":
            in_dataset_count += 1

    phase_rows = defaultdict(list)
    for row in comparison_rows:
        phase_rows[row["phase_id"]].append(row)

    report = []
    report.append("# Institutional Accounts: Focused Phase Comparison\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    report.append("## Registry\n")
    report.append(f"- Media: {category_counts['media']}")
    report.append(f"- Political: {category_counts['political']}")
    report.append(f"- NGO: {category_counts['ngo']}")
    report.append(f"- Total classified: {sum(category_counts.values())}")
    report.append(f"- Found in dataset: {in_dataset_count}\n")

    report.append("## Main Comparison Table\n")
    report.append(
        "Institutional accounts are pooled and compared to all regular users "
        "(non-institutional participants, including phase 1 authors) and to the "
        "phase 1 author cohort separately. The main summary uses each group's "
        "median percentile within the full phase distribution for the metric. "
        "Mean ratios are included to catch phases where only a few institutional "
        "accounts carry the advantage.\n"
    )
    report.append(
        "| Phase | Metric | Inst med %-ile | Regular med %-ile | Phase1 med %-ile | "
        "Inst-Reg gap | Inst/Reg mean ratio | Inst/Phase1 mean ratio |"
    )
    report.append(
        "|-------|--------|---------------:|------------------:|------------------:|-------------:|--------------------:|----------------------:|"
    )

    for row in sorted(
        comparison_rows, key=lambda item: (item["phase_id"], item["metric_order"])
    ):
        report.append(
            f"| {row['phase_label']} | {row['metric_label']} | "
            f"{format_number(row['institutional_median_percentile'], 2)} | "
            f"{format_number(row['regular_median_percentile'], 2)} | "
            f"{format_number(row['phase1_median_percentile'], 2)} | "
            f"{format_number(row['institutional_minus_regular_percentile'], 2)} | "
            f"{format_number(row['institutional_vs_regular_mean_ratio'], 2)} | "
            f"{format_number(row['institutional_vs_phase1_mean_ratio'], 2)} |"
        )

    report.append("\n## Phase Highlights\n")
    for phase_id in sorted(phase_rows):
        rows = sorted(phase_rows[phase_id], key=lambda item: item["metric_order"])
        inst_n = rows[0]["n_institutional"] if rows else 0
        phase_label = rows[0]["phase_label"] if rows else f"phase_{phase_id}"
        wins_vs_regular = sum(
            row["institutional_minus_regular_percentile"] > 0 for row in rows
        )
        wins_vs_phase1 = sum(
            row["institutional_minus_phase1_percentile"] > 0 for row in rows
        )
        best_metric = max(
            rows, key=lambda item: item["institutional_minus_regular_percentile"]
        )
        weakest_metric = min(
            rows, key=lambda item: item["institutional_minus_phase1_percentile"]
        )
        best_ratio = max(
            rows,
            key=lambda item: (
                item["institutional_vs_regular_mean_ratio"]
                if item["institutional_vs_regular_mean_ratio"]
                == item["institutional_vs_regular_mean_ratio"]
                else float("-inf")
            ),
        )

        note = " Low-N phase; read descriptively." if inst_n < 3 else ""
        if (
            wins_vs_regular == 0
            and best_ratio["institutional_vs_regular_mean_ratio"] > 1
        ):
            report.append(
                f"- **{phase_label}:** institutional medians do not exceed regular-user medians on any metric, "
                f"but the upper tail still matters: {best_ratio['metric_label']} is "
                f"{format_number(best_ratio['institutional_vs_regular_mean_ratio'], 2)}x the regular-user mean. "
                f"Phase 1 authors remain ahead overall, with the biggest institutional deficit on "
                f"{weakest_metric['metric_label']} "
                f"({format_number(weakest_metric['institutional_minus_phase1_percentile'], 2)} pts)."
                f"{note}"
            )
        else:
            report.append(
                f"- **{phase_label}:** institutions lead regular users on {wins_vs_regular}/4 metrics "
                f"and phase 1 authors on {wins_vs_phase1}/4. Strongest edge vs regular users: "
                f"{best_metric['metric_label']} ({format_number(best_metric['institutional_minus_regular_percentile'], 2)} pts). "
                f"Largest mean-ratio edge vs regular users: {best_ratio['metric_label']} "
                f"({format_number(best_ratio['institutional_vs_regular_mean_ratio'], 2)}x). "
                f"Weakest position vs phase 1 authors: {weakest_metric['metric_label']} "
                f"({format_number(weakest_metric['institutional_minus_phase1_percentile'], 2)} pts)."
                f"{note}"
            )

    report.append("\n## Notes\n")
    report.append(
        "- `phase_1` is kept as historical context only; the repeated comparison starts at `phase_2`."
    )
    report.append(
        "- Late phases with `n_institutional < 3` are descriptive snapshots, not stable group comparisons."
    )
    report.append(
        "- The hub-evolution outputs remain available as appendix material, but they are no longer the main analytical surface."
    )

    with open(OUT_DIR / "report.md", "w") as handle:
        handle.write("\n".join(report) + "\n")


def main():
    print("Loading source data...")
    nodes = load_nodes()
    phases = load_phase_boundaries()
    tweets = load_tweets()
    phase_node = load_phase_node_metrics()

    print(f"Loading institutional classification from {CLASSIFICATION_FILE}...")
    accounts = load_institutional_accounts()
    print(f"  {len(accounts)} institutional accounts loaded")

    prune_legacy_outputs()

    print("Writing registry...")
    institutional_usernames, registry_rows = build_registry(nodes, accounts)

    print("Computing PageRank by phase...")
    phase_graphs = build_phase_graphs(tweets, phases)
    phase_pr = compute_phase_pageranks(phase_graphs)

    print("Resolving phase 1 author cohort...")
    phase1_user_ids, phase1_usernames = load_phase1_cohort_usernames(phase_node)

    print("Building canonical comparison table...")
    comparison_rows = build_phase_metric_rows(
        phases=phases,
        phase_node=phase_node,
        phase_pr=phase_pr,
        institutional_usernames=institutional_usernames,
        phase1_usernames=phase1_usernames,
    )

    with open(OUT_DIR / "phase_metric_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(
            sorted(
                comparison_rows, key=lambda row: (row["phase_id"], row["metric_order"])
            )
        )

    print("Writing report...")
    build_report(registry_rows, comparison_rows)

    print(f"\nDone. Outputs in {OUT_DIR}/")
    print(f"  account_registry.csv        ({len(registry_rows)} rows)")
    print(f"  phase_metric_comparison.csv ({len(comparison_rows)} rows)")
    print("  report.md")
    print(
        f"  phase 1 author cohort: {len(phase1_user_ids)} user_ids, "
        f"{len(phase1_usernames)} matched usernames"
    )


if __name__ == "__main__":
    main()
