"""Track how institutional accounts evolve into network hubs across campaign phases.

For each phase, constructs a weighted directed retweet graph and computes
multiple hub-related centrality metrics for all nodes. Then extracts
institutional accounts' positions and percentile ranks.

Hub metrics for directed networks:
  - HITS hub score: quality of outgoing links (retweeting important accounts)
  - HITS authority score: quality of incoming links (being retweeted by important retweeters)
  - PageRank: recursive importance in directed graph
  - In-degree / out-degree: unique retweeters / retweetees
  - In-strength / out-strength: weighted retweet volume
  - Betweenness centrality: bridging position between other nodes
  - Freeman in-degree centralization: dominance of top node (graph-level)

Outputs:
  analysis/institutional_hub_evolution/
    hub_metrics_by_phase.csv       – all metrics for institutional accounts per phase
    hub_evolution_summary.csv      – key hub indicators across phases
    phase_graph_stats.csv          – graph-level hub diagnostics per phase
    report.md                      – interpretive report
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
DATA = Path("/home/socio/nisamprijavila/data")
PHASE_DIR = Path("/home/socio/nisamprijavila/analysis/retweet_network_phases")
OUT_DIR = Path("/home/socio/nisamprijavila/analysis/institutional_hub_evolution")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── load institutional account registry ────────────────────────────────
def load_registry():
    reg = {}
    with open(
        DATA.parent / "analysis" / "institutional_accounts" / "account_registry.csv"
    ) as f:
        for r in csv.DictReader(f):
            if r["in_dataset"] == "yes":
                reg[r["username"]] = r["category"]
    return reg


# ── load phase boundaries ──────────────────────────────────────────────
def load_phases():
    phases = []
    with open(PHASE_DIR / "phase_boundaries.csv") as f:
        for r in csv.DictReader(f):
            phases.append(
                {
                    "phase_id": int(r["phase_id"]),
                    "label": r["phase_label"],
                    "start": datetime.fromisoformat(r["start_utc"]),
                    "end": datetime.fromisoformat(r["end_utc"]),
                    "n_events": int(r["n_events"]),
                }
            )
    return phases


# ── load tweets and build phase-level directed graphs ──────────────────
def build_phase_graphs(phases):
    """Build weighted directed graphs per phase from retweet data."""
    tweets = []
    with open(DATA / "np_without_duplicates.csv") as f:
        for r in csv.DictReader(f):
            tweets.append(r)

    # Map post_id -> author
    post_authors = {}
    for t in tweets:
        if t["post_type"] == "post":
            post_authors[t["id_str"]] = t["from_user"]

    # Build edge lists per phase
    phase_edges = defaultdict(list)  # phase_id -> [(retweeter, author)]
    for t in tweets:
        if t["post_type"] != "retweet":
            continue
        try:
            ts = datetime.fromisoformat(t["time"]).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        parent_author = post_authors.get(t.get("parent_id_str", ""))
        if not parent_author:
            continue
        retweeter = t["from_user"]
        if retweeter == parent_author:
            continue
        for p in phases:
            if p["start"] <= ts < p["end"]:
                phase_edges[p["phase_id"]].append((retweeter, parent_author))
                break

    # Build networkx DiGraphs
    graphs = {}
    for pid, edges in phase_edges.items():
        G = nx.DiGraph()
        edge_weights = defaultdict(int)
        for src, tgt in edges:
            edge_weights[(src, tgt)] += 1
        for (src, tgt), w in edge_weights.items():
            G.add_edge(src, tgt, weight=w)
        graphs[pid] = G

    return graphs


def compute_percentile(value, all_values):
    """Compute percentile rank of value within all_values."""
    if not all_values:
        return 0.0
    n_below = sum(1 for v in all_values if v < value)
    return 100.0 * n_below / len(all_values)


def freeman_centralization(values):
    """Freeman centralization: how much does the top node dominate?"""
    if len(values) < 2:
        return 0.0
    max_v = max(values)
    n = len(values)
    numerator = sum(max_v - v for v in values)
    # Maximum possible for a star graph
    denominator = (n - 1) * (n - 2) if n > 2 else 1
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    print("Loading data...")
    registry = load_registry()
    phases = load_phases()

    print("Building phase graphs...")
    graphs = build_phase_graphs(phases)

    inst_usernames = set(registry.keys())
    all_metrics_rows = []
    graph_stats_rows = []

    phase_ids = sorted(phase["phase_id"] for phase in phases)
    for pid in phase_ids:
        phase = next(phase for phase in phases if phase["phase_id"] == pid)
        G = graphs.get(pid, nx.DiGraph())

        if G.number_of_nodes() == 0:
            continue

        print(
            f"  Phase {pid} ({phase['label']}): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        # ── Compute centrality metrics ──
        # HITS
        try:
            hubs, authorities = nx.hits(G, max_iter=500, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            hubs = {n: 0.0 for n in G.nodes()}
            authorities = {n: 0.0 for n in G.nodes()}

        # PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight", max_iter=500, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

        # Weighted in/out strength
        in_strength = dict(G.in_degree(weight="weight"))
        out_strength = dict(G.out_degree(weight="weight"))

        # Betweenness centrality (unweighted for speed on large graphs)
        if G.number_of_nodes() > 5000:
            # Approximate for large graphs
            betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
        else:
            betweenness = nx.betweenness_centrality(G)

        # Collect all values for percentile computation
        all_hub = list(hubs.values())
        all_auth = list(authorities.values())
        all_pr = list(pagerank.values())
        all_in_str = list(in_strength.values())
        all_out_str = list(out_strength.values())
        all_betw = list(betweenness.values())
        n_nodes = G.number_of_nodes()
        total_in_str = sum(all_in_str)

        # ── Graph-level hub diagnostics ──
        in_centralization = freeman_centralization(list(in_strength.values()))
        out_centralization = freeman_centralization(list(out_strength.values()))

        # Hub-authority overlap: how many top-20 hubs are also top-20 authorities?
        sorted_hubs = sorted(hubs.items(), key=lambda x: -x[1])[:20]
        sorted_auths = sorted(authorities.items(), key=lambda x: -x[1])[:20]
        top20_hub_set = {u for u, _ in sorted_hubs}
        top20_auth_set = {u for u, _ in sorted_auths}
        hub_auth_overlap = len(top20_hub_set & top20_auth_set)

        # Assortativity
        try:
            degree_assort = nx.degree_assortativity_coefficient(G, weight="weight")
        except (ValueError, nx.NetworkXError):
            degree_assort = float("nan")

        graph_stats_rows.append(
            {
                "phase_id": pid,
                "phase_label": phase["label"],
                "n_nodes": n_nodes,
                "n_edges": G.number_of_edges(),
                "in_centralization": round(in_centralization, 6),
                "out_centralization": round(out_centralization, 6),
                "top20_hub_auth_overlap": hub_auth_overlap,
                "degree_assortativity": round(degree_assort, 4)
                if not np.isnan(degree_assort)
                else "",
                "top_hub": sorted_hubs[0][0] if sorted_hubs else "",
                "top_hub_score": round(sorted_hubs[0][1], 6) if sorted_hubs else 0,
                "top_authority": sorted_auths[0][0] if sorted_auths else "",
                "top_authority_score": round(sorted_auths[0][1], 6)
                if sorted_auths
                else 0,
            }
        )

        # ── Extract institutional account metrics ──
        for username in inst_usernames:
            if username not in G.nodes():
                continue

            cat = registry[username]
            row = {
                "phase_id": pid,
                "phase_label": phase["label"],
                "username": username,
                "category": cat,
                "n_nodes_in_phase": n_nodes,
                # Raw scores
                "hits_hub": round(hubs.get(username, 0), 8),
                "hits_authority": round(authorities.get(username, 0), 8),
                "pagerank": round(pagerank.get(username, 0), 8),
                "in_degree": G.in_degree(username),
                "out_degree": G.out_degree(username),
                "in_strength": in_strength.get(username, 0),
                "out_strength": out_strength.get(username, 0),
                "betweenness": round(betweenness.get(username, 0), 8),
                # Percentile ranks
                "hits_hub_pctl": round(
                    compute_percentile(hubs.get(username, 0), all_hub), 2
                ),
                "hits_authority_pctl": round(
                    compute_percentile(authorities.get(username, 0), all_auth), 2
                ),
                "pagerank_pctl": round(
                    compute_percentile(pagerank.get(username, 0), all_pr), 2
                ),
                "in_strength_pctl": round(
                    compute_percentile(in_strength.get(username, 0), all_in_str), 2
                ),
                "out_strength_pctl": round(
                    compute_percentile(out_strength.get(username, 0), all_out_str), 2
                ),
                "betweenness_pctl": round(
                    compute_percentile(betweenness.get(username, 0), all_betw), 2
                ),
                # Share of total
                "in_strength_share": round(
                    in_strength.get(username, 0) / max(total_in_str, 1), 6
                ),
                # Role classification
                "is_top10pct_authority": 1
                if authorities.get(username, 0)
                >= sorted(all_auth, reverse=True)[max(0, n_nodes // 10 - 1)]
                else 0,
                "is_top10pct_hub": 1
                if hubs.get(username, 0)
                >= sorted(all_hub, reverse=True)[max(0, n_nodes // 10 - 1)]
                else 0,
            }
            all_metrics_rows.append(row)

    # Write hub metrics
    print("Writing outputs...")
    with open(OUT_DIR / "hub_metrics_by_phase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_metrics_rows[0].keys())
        w.writeheader()
        w.writerows(
            sorted(
                all_metrics_rows, key=lambda r: (r["phase_id"], -r["hits_authority"])
            )
        )

    # Write graph stats
    with open(OUT_DIR / "phase_graph_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=graph_stats_rows[0].keys())
        w.writeheader()
        w.writerows(graph_stats_rows)

    # ── Hub evolution summary: track key accounts across phases ──
    # Find accounts that appear in top positions in any phase
    notable = set()
    for r in all_metrics_rows:
        if (
            r["hits_authority_pctl"] >= 95
            or r["hits_hub_pctl"] >= 95
            or r["pagerank_pctl"] >= 95
            or r["betweenness_pctl"] >= 95
        ):
            notable.add(r["username"])

    summary_rows = []
    for username in sorted(notable):
        for pid in phase_ids:
            matches = [
                r
                for r in all_metrics_rows
                if r["username"] == username and r["phase_id"] == pid
            ]
            if matches:
                r = matches[0]
                summary_rows.append(
                    {
                        "username": username,
                        "category": r["category"],
                        "phase_id": pid,
                        "hits_authority_pctl": r["hits_authority_pctl"],
                        "hits_hub_pctl": r["hits_hub_pctl"],
                        "pagerank_pctl": r["pagerank_pctl"],
                        "betweenness_pctl": r["betweenness_pctl"],
                        "in_strength": r["in_strength"],
                        "out_strength": r["out_strength"],
                        "in_strength_share": r["in_strength_share"],
                    }
                )
            else:
                summary_rows.append(
                    {
                        "username": username,
                        "category": registry.get(username, ""),
                        "phase_id": pid,
                        "hits_authority_pctl": "",
                        "hits_hub_pctl": "",
                        "pagerank_pctl": "",
                        "betweenness_pctl": "",
                        "in_strength": "",
                        "out_strength": "",
                        "in_strength_share": "",
                    }
                )

    with open(OUT_DIR / "hub_evolution_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    # ── Generate report ──
    print("Generating report...")
    report = []
    report.append("# Institutional Hub Evolution in #NisamPrijavila\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    report.append("## Methodology\n")
    report.append(
        f"For each of the {len(phase_ids)} campaign phases, a weighted directed retweet graph is constructed "
        "(edge A→B means A retweeted B, weight = count). Six node-level hub metrics are computed:\n"
    )
    report.append(
        "- **HITS authority:** recursive importance from being retweeted by important retweeters (Kleinberg 1999)"
    )
    report.append(
        "- **HITS hub:** recursive importance from retweeting important authorities"
    )
    report.append("- **PageRank:** random-walk importance in directed graph")
    report.append(
        "- **In-strength / out-strength:** total weighted retweet volume received / performed"
    )
    report.append(
        "- **Betweenness centrality:** fraction of shortest paths passing through the node"
    )
    report.append(
        "\nGraph-level diagnostics: Freeman in-centralization, degree assortativity, "
        "top-20 hub-authority overlap.\n"
    )

    # Graph-level stats table
    report.append("## Graph-Level Hub Diagnostics\n")
    report.append(
        "| Phase | Nodes | Edges | In-Central. | Out-Central. | Hub-Auth Overlap | Assortativity | Top Authority | Top Hub |"
    )
    report.append(
        "|-------|------:|------:|------------:|-------------:|-----------------:|--------------:|--------------:|--------:|"
    )
    for gs in graph_stats_rows:
        report.append(
            f"| {gs['phase_label']} | {gs['n_nodes']:,} | {gs['n_edges']:,} | "
            f"{gs['in_centralization']:.4f} | {gs['out_centralization']:.4f} | "
            f"{gs['top20_hub_auth_overlap']}/20 | {gs['degree_assortativity']} | "
            f"{gs['top_authority']} | {gs['top_hub']} |"
        )

    # Notable institutional accounts evolution
    report.append(
        "\n## Institutional Hub Evolution (accounts reaching 95th+ %-ile on any metric)\n"
    )

    for username in sorted(notable):
        cat = registry.get(username, "")
        acct_rows = [r for r in all_metrics_rows if r["username"] == username]
        if not acct_rows:
            continue
        report.append(f"\n### {username} ({cat})\n")
        report.append(
            "| Phase | Auth %-ile | Hub %-ile | PR %-ile | Betw %-ile | In-Str | Out-Str | In-Str Share |"
        )
        report.append(
            "|-------|----------:|---------:|---------:|-----------:|-------:|--------:|-------------:|"
        )
        for pid in phase_ids:
            matches = [r for r in acct_rows if r["phase_id"] == pid]
            if matches:
                r = matches[0]
                report.append(
                    f"| {r['phase_label']} | {r['hits_authority_pctl']:.1f} | {r['hits_hub_pctl']:.1f} | "
                    f"{r['pagerank_pctl']:.1f} | {r['betweenness_pctl']:.1f} | "
                    f"{r['in_strength']} | {r['out_strength']} | {r['in_strength_share']:.4f} |"
                )
            else:
                report.append(f"| phase_{pid} | — | — | — | — | — | — | — |")

    # Key findings
    report.append("\n## Key Findings\n")

    # Find which institutional accounts reach highest authority scores
    top_auths = sorted(all_metrics_rows, key=lambda r: -r["hits_authority_pctl"])
    seen = set()
    report.append("### Top Institutional Authorities by Phase\n")
    for r in top_auths:
        if r["username"] not in seen and r["hits_authority_pctl"] >= 95:
            seen.add(r["username"])
            report.append(
                f"- **{r['username']}** ({r['category']}): authority {r['hits_authority_pctl']:.1f}th %-ile "
                f"in {r['phase_label']} (in-str={r['in_strength']}, PR={r['pagerank_pctl']:.1f}th %-ile)"
            )
        if len(seen) >= 15:
            break

    # Hub-authority separation
    report.append("\n### Hub-Authority Separation\n")
    for gs in graph_stats_rows:
        report.append(
            f"- **{gs['phase_label']}:** {gs['top20_hub_auth_overlap']}/20 overlap "
            f"(top hub: {gs['top_hub']}, top authority: {gs['top_authority']})"
        )

    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(report))

    # ── Console output ──
    print(f"\nDone. Outputs in {OUT_DIR}/")
    print(f"  hub_metrics_by_phase.csv     ({len(all_metrics_rows)} rows)")
    print(f"  hub_evolution_summary.csv    ({len(summary_rows)} rows)")
    print(f"  phase_graph_stats.csv        ({len(graph_stats_rows)} phases)")
    print("  report.md")

    print(f"\n{'=' * 80}")
    print(" GRAPH-LEVEL HUB DIAGNOSTICS")
    print(f"{'=' * 80}")
    print(
        f"  {'Phase':10s}  {'Nodes':>6s}  {'In-Cent':>8s}  {'Out-Cent':>9s}  {'H-A Ovlp':>9s}  {'Assort':>7s}"
    )
    for gs in graph_stats_rows:
        print(
            f"  {gs['phase_label']:10s}  {gs['n_nodes']:>6,}  {gs['in_centralization']:>8.4f}  "
            f"{gs['out_centralization']:>9.4f}  {gs['top20_hub_auth_overlap']:>5}/20    {str(gs['degree_assortativity']):>7s}"
        )

    print(f"\n{'=' * 80}")
    print(" NOTABLE INSTITUTIONAL ACCOUNTS (95th+ %-ile on any metric)")
    print(f"{'=' * 80}")
    for username in sorted(notable):
        cat = registry.get(username, "")
        acct_rows = [r for r in all_metrics_rows if r["username"] == username]
        phases_present = [r["phase_id"] for r in acct_rows]
        peak_auth = max((r["hits_authority_pctl"] for r in acct_rows), default=0)
        peak_hub = max((r["hits_hub_pctl"] for r in acct_rows), default=0)
        peak_pr = max((r["pagerank_pctl"] for r in acct_rows), default=0)
        peak_betw = max((r["betweenness_pctl"] for r in acct_rows), default=0)
        print(
            f"  {username:25s} ({cat:10s})  phases={phases_present}  "
            f"peak: auth={peak_auth:.0f} hub={peak_hub:.0f} PR={peak_pr:.0f} betw={peak_betw:.0f}"
        )


if __name__ == "__main__":
    main()
