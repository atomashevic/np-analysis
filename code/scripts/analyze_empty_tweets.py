#!/usr/bin/env python3
"""
Analyze "empty" tweets — originals that contain only #NisamPrijavila + a t.co URL.

Usage:
    uv run scripts/analyze_empty_tweets.py
"""

import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "np.csv"
OUTPUT = ROOT / "data" / "empty_tweet_analysis.md"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0"


def resolve_url(url: str, timeout: int = 5) -> str:
    try:
        r = SESSION.head(url, allow_redirects=True, timeout=timeout)
        return r.url
    except Exception:
        try:
            r = SESSION.get(url, allow_redirects=True, timeout=timeout, stream=True)
            return r.url
        except Exception as e:
            return f"ERROR: {e}"


def is_empty_tweet(text: str) -> bool:
    """True if after removing hashtags, t.co URLs, and whitespace nothing remains."""
    cleaned = text
    cleaned = re.sub(r"https?://t\.co/\S+", "", cleaned)
    cleaned = re.sub(r"#\S+", "", cleaned)
    cleaned = re.sub(r"@\S+", "", cleaned)
    cleaned = re.sub(r"[^\w]", "", cleaned)
    return len(cleaned.strip()) == 0


def extract_tco_urls(text: str) -> list[str]:
    return re.findall(r"https?://t\.co/\S+", text)


def main() -> None:
    df = pd.read_csv(INPUT, dtype={"id_str": str})

    is_retweet = df["text"].str.startswith("RT @")
    originals = df[~is_retweet].copy()
    n_originals = len(originals)

    empty_mask = originals["text"].apply(is_empty_tweet)
    empty_df = originals[empty_mask].copy()
    n_empty = len(empty_df)

    print(f"Total original tweets : {n_originals:,}")
    print(f"Empty (URL-only) tweets: {n_empty:,} ({n_empty/n_originals*100:.1f}%)")

    # Extract all t.co URLs from empty tweets
    all_tco = []
    for text in empty_df["text"]:
        all_tco.extend(extract_tco_urls(text))

    print(f"\nTotal t.co URLs to resolve: {len(all_tco):,}")
    unique_tco = list(set(all_tco))
    print(f"Unique t.co URLs: {len(unique_tco):,}")

    print("\nResolving URLs (this may take a while)...")
    resolved = {}
    for i, url in enumerate(unique_tco):
        if i % 10 == 0:
            print(f"  {i}/{len(unique_tco)}...", flush=True)
        resolved[url] = resolve_url(url)

    # Count destinations by domain
    domains = Counter()
    errors = 0
    for final_url in resolved.values():
        if final_url.startswith("ERROR"):
            errors += 1
        else:
            try:
                domain = urlparse(final_url).netloc.lower()
                domain = re.sub(r"^www\.", "", domain)
                domains[domain] += 1
            except Exception:
                errors += 1

    top_domains = domains.most_common(20)

    # Write report
    lines = [
        "# Empty Tweet Analysis — #NisamPrijavila",
        "",
        "## Summary",
        f"- Total original tweets: {n_originals:,}",
        f"- Empty (hashtag + URL only) tweets: {n_empty:,} ({n_empty/n_originals*100:.1f}% of originals)",
        f"- Total t.co URLs found: {len(all_tco):,}",
        f"- Unique t.co URLs resolved: {len(unique_tco):,}",
        f"- Failed to resolve: {errors}",
        "",
        "## Top 20 Link Destinations (by domain)",
        "",
        "| Rank | Domain | Count |",
        "|------|--------|-------|",
    ]
    for rank, (domain, count) in enumerate(top_domains, 1):
        lines.append(f"| {rank} | {domain} | {count} |")

    lines += ["", "## Sample Empty Tweets (first 10)", ""]
    for _, row in empty_df.head(10).iterrows():
        lines.append(f"- `{row['id_str']}`: {row['text'][:120]}")

    OUTPUT.write_text("\n".join(lines))
    print(f"\nReport saved → {OUTPUT}")


if __name__ == "__main__":
    main()
