#!/usr/bin/env python3
"""
Export anonymized original tweets (no retweets) to Excel.

Usage:
    uv run scripts/export_anonymized_tweets.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "data" / "np.csv"
OUTPUT = ROOT / "data" / "np_anonymized_originals.xlsx"


def anonymize_mentions(text: str) -> str:
    """Replace all @username mentions with @[user]."""
    return re.sub(r"@\w+", "@[user]", text)


def main() -> None:
    df = pd.read_csv(INPUT, dtype={"id_str": str})

    is_retweet = df["text"].str.startswith("RT @")
    n_retweets = is_retweet.sum()
    n_originals = (~is_retweet).sum()

    print(f"Total tweets  : {len(df):,}")
    print(f"Retweets      : {n_retweets:,}")
    print(f"Original tweets: {n_originals:,}")

    originals = df[~is_retweet].copy()

    # Anonymize: scrub @mentions from text (covers from_user references too)
    originals["text"] = originals["text"].apply(anonymize_mentions)

    output_df = originals[["id_str", "text"]].rename(columns={"id_str": "tweet_id"})

    # Remove duplicates by tweet_id, then by text
    n_before = len(output_df)
    output_df = output_df.drop_duplicates(subset=["tweet_id"])
    n_after_id = len(output_df)
    output_df = output_df.drop_duplicates(subset=["text"])
    n_after_text = len(output_df)

    print(f"Duplicate tweet_ids removed : {n_before - n_after_id:,}")
    print(f"Duplicate texts removed     : {n_after_id - n_after_text:,}")
    print(f"Final unique tweets         : {n_after_text:,}")

    output_df.to_excel(OUTPUT, index=False, engine="openpyxl")
    print(f"\nSaved {len(output_df):,} anonymized original tweets → {OUTPUT}")


if __name__ == "__main__":
    main()
