"""
Fetch all tweets containing #nisamprijavila
from 24 December 2021 10:40 UTC to 25 December 2021 15:11 UTC
using twitterapi.io Advanced Search API.

Usage:
    python fetch_nisamprijavila.py

Before running, set your API key:
    export TWITTERAPI_KEY="your_api_key_here"
    
Or edit the API_KEY variable below.
"""

import requests
import time
import json
import os
import csv
from typing import List, Dict, Optional
from datetime import datetime


# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.environ.get("TWITTERAPI_KEY", "new1_d487fba1bb5446b6843ee0d0ffedaf2a")
BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Twitter advanced search query
# Times are GMT+0 (UTC). The _UTC suffix tells the API to interpret them as UTC.
# Verified: np.csv created_at uses +0000 offset, so the time column is already UTC.
QUERY = "#nisamprijavila since:2021-12-24_04:40:00_UTC until:2021-12-25_21:11:00_UTC"
QUERY_TYPE = "Latest"  # "Latest" for chronological, "Top" for popular

# Output files
OUTPUT_JSON = "data/additional_tweets.json"
OUTPUT_CSV = "data/additional_network.csv"

MAX_RETRIES = 3
# ──────────────────────────────────────────────────────────────────────────


def fetch_all_tweets(query: str, api_key: str, query_type: str = "Latest") -> List[Dict]:
    """
    Fetch all tweets matching the query, handling:
      - cursor-based pagination (pages of ~20 tweets)
      - max_id fallback to go beyond Twitter's ~800-1200 tweet pagination ceiling
      - deduplication by tweet ID
      - rate-limit / error retry with exponential backoff
    """
    headers = {"X-API-Key": api_key}
    all_tweets: List[Dict] = []
    seen_ids: set = set()
    cursor: Optional[str] = None
    last_min_id: Optional[str] = None
    page = 0

    while True:
        page += 1
        params = {"query": query, "queryType": query_type}

        if cursor:
            params["cursor"] = cursor
        elif last_min_id:
            # Extend beyond the initial pagination window
            params["query"] = f"{query} max_id:{last_min_id}"

        # ── Retry loop ─────────────────────────────────────────────────
        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(BASE_URL, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
                success = True
                break
            except requests.exceptions.RequestException as exc:
                status = getattr(resp, "status_code", None) if "resp" in dir() else None
                if status == 429:
                    wait = 5  # free-tier QPS is low; wait longer
                    print(f"  ⏳ Rate-limited. Waiting {wait}s …")
                    time.sleep(wait)
                else:
                    wait = 2 ** attempt
                    print(f"  ⚠️  Request error ({exc}). Retry {attempt}/{MAX_RETRIES} in {wait}s …")
                    time.sleep(wait)

        if not success:
            print(f"  ❌ Giving up after {MAX_RETRIES} retries.")
            break

        # ── Process response ───────────────────────────────────────────
        tweets = data.get("tweets", [])
        has_next = data.get("has_next_page", False)
        cursor = data.get("next_cursor", None)

        new_tweets = [t for t in tweets if t.get("id") not in seen_ids]
        for t in new_tweets:
            seen_ids.add(t["id"])
            all_tweets.append(t)

        print(f"  Page {page}: {len(new_tweets)} new tweets (total so far: {len(all_tweets)})")

        # Nothing new and no more pages → we're done
        if not new_tweets and not has_next:
            break

        # If the API says there are more pages, keep going with cursor
        if has_next:
            continue

        # No next page but we got tweets → switch to max_id pagination
        if new_tweets:
            last_min_id = new_tweets[-1]["id"]
            cursor = None
            continue

        # Safety net
        break

    return all_tweets


def save_json(tweets: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(tweets)} tweets → {path}")


def save_network_csv(tweets: List[Dict], path: str):
    """Save in np_network.csv format: post_id, user_id, username, post_body, type, parent_id, parent_user_id."""
    fieldnames = ["post_id", "user_id", "username", "post_body", "type", "parent_id", "parent_user_id"]
    stats = {"post": 0, "comment": 0, "retweet": 0}

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in tweets:
            author = t.get("author", {})
            rt = t.get("retweeted_tweet") or t.get("retweetedTweet")
            post_id = t.get("id")
            user_id = author.get("id", "-1")
            username = author.get("userName", "")
            text = t.get("text", "")

            if rt:
                row_type = "retweet"
                rt_author = rt.get("author", {})
                parent_id = rt.get("id", "-1")
                parent_user_id = rt_author.get("id", "-1")
            elif t.get("isReply"):
                row_type = "comment"
                parent_id = t.get("inReplyToId", "-1") or "-1"
                parent_user_id = t.get("inReplyToUserId", "-1") or "-1"
            else:
                row_type = "post"
                parent_id = "-1"
                parent_user_id = "-1"

            stats[row_type] += 1
            writer.writerow({
                "post_id": post_id,
                "user_id": user_id,
                "username": username,
                "post_body": text,
                "type": row_type,
                "parent_id": parent_id,
                "parent_user_id": parent_user_id,
            })

    print(f"Saved {sum(stats.values())} tweets -> {path}")
    print(f"  posts: {stats['post']}, comments: {stats['comment']}, retweets: {stats['retweet']}")


def main():
    print("=" * 60)
    print("Fetching #nisamprijavila tweets")
    print(f"  Window : 2021-12-24 04:40 UTC → 2021-12-25 21:11 UTC")
    print(f"  Query  : {QUERY}")
    print(f"  Type   : {QUERY_TYPE}")
    print("=" * 60)

    if API_KEY == "your_api_key_here":
        print("\n⚠️  Set your API key first!")
        print("   export TWITTERAPI_KEY='your_key'")
        print("   or edit API_KEY in this script.\n")
        return

    tweets = fetch_all_tweets(QUERY, API_KEY, QUERY_TYPE)

    print(f"\n✅ Done. Fetched {len(tweets)} unique tweets.\n")

    if tweets:
        save_json(tweets, OUTPUT_JSON)
        save_network_csv(tweets, OUTPUT_CSV)

        # Quick summary
        earliest = min(tweets, key=lambda t: t.get("createdAt", ""))
        latest = max(tweets, key=lambda t: t.get("createdAt", ""))
        print(f"\n📈 Summary:")
        print(f"   Earliest tweet : {earliest.get('createdAt')}")
        print(f"   Latest tweet   : {latest.get('createdAt')}")
        print(f"   Total likes    : {sum(t.get('likeCount', 0) for t in tweets):,}")
        print(f"   Total retweets : {sum(t.get('retweetCount', 0) for t in tweets):,}")
        unique_authors = len(set(t.get("author", {}).get("id") for t in tweets))
        print(f"   Unique authors : {unique_authors:,}")


if __name__ == "__main__":
    main()
