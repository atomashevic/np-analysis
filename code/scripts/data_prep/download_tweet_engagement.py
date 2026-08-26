#!/usr/bin/env python3
"""Collect retweeters and quote tweets for downloaded #nisamprijavila tweets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RETWEETERS_URL = "https://api.twitterapi.io/twitter/tweet/retweeters"
QUOTES_URL = "https://api.twitterapi.io/twitter/tweet/quotes"
FALLBACK_API_KEY = "new1_d487fba1bb5446b6843ee0d0ffedaf2a"
DEFAULT_INPUT_TWEETS = Path("data/twitterapi_collection/np_full_20260428T162212Z_paid/tweets.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/twitterapi_engagement_collection")
ENDPOINTS = ("retweeters", "quotes")


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None, body: bytes = b"") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass
class ApiResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json_bytes(obj))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    return {"commit": git(["rev-parse", "HEAD"]), "status_short": git(["status", "--short"])}


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def api_get(url: str, params: dict[str, str], api_key: str, timeout: int) -> ApiResponse:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            return ApiResponse(response.getcode(), response.read(), dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise ApiError(f"HTTP {exc.code}", exc.code, body) from exc
    except urllib.error.URLError as exc:
        raise ApiError(str(exc), None, b"") from exc


def response_pagination(data: dict[str, Any]) -> tuple[bool, str | None]:
    cursor = data.get("next_cursor")
    if cursor is None:
        cursor = data.get("nextCursor")
    return bool(data.get("has_next_page")), str(cursor) if cursor else None


def load_seed_tweets(path: Path) -> list[dict[str, Any]]:
    tweets = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tweets.append(json.loads(line))
    tweets.sort(key=lambda t: int(str(t.get("id") or "0")), reverse=True)
    return tweets


def item_id(endpoint: str, item: dict[str, Any]) -> str:
    if endpoint == "retweeters":
        return str(item.get("id") or item.get("userId") or "")
    return str(item.get("id") or "")


def result_items(endpoint: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    key = "users" if endpoint == "retweeters" else "tweets"
    items = data.get(key) or []
    if not isinstance(items, list):
        raise RuntimeError(f"Response field {key!r} is not a list")
    return items


def expected_count(endpoint: str, tweet: dict[str, Any]) -> int:
    field = "retweetCount" if endpoint == "retweeters" else "quoteCount"
    try:
        return int(tweet.get(field) or 0)
    except (TypeError, ValueError):
        return 0


def endpoint_url(endpoint: str) -> str:
    return RETWEETERS_URL if endpoint == "retweeters" else QUOTES_URL


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "api_calls": run_dir / "api_calls.jsonl",
        "failure": run_dir / "failure.json",
        "manifest": run_dir / "run_manifest.json",
        "quote_tweets": run_dir / "quote_tweets.jsonl",
        "raw": run_dir / "raw_responses",
        "retweeters": run_dir / "retweeters.jsonl",
        "state": run_dir / "state.json",
        "summary": run_dir / "summary.json",
        "target_log": run_dir / "target_log.jsonl",
    }


def initialize(run_dir: Path, resume: bool) -> dict[str, Path]:
    if run_dir.exists() and not resume:
        raise SystemExit(f"Output run directory already exists: {run_dir}")
    paths = output_paths(run_dir)
    paths["raw"].mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        (paths["raw"] / endpoint).mkdir(exist_ok=True)
    return paths


def seen_result_ids(path: Path, endpoint: str) -> set[tuple[str, str]]:
    seen = set()
    for row in read_jsonl(path):
        seed_id = str(row.get("seed_tweet_id") or "")
        obj = row.get("user") if endpoint == "retweeters" else row.get("quote_tweet")
        if isinstance(obj, dict):
            seen.add((seed_id, item_id(endpoint, obj)))
    return seen


def load_state(paths: dict[str, Path], resume: bool) -> dict[str, Any]:
    if resume and paths["state"].exists():
        return json.loads(paths["state"].read_text(encoding="utf-8"))
    return {"target_index": 0, "endpoint_index": 0, "cursor": None}


def next_call_index(paths: dict[str, Path]) -> int:
    max_seen = 0
    for row in read_jsonl(paths["api_calls"]):
        call_id = str(row.get("call_id") or "")
        if call_id.startswith("call_"):
            try:
                max_seen = max(max_seen, int(call_id.split("_", 1)[1]))
            except ValueError:
                pass
    return max_seen + 1


def target_sequence(tweets: list[dict[str, Any]], endpoints: list[str]) -> list[tuple[int, str, dict[str, Any]]]:
    return [(tweet_index, endpoint, tweet) for tweet_index, tweet in enumerate(tweets) for endpoint in endpoints]


def summarize(run_dir: Path, paths: dict[str, Path], completed: bool) -> dict[str, Any]:
    api_rows = read_jsonl(paths["api_calls"])
    target_rows = read_jsonl(paths["target_log"])
    summary = {
        "api_attempts": sum(1 for row in api_rows if row.get("attempt") is not None),
        "api_events": sum(1 for row in api_rows if row.get("event")),
        "collection_completed": completed,
        "collection_finished_at": isoformat_z(utc_now()),
        "file_hashes": {
            "api_calls.jsonl": file_sha256(paths["api_calls"]),
            "quote_tweets.jsonl": file_sha256(paths["quote_tweets"]),
            "retweeters.jsonl": file_sha256(paths["retweeters"]),
            "target_log.jsonl": file_sha256(paths["target_log"]),
        },
        "quote_tweets_stored": len(read_jsonl(paths["quote_tweets"])),
        "retweeters_stored": len(read_jsonl(paths["retweeters"])),
        "run_dir": str(run_dir),
        "targets_logged": len(target_rows),
        "targets_skipped_zero_count": sum(1 for row in target_rows if row.get("event") == "target_skipped_zero_count"),
    }
    write_json(paths["summary"], summary)
    return summary


def manifest(args: argparse.Namespace, run_dir: Path, seed_count: int, endpoints: list[str]) -> dict[str, Any]:
    return {
        "collection_started_at": isoformat_z(utc_now()),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "endpoints": endpoints,
        "git": git_info(Path.cwd()),
        "input_tweets": str(args.input_tweets),
        "output_dir": str(run_dir),
        "quote_endpoint": QUOTES_URL,
        "retweeters_endpoint": RETWEETERS_URL,
        "script": str(Path(__file__).resolve()),
        "seed_tweets": seed_count,
        "skip_zero_counts": not args.include_zero_counts,
    }


def save_items(
    endpoint: str,
    paths: dict[str, Path],
    seed: dict[str, Any],
    items: list[dict[str, Any]],
    call_id: str,
    response_hash: str,
    seen_ids: set[tuple[str, str]],
) -> tuple[int, int]:
    seed_id = str(seed.get("id") or "")
    stored = 0
    duplicates = 0
    out_path = paths["retweeters"] if endpoint == "retweeters" else paths["quote_tweets"]
    with out_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(items):
            identifier = item_id(endpoint, item)
            key = (seed_id, identifier)
            duplicate = key in seen_ids
            append_jsonl(
                paths["target_log"],
                {
                    "call_id": call_id,
                    "downloaded_at": isoformat_z(utc_now()),
                    "duplicate": duplicate,
                    "endpoint": endpoint,
                    "expected_count": expected_count(endpoint, seed),
                    "item_id": identifier,
                    "item_index": index,
                    "item_sha256": sha256_json(item),
                    "response_sha256": response_hash,
                    "seed_tweet_id": seed_id,
                    "seed_tweet_url": seed.get("url") or seed.get("twitterUrl"),
                    "stored": not duplicate and bool(identifier),
                },
            )
            if duplicate or not identifier:
                duplicates += 1
                continue
            seen_ids.add(key)
            row = {
                "collected_at": isoformat_z(utc_now()),
                "endpoint": endpoint,
                "expected_count": expected_count(endpoint, seed),
                "response_sha256": response_hash,
                "seed_tweet_id": seed_id,
                "seed_tweet_url": seed.get("url") or seed.get("twitterUrl"),
            }
            if endpoint == "retweeters":
                row["user"] = item
            else:
                row["quote_tweet"] = item
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            stored += 1
    return stored, duplicates


def collect(args: argparse.Namespace) -> dict[str, Any]:
    seed_tweets = load_seed_tweets(args.input_tweets)
    endpoints = ENDPOINTS if args.endpoint == "both" else (args.endpoint,)
    targets = target_sequence(seed_tweets[: args.limit_tweets] if args.limit_tweets else seed_tweets, list(endpoints))
    run_dir = Path(args.resume).resolve() if args.resume else (args.output_root / args.run_id).resolve()

    if args.dry_run:
        counts = {endpoint: sum(expected_count(endpoint, tweet) > 0 for tweet in seed_tweets) for endpoint in ENDPOINTS}
        payload = {
            "dry_run": True,
            "endpoint": args.endpoint,
            "input_tweets": str(args.input_tweets),
            "nonzero_targets": counts,
            "output_dir": str(run_dir),
            "seed_tweets": len(seed_tweets),
            "skip_zero_counts": not args.include_zero_counts,
            "target_endpoint_pairs": len(targets),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return payload

    paths = initialize(run_dir, bool(args.resume))
    state = load_state(paths, bool(args.resume))
    write_json(paths["manifest"], manifest(args, run_dir, len(seed_tweets), list(endpoints)))
    seen = {
        "retweeters": seen_result_ids(paths["retweeters"], "retweeters"),
        "quotes": seen_result_ids(paths["quote_tweets"], "quotes"),
    }
    call_index = next_call_index(paths)
    api_key = os.environ.get("TWITTERAPI_KEY", FALLBACK_API_KEY)
    target_index = int(state.get("target_index") or 0)
    cursor = state.get("cursor")
    completed = False

    try:
        while target_index < len(targets):
            tweet_index, endpoint, seed = targets[target_index]
            seed_id = str(seed.get("id") or "")
            count = expected_count(endpoint, seed)
            if count <= 0 and not args.include_zero_counts and not cursor:
                append_jsonl(
                    paths["target_log"],
                    {
                        "endpoint": endpoint,
                        "event": "target_skipped_zero_count",
                        "expected_count": count,
                        "seed_tweet_id": seed_id,
                        "seed_tweet_url": seed.get("url") or seed.get("twitterUrl"),
                        "target_index": target_index,
                        "tweet_index": tweet_index,
                    },
                )
                target_index += 1
                write_json(paths["state"], {"target_index": target_index, "cursor": None, "updated_at": isoformat_z(utc_now())})
                continue

            duplicate_pages = 0
            empty_pages = 0
            target_failed = False
            while True:
                call_id = f"call_{call_index:06d}"
                params = {"tweetId": seed_id}
                if cursor:
                    params["cursor"] = str(cursor)
                data = None
                successful_attempt = None
                for attempt in range(1, args.max_retries + 1):
                    started = utc_now()
                    raw_path = None
                    response_hash = None
                    response_size = 0
                    status_code = None
                    try:
                        response = api_get(endpoint_url(endpoint), params, api_key, args.timeout)
                        finished = utc_now()
                        status_code = response.status_code
                        response_size = len(response.body)
                        response_hash = sha256_bytes(response.body)
                        raw_path = paths["raw"] / endpoint / f"{seed_id}_{call_id}_attempt_{attempt:02d}.json"
                        raw_path.write_bytes(response.body)
                        data = json.loads(response.body.decode("utf-8"))
                        items = result_items(endpoint, data)
                        has_next, next_cursor = response_pagination(data)
                        new_estimate = sum(
                            1 for item in items if (seed_id, item_id(endpoint, item)) not in seen[endpoint]
                        )
                        append_jsonl(
                            paths["api_calls"],
                            {
                                "attempt": attempt,
                                "call_id": call_id,
                                "cursor": cursor,
                                "duration_seconds": round((finished - started).total_seconds(), 3),
                                "endpoint": endpoint,
                                "error": None,
                                "expected_count": count,
                                "finished_at": isoformat_z(finished),
                                "has_next_page": has_next,
                                "new_items_estimate": new_estimate,
                                "next_cursor": next_cursor,
                                "params": params,
                                "raw_response_path": str(raw_path.relative_to(run_dir)),
                                "response_sha256": response_hash,
                                "response_size_bytes": response_size,
                                "returned_items": len(items),
                                "seed_tweet_id": seed_id,
                                "started_at": isoformat_z(started),
                                "status_code": status_code,
                                "success": True,
                            },
                        )
                        successful_attempt = attempt
                        break
                    except Exception as exc:
                        finished = utc_now()
                        if isinstance(exc, ApiError):
                            status_code = exc.status_code
                            response_size = len(exc.body)
                            if exc.body:
                                response_hash = sha256_bytes(exc.body)
                                raw_path = paths["raw"] / endpoint / f"{seed_id}_{call_id}_attempt_{attempt:02d}_error.json"
                                raw_path.write_bytes(exc.body)
                        append_jsonl(
                            paths["api_calls"],
                            {
                                "attempt": attempt,
                                "call_id": call_id,
                                "cursor": cursor,
                                "duration_seconds": round((finished - started).total_seconds(), 3),
                                "endpoint": endpoint,
                                "error": str(exc),
                                "expected_count": count,
                                "finished_at": isoformat_z(finished),
                                "params": params,
                                "raw_response_path": str(raw_path.relative_to(run_dir)) if raw_path else None,
                                "response_sha256": response_hash,
                                "response_size_bytes": response_size,
                                "seed_tweet_id": seed_id,
                                "started_at": isoformat_z(started),
                                "status_code": status_code,
                                "success": False,
                            },
                        )
                    if attempt < args.max_retries:
                        time.sleep(args.retry_sleep * attempt)

                if data is None or successful_attempt is None:
                    append_jsonl(
                        paths["target_log"],
                        {
                            "call_id": call_id,
                            "endpoint": endpoint,
                            "error": f"{endpoint} {seed_id} {call_id} failed after {args.max_retries} attempts",
                            "event": "target_failed",
                            "expected_count": count,
                            "failed_at": isoformat_z(utc_now()),
                            "seed_tweet_id": seed_id,
                            "target_index": target_index,
                            "tweet_index": tweet_index,
                        },
                    )
                    if args.stop_on_error:
                        raise RuntimeError(
                            f"{endpoint} {seed_id} {call_id} failed after {args.max_retries} attempts"
                        )
                    call_index += 1
                    cursor = None
                    target_failed = True
                    break

                items = result_items(endpoint, data)
                response_hash = sha256_bytes(
                    (paths["raw"] / endpoint / f"{seed_id}_{call_id}_attempt_{successful_attempt:02d}.json").read_bytes()
                )
                stored, duplicates = save_items(endpoint, paths, seed, items, call_id, response_hash, seen[endpoint])
                has_next, next_cursor = response_pagination(data)
                if not items:
                    empty_pages += 1
                else:
                    empty_pages = 0
                if items and stored == 0:
                    duplicate_pages += 1
                else:
                    duplicate_pages = 0
                call_index += 1
                cursor = next_cursor if has_next and next_cursor else None
                write_json(
                    paths["state"],
                    {
                        "call_index_next": call_index,
                        "cursor": cursor,
                        "endpoint": endpoint,
                        "last_call_id": call_id,
                        "seed_tweet_id": seed_id,
                        "target_index": target_index,
                        "updated_at": isoformat_z(utc_now()),
                    },
                )
                print(
                    f"{call_id} {endpoint} seed={seed_id} returned={len(items)} stored={stored} duplicates={duplicates}"
                )
                if empty_pages >= args.empty_page_limit:
                    cursor = None
                    break
                if duplicate_pages >= args.duplicate_page_limit:
                    append_jsonl(
                        paths["api_calls"],
                        {
                            "call_id": call_id,
                            "duplicate_pages": duplicate_pages,
                            "endpoint": endpoint,
                            "event": "duplicate_page_stop",
                            "seed_tweet_id": seed_id,
                        },
                    )
                    cursor = None
                    break
                if cursor:
                    continue
                break

            if target_failed:
                target_index += 1
                write_json(
                    paths["state"],
                    {
                        "target_index": target_index,
                        "cursor": None,
                        "last_failed_seed_tweet_id": seed_id,
                        "last_failed_endpoint": endpoint,
                        "updated_at": isoformat_z(utc_now()),
                    },
                )
                continue

            append_jsonl(
                paths["target_log"],
                {
                    "endpoint": endpoint,
                    "event": "target_completed",
                    "expected_count": count,
                    "seed_tweet_id": seed_id,
                    "target_index": target_index,
                    "tweet_index": tweet_index,
                },
            )
            target_index += 1
            write_json(paths["state"], {"target_index": target_index, "cursor": None, "updated_at": isoformat_z(utc_now())})

        completed = True
    except KeyboardInterrupt:
        raise
    except Exception:
        write_json(paths["failure"], {"failed_at": isoformat_z(utc_now()), "traceback": traceback.format_exc()})
        raise
    finally:
        if completed:
            if paths["failure"].exists():
                paths["failure"].unlink()
            state = json.loads(paths["state"].read_text(encoding="utf-8")) if paths["state"].exists() else {}
            state["completed"] = True
            state["completed_at"] = isoformat_z(utc_now())
            write_json(paths["state"], state)
        summary = summarize(run_dir, paths, completed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--duplicate-page-limit", type=int, default=3)
    parser.add_argument("--empty-page-limit", type=int, default=1)
    parser.add_argument("--endpoint", choices=("both", "retweeters", "quotes"), default="both")
    parser.add_argument("--include-zero-counts", action="store_true")
    parser.add_argument("--input-tweets", type=Path, default=DEFAULT_INPUT_TWEETS)
    parser.add_argument("--limit-tweets", type=int)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--run-id", default=run_id())
    parser.add_argument("--stop-on-error", action="store_true", help="Abort instead of logging and skipping a failed target.")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.duplicate_page_limit < 1:
        parser.error("--duplicate-page-limit must be >= 1")
    if args.empty_page_limit < 1:
        parser.error("--empty-page-limit must be >= 1")
    if args.limit_tweets is not None and args.limit_tweets < 1:
        parser.error("--limit-tweets must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("uv") is None and sys.version_info < (3, 10):
        print("This script expects the repo Python environment; prefer: uv run --python 3.12 ...", file=sys.stderr)
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
