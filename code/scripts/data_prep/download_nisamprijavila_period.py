#!/usr/bin/env python3
"""Download #nisamprijavila tweets with an auditable collection trail.

The default window matches data/np.csv:
2021-12-25 15:11:29 UTC through 2022-01-13 07:25:15 UTC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
FALLBACK_API_KEY = "new1_d487fba1bb5446b6843ee0d0ffedaf2a"
DEFAULT_HASHTAG = "#nisamprijavila"
DEFAULT_START = "2021-12-25 15:11:29"
DEFAULT_END_INCLUSIVE = "2022-01-13 07:25:15"
DEFAULT_OUTPUT_ROOT = Path("data/twitterapi_collection")
NETWORK_FIELDNAMES = [
    "post_id",
    "user_id",
    "username",
    "post_body",
    "type",
    "parent_id",
    "parent_user_id",
]


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


def parse_utc(value: str) -> datetime:
    cleaned = value.strip().replace("T", " ").replace("Z", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Expected UTC datetime like '2021-12-25 15:11:29', got {value!r}")


def twitter_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S_UTC")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json_bytes(obj))


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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
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
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "status_short": run_git(["status", "--short"]),
    }


def make_query(hashtag: str, start: datetime, end_exclusive: datetime, max_id: str | None = None) -> str:
    query = f"{hashtag} since:{twitter_time(start)} until:{twitter_time(end_exclusive)}"
    if max_id:
        query = f"{query} max_id:{max_id}"
    return query


def make_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def api_get(url: str, params: dict[str, str], api_key: str, timeout: int) -> ApiResponse:
    encoded = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            return ApiResponse(
                status_code=response.getcode(),
                body=response.read(),
                headers=dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise ApiError(f"HTTP {exc.code}", exc.code, body) from exc
    except urllib.error.URLError as exc:
        raise ApiError(str(exc), None, b"") from exc


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def classify_tweet(tweet: dict[str, Any]) -> dict[str, str]:
    author = tweet.get("author") or {}
    retweeted = tweet.get("retweeted_tweet") or tweet.get("retweetedTweet")
    if retweeted:
        rt_author = retweeted.get("author") or {}
        return {
            "post_id": str(tweet.get("id") or ""),
            "user_id": str(author.get("id") or "-1"),
            "username": str(author.get("userName") or ""),
            "post_body": str(tweet.get("text") or ""),
            "type": "retweet",
            "parent_id": str(retweeted.get("id") or "-1"),
            "parent_user_id": str(rt_author.get("id") or "-1"),
        }
    if tweet.get("isReply"):
        return {
            "post_id": str(tweet.get("id") or ""),
            "user_id": str(author.get("id") or "-1"),
            "username": str(author.get("userName") or ""),
            "post_body": str(tweet.get("text") or ""),
            "type": "comment",
            "parent_id": str(tweet.get("inReplyToId") or "-1"),
            "parent_user_id": str(tweet.get("inReplyToUserId") or "-1"),
        }
    return {
        "post_id": str(tweet.get("id") or ""),
        "user_id": str(author.get("id") or "-1"),
        "username": str(author.get("userName") or ""),
        "post_body": str(tweet.get("text") or ""),
        "type": "post",
        "parent_id": "-1",
        "parent_user_id": "-1",
    }


def tweet_sort_key(tweet: dict[str, Any]) -> tuple[str, int]:
    created_at = str(tweet.get("createdAt") or "")
    try:
        tweet_id = int(str(tweet.get("id") or "0"))
    except ValueError:
        tweet_id = 0
    return created_at, tweet_id


def oldest_tweet_id(tweets: list[dict[str, Any]]) -> str | None:
    ids = []
    for tweet in tweets:
        try:
            ids.append(int(str(tweet.get("id"))))
        except (TypeError, ValueError):
            pass
    if not ids:
        return None
    oldest = min(ids)
    return str(oldest - 1) if oldest > 0 else str(oldest)


def response_pagination(data: dict[str, Any]) -> tuple[bool, str | None]:
    has_next = bool(data.get("has_next_page"))
    cursor = data.get("next_cursor")
    if cursor is None:
        cursor = data.get("nextCursor")
    return has_next, str(cursor) if cursor else None


def load_resume_state(run_dir: Path) -> tuple[dict[str, Any], set[str]]:
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    seen_ids = set()
    tweets_path = run_dir / "tweets.jsonl"
    if tweets_path.exists():
        with tweets_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                tweet = json.loads(line)
                tweet_id = tweet.get("id")
                if tweet_id is not None:
                    seen_ids.add(str(tweet_id))
    return state, seen_ids


def next_existing_call_index(api_calls_path: Path) -> int:
    max_call = 0
    for row in read_jsonl(api_calls_path):
        call_id = str(row.get("call_id") or "")
        if call_id.startswith("call_"):
            try:
                max_call = max(max_call, int(call_id.split("_", 1)[1]))
            except ValueError:
                pass
    return max_call + 1


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "manifest": run_dir / "run_manifest.json",
        "api_calls": run_dir / "api_calls.jsonl",
        "tweet_log": run_dir / "tweet_download_log.jsonl",
        "tweets": run_dir / "tweets.jsonl",
        "network": run_dir / "tweets_network.csv",
        "summary": run_dir / "summary.json",
        "state": run_dir / "state.json",
        "raw": run_dir / "raw_responses",
    }


def initialize_run_dir(run_dir: Path, resume: bool) -> dict[str, Path]:
    paths = output_paths(run_dir)
    if run_dir.exists() and not resume:
        raise SystemExit(f"Output run directory already exists: {run_dir}")
    paths["raw"].mkdir(parents=True, exist_ok=True)
    if not paths["network"].exists():
        with paths["network"].open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=NETWORK_FIELDNAMES).writeheader()
    return paths


def build_manifest(args: argparse.Namespace, run_dir: Path, start: datetime, end_exclusive: datetime) -> dict[str, Any]:
    return {
        "api_base_url": args.base_url,
        "collection_started_at": isoformat_z(utc_now()),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "effective_until_exclusive_utc": isoformat_z(end_exclusive),
        "end_inclusive_utc": isoformat_z(args.end_inclusive),
        "git": git_info(Path.cwd()),
        "hashtag": args.hashtag,
        "output_dir": str(run_dir),
        "query": make_query(args.hashtag, start, end_exclusive),
        "query_type": args.query_type,
        "script": str(Path(__file__).resolve()),
        "start_utc": isoformat_z(start),
    }


def write_tweet_outputs(
    paths: dict[str, Path],
    tweets: list[dict[str, Any]],
    call_id: str,
    attempt: int,
    response_hash: str,
    seen_ids: set[str],
) -> tuple[int, int, list[str]]:
    new_count = 0
    duplicate_count = 0
    returned_ids = []
    with paths["tweets"].open("a", encoding="utf-8") as tweets_handle, paths["network"].open(
        "a", newline="", encoding="utf-8"
    ) as network_handle:
        network_writer = csv.DictWriter(network_handle, fieldnames=NETWORK_FIELDNAMES)
        for index, tweet in enumerate(tweets):
            tweet_id = str(tweet.get("id") or "")
            if tweet_id:
                returned_ids.append(tweet_id)
            duplicate = tweet_id in seen_ids
            stored = bool(tweet_id) and not duplicate
            tweet_hash = sha256_json(tweet)
            author = tweet.get("author") or {}
            append_jsonl(
                paths["tweet_log"],
                {
                    "attempt": attempt,
                    "author_id": author.get("id"),
                    "call_id": call_id,
                    "downloaded_at": isoformat_z(utc_now()),
                    "duplicate": duplicate,
                    "response_sha256": response_hash,
                    "returned_index": index,
                    "stored": stored,
                    "tweet_created_at": tweet.get("createdAt"),
                    "tweet_id": tweet_id,
                    "tweet_sha256": tweet_hash,
                    "url": tweet.get("url") or tweet.get("twitterUrl"),
                    "username": author.get("userName"),
                },
            )
            if duplicate or not tweet_id:
                duplicate_count += 1
                continue
            seen_ids.add(tweet_id)
            tweets_handle.write(json.dumps(tweet, ensure_ascii=False, sort_keys=True))
            tweets_handle.write("\n")
            network_writer.writerow(classify_tweet(tweet))
            new_count += 1
    return new_count, duplicate_count, returned_ids


def summarize(run_dir: Path, paths: dict[str, Path], manifest: dict[str, Any], completed: bool) -> dict[str, Any]:
    tweets = []
    for line in paths["tweets"].read_text(encoding="utf-8").splitlines() if paths["tweets"].exists() else []:
        if line.strip():
            tweets.append(json.loads(line))
    api_rows = read_jsonl(paths["api_calls"])
    tweet_rows = read_jsonl(paths["tweet_log"])
    created = [str(tweet.get("createdAt")) for tweet in tweets if tweet.get("createdAt")]
    unique_authors = {
        str((tweet.get("author") or {}).get("id"))
        for tweet in tweets
        if (tweet.get("author") or {}).get("id") is not None
    }
    summary = {
        "api_attempts": sum(1 for row in api_rows if row.get("attempt") is not None),
        "api_events": sum(1 for row in api_rows if row.get("event")),
        "collection_completed": completed,
        "collection_finished_at": isoformat_z(utc_now()),
        "duplicates_returned": sum(1 for row in tweet_rows if row.get("duplicate")),
        "earliest_tweet_created_at": min(created) if created else None,
        "file_hashes": {
            "api_calls.jsonl": file_sha256(paths["api_calls"]),
            "tweet_download_log.jsonl": file_sha256(paths["tweet_log"]),
            "tweets.jsonl": file_sha256(paths["tweets"]),
            "tweets_network.csv": file_sha256(paths["network"]),
        },
        "latest_tweet_created_at": max(created) if created else None,
        "manifest": manifest,
        "run_dir": str(run_dir),
        "tweets_returned_in_log": len(tweet_rows),
        "unique_authors": len(unique_authors),
        "unique_tweets_stored": len(tweets),
    }
    write_json(paths["summary"], summary)
    return summary


def collect(args: argparse.Namespace, http_get: Callable[[str, dict[str, str], str, int], ApiResponse] = api_get) -> dict[str, Any]:
    start = args.start
    end_exclusive = args.end_inclusive + timedelta(seconds=1)
    query = make_query(args.hashtag, start, end_exclusive)
    run_dir = Path(args.resume).resolve() if args.resume else (args.output_root / args.run_id).resolve()

    if args.dry_run:
        dry = {
            "api_base_url": args.base_url,
            "dry_run": True,
            "output_dir": str(run_dir),
            "query": query,
            "query_type": args.query_type,
        }
        print(json.dumps(dry, ensure_ascii=False, indent=2, sort_keys=True))
        return dry

    paths = initialize_run_dir(run_dir, resume=bool(args.resume))
    state, seen_ids = load_resume_state(run_dir)
    manifest = build_manifest(args, run_dir, start, end_exclusive)
    if paths["manifest"].exists() and args.resume:
        existing_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest = {**existing_manifest, "resumed_at": isoformat_z(utc_now())}
    write_json(paths["manifest"], manifest)

    api_key = os.environ.get("TWITTERAPI_KEY", FALLBACK_API_KEY)
    call_index = next_existing_call_index(paths["api_calls"])
    cursor = state.get("cursor")
    max_id = state.get("max_id")
    duplicate_only_pages = 0
    completed = False

    try:
        while True:
            call_id = f"call_{call_index:06d}"
            params = {"query": make_query(args.hashtag, start, end_exclusive, max_id), "queryType": args.query_type}
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
                    response = http_get(args.base_url, params, api_key, args.timeout)
                    finished = utc_now()
                    status_code = response.status_code
                    response_size = len(response.body)
                    response_hash = sha256_bytes(response.body)
                    raw_path = paths["raw"] / f"{call_id}_attempt_{attempt:02d}.json"
                    raw_path.write_bytes(response.body)
                    data = json.loads(response.body.decode("utf-8"))
                    tweets_for_log = data.get("tweets") or []
                    if not isinstance(tweets_for_log, list):
                        raise RuntimeError(f"{call_id} response field 'tweets' is not a list")
                    has_next_for_log, next_cursor_for_log = response_pagination(data)
                    duplicate_for_log = 0
                    new_for_log = 0
                    for tweet in tweets_for_log:
                        tweet_id = str(tweet.get("id") or "")
                        if tweet_id and tweet_id not in seen_ids:
                            new_for_log += 1
                        else:
                            duplicate_for_log += 1
                    append_jsonl(
                        paths["api_calls"],
                        {
                            "attempt": attempt,
                            "call_id": call_id,
                            "cursor": cursor,
                            "duplicate_tweets": duplicate_for_log,
                            "duration_seconds": round((finished - started).total_seconds(), 3),
                            "error": None,
                            "finished_at": isoformat_z(finished),
                            "has_next_page": has_next_for_log,
                            "max_id": max_id,
                            "new_tweets": new_for_log,
                            "next_cursor": next_cursor_for_log,
                            "params": params,
                            "raw_response_path": str(raw_path.relative_to(run_dir)),
                            "response_sha256": response_hash,
                            "response_size_bytes": len(response.body),
                            "returned_tweets": len(tweets_for_log),
                            "started_at": isoformat_z(started),
                            "status_code": response.status_code,
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
                            raw_path = paths["raw"] / f"{call_id}_attempt_{attempt:02d}_error.json"
                            raw_path.write_bytes(exc.body)
                    append_jsonl(
                        paths["api_calls"],
                        {
                            "attempt": attempt,
                            "call_id": call_id,
                            "cursor": cursor,
                            "duration_seconds": round((finished - started).total_seconds(), 3),
                            "error": str(exc),
                            "finished_at": isoformat_z(finished),
                            "max_id": max_id,
                            "params": params,
                            "raw_response_path": str(raw_path.relative_to(run_dir)) if raw_path else None,
                            "response_sha256": response_hash,
                            "response_size_bytes": response_size,
                            "started_at": isoformat_z(started),
                            "status_code": status_code,
                            "success": False,
                        },
                    )
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep * attempt)

            if data is None or successful_attempt is None:
                raise RuntimeError(f"{call_id} failed after {args.max_retries} attempts")

            tweets = data.get("tweets") or []
            if not isinstance(tweets, list):
                raise RuntimeError(f"{call_id} response field 'tweets' is not a list")
            has_next, next_cursor = response_pagination(data)
            response_hash = sha256_bytes((paths["raw"] / f"{call_id}_attempt_{successful_attempt:02d}.json").read_bytes())
            new_count, duplicate_count, returned_ids = write_tweet_outputs(
                paths, tweets, call_id, successful_attempt, response_hash, seen_ids
            )

            if tweets and new_count == 0:
                duplicate_only_pages += 1
            else:
                duplicate_only_pages = 0

            if (
                has_next
                and next_cursor
                and duplicate_only_pages >= args.duplicate_page_limit
                and oldest_tweet_id(tweets)
            ):
                next_max_id = oldest_tweet_id(tweets)
                if max_id and next_max_id == str(max_id):
                    append_jsonl(
                        paths["api_calls"],
                        {
                            "call_id": call_id,
                            "duplicate_only_pages": duplicate_only_pages,
                            "event": "max_id_stalled_stop",
                            "max_id": max_id,
                        },
                    )
                    cursor = None
                    completed = True
                    write_json(
                        paths["state"],
                        {
                            "call_index_next": call_index + 1,
                            "cursor": cursor,
                            "last_call_id": call_id,
                            "max_id": max_id,
                            "seen_ids_count": len(seen_ids),
                            "duplicate_only_pages": duplicate_only_pages,
                            "updated_at": isoformat_z(utc_now()),
                        },
                    )
                    break
                append_jsonl(
                    paths["api_calls"],
                    {
                        "abandoned_cursor": next_cursor,
                        "call_id": call_id,
                        "duplicate_only_pages": duplicate_only_pages,
                        "event": "cursor_duplicate_loop_break",
                        "next_max_id": next_max_id,
                    },
                )
                cursor = None
                max_id = next_max_id
                duplicate_only_pages = 0
            elif has_next and next_cursor:
                cursor = next_cursor
            else:
                cursor = None
                max_id = oldest_tweet_id(tweets)

            write_json(
                paths["state"],
                {
                    "call_index_next": call_index + 1,
                    "cursor": cursor,
                    "last_call_id": call_id,
                    "max_id": max_id,
                    "seen_ids_count": len(seen_ids),
                    "duplicate_only_pages": duplicate_only_pages,
                    "updated_at": isoformat_z(utc_now()),
                },
            )

            print(
                f"{call_id}: returned={len(tweets)} new={new_count} duplicates={duplicate_count} total={len(seen_ids)}"
            )
            call_index += 1

            if cursor or max_id:
                continue
            if tweets and max_id and returned_ids:
                continue
            completed = True
            break
    except KeyboardInterrupt:
        raise
    except Exception:
        write_json(run_dir / "failure.json", {"failed_at": isoformat_z(utc_now()), "traceback": traceback.format_exc()})
        raise
    finally:
        summary = summarize(run_dir, paths, manifest, completed)
        if completed:
            state_path = paths["state"]
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
            else:
                state = {}
            state["completed"] = True
            state["completed_at"] = isoformat_z(utc_now())
            write_json(state_path, state)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def run_self_test() -> None:
    responses = [
        {
            "tweets": [
                {"id": "30", "createdAt": "Sat Dec 25 15:15:00 +0000 2021", "text": "a", "author": {"id": "1", "userName": "u1"}},
                {"id": "20", "createdAt": "Sat Dec 25 15:14:00 +0000 2021", "text": "b", "author": {"id": "2", "userName": "u2"}},
            ],
            "has_next_page": True,
            "next_cursor": "cursor-1",
        },
        {
            "tweets": [
                {"id": "20", "createdAt": "Sat Dec 25 15:14:00 +0000 2021", "text": "b", "author": {"id": "2", "userName": "u2"}},
                {"id": "10", "createdAt": "Sat Dec 25 15:13:00 +0000 2021", "text": "c", "author": {"id": "3", "userName": "u3"}},
            ],
            "has_next_page": False,
        },
        {"tweets": [], "has_next_page": False},
    ]
    calls = []

    def fake_get(url: str, params: dict[str, str], api_key: str, timeout: int) -> ApiResponse:
        del url, api_key, timeout
        calls.append(dict(params))
        body = json_bytes(responses[len(calls) - 1])
        return ApiResponse(status_code=200, body=body, headers={})

    with tempfile.TemporaryDirectory() as tmp:
        args = parse_args(
            [
                "--output-root",
                tmp,
                "--run-id",
                "selftest",
                "--max-retries",
                "1",
            ]
        )
        summary = collect(args, http_get=fake_get)
        run_dir = Path(tmp) / "selftest"
        assert summary["unique_tweets_stored"] == 3, summary
        assert summary["duplicates_returned"] == 1, summary
        assert len(read_jsonl(run_dir / "api_calls.jsonl")) == 3
        assert len(read_jsonl(run_dir / "tweet_download_log.jsonl")) == 4
        assert calls[1].get("cursor") == "cursor-1", calls
        assert "max_id:9" in calls[2]["query"], calls
        assert (run_dir / "raw_responses" / "call_000001_attempt_01.json").exists()
    print("self-test passed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--dry-run", action="store_true", help="Print query/run metadata without making API calls.")
    parser.add_argument(
        "--duplicate-page-limit",
        type=int,
        default=5,
        help="Switch from cursor to max_id after this many duplicate-only pages.",
    )
    parser.add_argument("--end-inclusive", type=parse_utc, default=parse_utc(DEFAULT_END_INCLUSIVE))
    parser.add_argument("--hashtag", default=DEFAULT_HASHTAG)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--query-type", default="Latest")
    parser.add_argument("--resume", type=Path, help="Resume an existing run directory.")
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--run-id", default=make_run_id())
    parser.add_argument("--self-test", action="store_true", help="Run mocked pagination/audit-log checks.")
    parser.add_argument("--start", type=parse_utc, default=parse_utc(DEFAULT_START))
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.duplicate_page_limit < 1:
        parser.error("--duplicate-page-limit must be >= 1")
    if args.end_inclusive < args.start:
        parser.error("--end-inclusive must be after --start")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    if shutil.which("uv") is None and sys.version_info < (3, 10):
        print("This script expects the repo Python environment; prefer: uv run --python 3.12 ...", file=sys.stderr)
    collect(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
