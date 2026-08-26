"""Classify Twitter accounts as personal or institutional using OpenAI.

Two-stage pipeline:
  Stage 1 — Classify all unique users by username + metadata (no web search).
            Results written incrementally to a stage-1 JSONL file.
  Stage 2 — Re-classify non-personal / unknown users with web search enabled
             so the model can verify organisations, parties, and media outlets.
            Final results written to the output JSONL file.

Both stages are fully resumable: if interrupted, re-run and already-classified
users are skipped automatically.

Usage (from project root):
    python scripts/nlp/institutions_chatgpt.py
    python scripts/nlp/institutions_chatgpt.py --workers 8

Categories (English labels used in output):
    personal  — individual / personal account
    media     — news outlet, TV, radio, newspaper, portal
    political — political party, movement, or politician
    ngo       — NGO, civil society, human rights organisation
    unknown   — cannot determine
"""

import argparse
import csv
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import pandas as pd
from tqdm import tqdm

# ── API key ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = openai.OpenAI()

MODEL = "gpt-5-mini"

# ── Label mapping (Serbian → English) ─────────────────────────────────────
LABEL_MAP = {
    "licni": "personal",
    "lični": "personal",
    "mediji": "media",
    "politika": "political",
    "nvo": "ngo",
    "nepoznato": "unknown",
}

VALID_LABELS = set(LABEL_MAP.values())

# ── Stage 1 prompt (no web search) ────────────────────────────────────────
STAGE1_PROMPT = """Klasifikuj ovaj Twitter nalog na osnovu korisničkog imena i metapodataka.

Kategorije:
- licni — lični nalog, obična osoba
- mediji — medijska organizacija (TV, novine, radio, portal, novinarska redakcija)
- politika — politička stranka, pokret, ili političar
- nvo — nevladina organizacija, civilno društvo, organizacija za ljudska prava
- nepoznato — ne može se utvrditi

Odgovori SAMO jednom rečju: licni, mediji, politika, nvo, ili nepoznato.

Korisničko ime: {username}
Broj pratilaca: {followers}
Broj praćenih: {friends}
Lokacija: {location}
Broj objava: {n_posts}
Broj retvitova: {n_retweets}
Broj komentara: {n_comments}"""

# ── Stage 2 prompt (with web search) ──────────────────────────────────────
STAGE2_PROMPT = """Klasifikuj ovaj Twitter nalog. Koristi pretragu interneta da proveriš da li je
ovo nalog medijske kuće, političke stranke/pokreta/političara, ili nevladine organizacije.

Kategorije:
- licni — lični nalog, obična osoba
- mediji — medijska organizacija (TV, novine, radio, portal, novinarska redakcija)
- politika — politička stranka, pokret, ili političar
- nvo — nevladina organizacija, civilno društvo, organizacija za ljudska prava
- nepoznato — ne može se utvrditi

Odgovori SAMO jednom rečju: licni, mediji, politika, nvo, ili nepoznato.

Korisničko ime: {username}
Broj pratilaca: {followers}
Broj praćenih: {friends}
Lokacija: {location}
Broj objava: {n_posts}
Broj retvitova: {n_retweets}
Broj komentara: {n_comments}"""


# ── Helpers ────────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def normalise_label(raw: str) -> str:
    """Map a raw model response to a canonical English label."""
    token = raw.strip().lower().rstrip(".").strip()
    if token in LABEL_MAP:
        return LABEL_MAP[token]
    if token in VALID_LABELS:
        return token
    for key, value in LABEL_MAP.items():
        if key in token:
            return value
    for label in VALID_LABELS:
        if label in token:
            return label
    return "unknown"


def build_user_meta(username: str, nodes: dict) -> dict:
    """Return metadata dict for a username, with safe defaults."""
    node = nodes.get(username, {})
    return {
        "username": username,
        "followers": node.get("followers_count", "N/A"),
        "friends": node.get("friends_count", "N/A"),
        "location": node.get("location", "").strip() or "N/A",
        "n_posts": node.get("n_posts", "N/A"),
        "n_retweets": node.get("n_retweets", "N/A"),
        "n_comments": node.get("n_comments", "N/A"),
    }


def classify_stage1(meta: dict) -> str:
    """Stage 1: classify without web search using Responses API."""
    prompt = STAGE1_PROMPT.format(**meta)
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )
    return response.output_text.strip()


def classify_stage2(meta: dict) -> str:
    """Stage 2: classify with web search tool (Serbia locale)."""
    prompt = STAGE2_PROMPT.format(**meta)
    response = client.responses.create(
        model=MODEL,
        tools=[
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": "RS",
                },
            }
        ],
        input=prompt,
    )
    return response.output_text.strip()


def load_existing(path: str) -> dict:
    """Load already-classified users from JSONL to enable resuming."""
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    existing[record["user"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def append_result(record: dict, path: str):
    """Append a single JSONL record (thread-safe)."""
    with _write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def load_nodes(path: str) -> dict:
    """Load np_nodes.csv into a username-keyed dict."""
    nodes = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            nodes[row["username"]] = row
    return nodes


def stage1_worker(username: str, nodes: dict) -> dict:
    """Classify a single user (Stage 1). Returns a result dict."""
    meta = build_user_meta(username, nodes)
    try:
        raw = classify_stage1(meta)
        label = normalise_label(raw)
    except Exception as e:
        raw = str(e)
        label = "unknown"
    return {
        "user": username,
        "label": label,
        "raw": raw,
    }


def stage2_worker(username: str, stage1_raw: str, nodes: dict) -> dict:
    """Classify a single user (Stage 2 with web search). Returns a result dict."""
    meta = build_user_meta(username, nodes)
    try:
        raw_s2 = classify_stage2(meta)
        label_s2 = normalise_label(raw_s2)
    except Exception as e:
        raw_s2 = str(e)
        label_s2 = "unknown"
    return {
        "user": username,
        "institution_type": label_s2,
        "raw_label_stage1": stage1_raw,
        "raw_label_stage2": raw_s2,
        "web_searched": True,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Classify Twitter accounts as personal or institutional (two-stage pipeline)."
    )
    parser.add_argument(
        "--input",
        default="data/np_without_duplicates.csv",
        help="Path to deduplicated tweet CSV (default: data/np_without_duplicates.csv)",
    )
    parser.add_argument(
        "--nodes",
        default="data/np_nodes.csv",
        help="Path to node attributes CSV (default: data/np_nodes.csv)",
    )
    parser.add_argument(
        "--output",
        default="results/users_institution_chatgpt.jsonl",
        help="Final output JSONL path (default: results/users_institution_chatgpt.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel API workers (default: 5)",
    )
    args = parser.parse_args()

    # Derived paths
    stage1_path = args.output.replace(".jsonl", "_stage1.jsonl")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load data
    print("Loading data...")
    data = pd.read_csv(args.input, dtype={"id_str": "str", "parent_id_str": "str"})
    users = sorted(data["from_user"].unique())
    nodes = load_nodes(args.nodes)
    print(f"  {len(users)} unique users, {len(nodes)} with node metadata")

    # ── Stage 1: classify all users without web search ─────────────────
    stage1_existing = load_existing(stage1_path)
    stage1_todo = [u for u in users if u not in stage1_existing]
    print(
        f"\n── Stage 1: {len(stage1_existing)} done, {len(stage1_todo)} remaining (no web search) ──"
    )

    if stage1_todo:
        pbar = tqdm(total=len(stage1_todo), desc="Stage 1", unit="user")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(stage1_worker, username, nodes): username
                for username in stage1_todo
            }
            for future in as_completed(futures):
                username = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"user": username, "label": "unknown", "raw": str(e)}

                # Write immediately to stage1 file
                append_result(result, stage1_path)
                stage1_existing[username] = result
                pbar.update(1)

        pbar.close()

    # Stage 1 summary
    s1_counts = Counter(r["label"] for r in stage1_existing.values())
    print(f"\n  Stage 1 distribution:")
    for label, count in s1_counts.most_common():
        print(f"    {label}: {count}")

    # ── Stage 2: re-classify non-personal / unknown with web search ────
    final_existing = load_existing(args.output)
    stage2_candidates = [
        u
        for u in users
        if u not in final_existing
        and stage1_existing.get(u, {}).get("label")
        in ("media", "political", "ngo", "unknown")
    ]

    # Write personal accounts directly to final output (skip Stage 2)
    personal_to_write = [
        u
        for u in users
        if u not in final_existing
        and stage1_existing.get(u, {}).get("label") == "personal"
    ]
    for username in personal_to_write:
        record = {
            "user": username,
            "institution_type": "personal",
            "raw_label_stage1": stage1_existing[username]["raw"],
            "raw_label_stage2": None,
            "web_searched": False,
        }
        append_result(record, args.output)
        final_existing[username] = record

    print(
        f"\n── Stage 2: {len(stage2_candidates)} candidates to verify (with web search) ──"
    )

    if stage2_candidates:
        pbar = tqdm(total=len(stage2_candidates), desc="Stage 2", unit="user")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    stage2_worker,
                    username,
                    stage1_existing[username]["raw"],
                    nodes,
                ): username
                for username in stage2_candidates
            }
            for future in as_completed(futures):
                username = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "user": username,
                        "institution_type": stage1_existing[username]["label"],
                        "raw_label_stage1": stage1_existing[username]["raw"],
                        "raw_label_stage2": str(e),
                        "web_searched": True,
                    }

                append_result(result, args.output)
                final_existing[username] = result
                pbar.update(1)

        pbar.close()

    # ── Summary ────────────────────────────────────────────────────────
    final = load_existing(args.output)
    final_counts = Counter(r["institution_type"] for r in final.values())
    web_searched = sum(1 for r in final.values() if r.get("web_searched"))

    print(f"\n── Done ──")
    print(f"  Total classified: {len(final)}")
    print(f"  Web-searched: {web_searched}")
    print(f"  Results saved to: {args.output}")
    print(f"  Stage 1 cache: {stage1_path}")
    print(f"  Distribution:")
    for label, count in final_counts.most_common():
        print(f"    {label}: {count}")


if __name__ == "__main__":
    main()
