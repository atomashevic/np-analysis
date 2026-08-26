#!/usr/bin/env python3
"""Build the de-identified event-level release described in the manuscript.

The exporter creates fresh random event and account tokens on every run. The
crosswalks exist only in process memory and are never written to disk. Source
text, original platform identifiers, account names, and annotator identifiers
are absent from the output.

Usage:
    uv run python scripts/build_deidentified_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "osf_package" / "data" / "deidentified_events.csv"
DEFAULT_MANIFEST = REPO / "osf_package" / "data" / "deidentified_events.manifest.json"

RAW_EVENTS = REPO / "data" / "np.csv"
DEDUPLICATED_EVENTS = REPO / "data" / "np_without_duplicates.csv"
NETWORK_EVENTS = REPO / "data" / "np_network.csv"
PHASE_BOUNDARIES = (
    REPO / "analysis" / "six_phase_counterfactual" / "phase_boundaries.csv"
)
GEMMA_LABELS = (
    REPO / "results" / "emotions_gemma" / "emotions_gemma4:31b_temp1.0_all_tweets.jsonl"
)
GPT5_LABELS = (
    REPO / "results" / "emotions_gpt" / "emotions_gpt-5_temp1.0_validation_1.jsonl"
)
GPT54_LABELS = (
    REPO / "results" / "emotions_gpt" / "emotions_gpt-5.4_temp1.0_validation_1.jsonl"
)
HUMAN_LABELS = (
    REPO / "results" / "emotions_ff" / "ff_samples_18_25_annotations_with_majority.csv"
)
INSTITUTION_LABELS = (
    REPO / "results" / "users_institution" / "users_institution_final.jsonl"
)
MANUAL_RESOLUTION = (
    REPO / "analysis" / "manual_coding_motives" / "manual_coding_id_resolution.csv"
)
MOTIVE_ANNOTATIONS = REPO / "data" / "manual_coding" / "motive_annotations_long.csv"
MOTIVE_CODEBOOK = REPO / "data" / "manual_coding" / "motive_codebook.csv"

INPUT_PATHS = (
    RAW_EVENTS,
    DEDUPLICATED_EVENTS,
    NETWORK_EVENTS,
    PHASE_BOUNDARIES,
    GEMMA_LABELS,
    GPT5_LABELS,
    GPT54_LABELS,
    HUMAN_LABELS,
    INSTITUTION_LABELS,
    MANUAL_RESOLUTION,
    MOTIVE_ANNOTATIONS,
    MOTIVE_CODEBOOK,
)

COLUMNS = (
    "id",
    "timestamp_utc",
    "anon_user_id",
    "phase",
    "post_type",
    "target_event_id",
    "target_anon_user_id",
    "emotion_gemma_4_31b",
    "emotion_gpt_5",
    "emotion_gpt_5_4",
    "human_emotion_labels",
    "human_emotion_modal",
    "institution_type_gpt5mini",
    "institution_type_final",
    "manual_reporting_status",
    "manual_first_person",
    "manual_motive_codes",
    "manual_motive_parent_categories",
)

EXPECTED_ROWS = 25_117
EXPECTED_TYPES = {"post": 3_874, "retweet": 20_427, "reply": 816}
EXPECTED_PHASES = {
    "phase_1": 610,
    "phase_2": 20_402,
    "phase_3": 2_639,
    "phase_4": 693,
    "phase_5": 379,
    "phase_6": 388,
    "outside_primary_phase_window": 6,
}
EXPECTED_TARGET_EVENTS = {"retweet": 20_381, "reply": 816}
EXPECTED_TARGET_USERS = {"retweet": 20_427, "reply": 816}
EXPECTED_GEMMA = 25_109
EXPECTED_VALIDATION = 400
EXPECTED_MANUAL_UNITS = 4_386
EXPECTED_MOTIVE_EVENTS = 1_122
EXPECTED_ANALYTIC_MOTIVE_EVENTS = 824
EXPECTED_DETAILED_ASSIGNMENTS = 1_266
EXPECTED_PARENT_ASSIGNMENTS = 1_238

RETWEET_RE = re.compile(r"^RT\s+@([A-Za-z0-9_]{1,15}):")
TOKEN_RE = re.compile(r"^(?:e|u)_[0-9a-f]{32}$")
REPORTING_STATUS = {
    "0": "reported",
    "1": "not_reported",
    "2": "reported_and_not_reported",
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
    "iscekkivanje": "anticipation",
    "neutralno": "neutral",
    "Emocionalno neutralno": "neutral",
    "nepoznato": "unknown",
    "Nepoznato": "unknown",
    "Ne mogu da razumem": "unknown",
}
EMOTION_ORDER = {
    label: index
    for index, label in enumerate(
        (
            "joy",
            "sadness",
            "trust",
            "disgust",
            "fear",
            "anger",
            "surprise",
            "anticipation",
            "neutral",
            "unknown",
        )
    )
}

Choice = Callable[[list[int]], int]
TokenFactory = Callable[[str], str]


@dataclass
class BuildContext:
    """In-memory state shared while transforming release rows."""

    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]]
    event_tokens: dict[object, str]
    account_tokens: dict[object, str]
    retweet_targets: dict[str, str]
    emotion_targets: dict[str, str]
    annotations: dict[str, object]
    choice: Choice
    make_token: TokenFactory


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_factory(prefix: str) -> str:
    """Create one opaque UUIDv4 token with a type prefix."""
    return f"{prefix}_{uuid.uuid4().hex}"


def clean(value: object) -> str:
    """Normalize a scalar loaded from a source table."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def account_key(user_id: object, screen_name: object) -> tuple[str, str] | None:
    """Represent an observed platform account without collapsing handle changes."""
    numeric_id = clean(user_id)
    handle = clean(screen_name).casefold()
    if not numeric_id and not handle:
        return None
    return numeric_id, handle


def translate_emotion(value: object) -> str:
    """Map the archived Serbian labels to the manuscript's English label set."""
    label = clean(value)
    if not label:
        return ""
    if label in EMOTION_ORDER:
        return label
    try:
        return LABEL_TRANSLATION[label]
    except KeyError as error:
        raise ValueError(f"Unrecognized emotion label: {label!r}") from error


def json_array(values: Iterable[str]) -> str:
    """Serialize a list compactly and consistently for a CSV cell."""
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def natural_code_key(value: str) -> tuple[int, int, str]:
    """Sort numeric motive codes before special codes."""
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?", value)
    if not match:
        return 10_000, 10_000, value
    return int(match.group(1)), int(match.group(2) or -1), value


def load_jsonl_labels(path: Path, id_field: str) -> dict[str, str]:
    """Load one unique event label per JSONL record."""
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            event_id = clean(record.get(id_field))
            if not event_id or "emotion" not in record:
                raise ValueError(f"Malformed label record at {path}:{line_number}")
            if event_id in result:
                raise ValueError(f"Duplicate event identifier in {path}: {event_id}")
            result[event_id] = translate_emotion(record["emotion"])
    return result


def load_human_labels(path: Path) -> dict[str, tuple[str, str]]:
    """Return all label votes and the modal label without annotator identifiers."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"tweet_id", "annotator_code", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Human annotation columns missing: {sorted(missing)}")
    if frame.duplicated(["tweet_id", "annotator_code"]).any():
        raise ValueError("Duplicate human annotation for one event and annotator")

    result: dict[str, tuple[str, str]] = {}
    for event_id, group in frame.groupby("tweet_id", sort=False):
        labels = [translate_emotion(value) for value in group["label"]]
        modal = Counter(labels).most_common(1)[0][0]
        ordered = sorted(labels, key=lambda label: (EMOTION_ORDER[label], label))
        result[str(event_id)] = json_array(ordered), modal
    return result


def load_institution_labels(path: Path) -> dict[str, tuple[str, str]]:
    """Load model and verified institution labels by case-insensitive handle."""
    result: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            handle = clean(record.get("user")).casefold()
            if not handle or handle in result:
                raise ValueError(f"Invalid institution record at {path}:{line_number}")
            result[handle] = (
                clean(record.get("gpt_label")),
                clean(record.get("institution_type")),
            )
    return result


def normalize_bool(value: object) -> str:
    """Represent archived Boolean annotations as lowercase CSV values."""
    normalized = clean(value).casefold()
    if normalized in {"true", "1"}:
        return "true"
    if normalized in {"false", "0"}:
        return "false"
    if not normalized:
        return ""
    raise ValueError(f"Unexpected Boolean value: {value!r}")


def load_manual_labels(
    resolution_path: Path,
    motive_path: Path,
    codebook_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Load the manuscript-reported manual annotations by resolved event ID."""
    resolution = pd.read_csv(resolution_path, dtype=str).fillna("")
    if resolution["resolved_tweet_id"].duplicated().any():
        raise ValueError("Manual resolution table has duplicate resolved event IDs")

    manual: dict[str, dict[str, str]] = {}
    for row in resolution.itertuples(index=False):
        event_id = clean(row.resolved_tweet_id)
        status_raw = clean(row.reporting_status)
        if status_raw and status_raw not in REPORTING_STATUS:
            raise ValueError(f"Unexpected reporting status: {status_raw!r}")
        manual[event_id] = {
            "manual_reporting_status": REPORTING_STATUS.get(status_raw, ""),
            "manual_first_person": normalize_bool(row.first_person_adjudicated),
        }

    codebook = pd.read_csv(codebook_path, dtype=str).fillna("")
    analytic_families = {
        clean(value)
        for value in codebook["motive_family"]
        if clean(value).isdigit() and clean(value) not in {"0", "21"}
    }
    motives = pd.read_csv(motive_path, dtype=str).fillna("")
    all_codes: dict[str, list[str]] = defaultdict(list)
    analytic_rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    excluded_events: set[str] = set()
    for row in motives.itertuples(index=False):
        event_id = clean(row.resolved_tweet_id)
        code = clean(row.motive_code)
        family = clean(row.motive_family)
        if not event_id or not code:
            raise ValueError("Motive annotation is missing a resolved event ID or code")
        all_codes[event_id].append(code)
        if family in {"0", "x"}:
            excluded_events.add(event_id)
        if family in analytic_families:
            analytic_rows[event_id].append((code, family))

    motive_event_ids = set(all_codes)
    for event_id in set(manual).union(motive_event_ids):
        row = manual.setdefault(
            event_id,
            {"manual_reporting_status": "", "manual_first_person": ""},
        )
        row["manual_motive_codes"] = json_array(
            sorted(set(all_codes[event_id]), key=natural_code_key)
        )
        families = (
            []
            if event_id in excluded_events
            else [family for _, family in analytic_rows[event_id]]
        )
        row["manual_motive_parent_categories"] = json_array(
            sorted(set(families), key=natural_code_key)
        )

    analytic_event_ids = {
        event_id
        for event_id, rows in analytic_rows.items()
        if event_id not in excluded_events and rows
    }
    metrics = {
        "manual_units": len(resolution),
        "motive_annotation_rows": len(motives),
        "motive_events": len(motive_event_ids),
        "analytic_motive_events": len(analytic_event_ids),
        "analytic_detailed_assignments": sum(
            len(rows)
            for event_id, rows in analytic_rows.items()
            if event_id not in excluded_events
        ),
        "analytic_parent_assignments": sum(
            len({family for _, family in rows})
            for event_id, rows in analytic_rows.items()
            if event_id not in excluded_events
        ),
    }
    return manual, metrics


def load_phases(path: Path) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Load ordered half-open phase intervals."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    phases = []
    for row in frame.sort_values("phase_id").itertuples(index=False):
        phases.append(
            (
                clean(row.phase_label),
                pd.Timestamp(row.start_utc),
                pd.Timestamp(row.end_utc),
            )
        )
    return phases


def phase_for(
    timestamp: pd.Timestamp,
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> str:
    """Assign one timestamp to the manuscript phase window."""
    for label, start, end in phases:
        if start <= timestamp < end:
            return label
    if timestamp == phases[-1][2]:
        return phases[-1][0]
    return "outside_primary_phase_window"


def jitter_timestamp(
    timestamp: pd.Timestamp,
    original_phase: str,
    phases: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    choice: Choice,
) -> str:
    """Remove seconds and apply bounded date- and phase-preserving minute jitter."""
    minute = timestamp.floor("min")
    allowed = []
    for offset in range(-15, 16):
        candidate = minute + pd.Timedelta(minutes=offset)
        if (
            candidate.date() == minute.date()
            and phase_for(candidate, phases) == original_phase
        ):
            allowed.append(offset)
    if not allowed:
        raise ValueError(f"No valid jitter offsets for {timestamp}")
    jittered = minute + pd.Timedelta(minutes=choice(allowed))
    return jittered.strftime("%Y-%m-%dT%H:%MZ")


def classify_event(row: object) -> str:
    """Classify archive records using the manuscript's event-type precedence."""
    if RETWEET_RE.match(clean(row.text)):
        return "retweet"
    if clean(row.in_reply_to_status_id_str):
        return "reply"
    return "post"


def retweet_target_account(row: object) -> tuple[str, str] | None:
    """Resolve the retweeted account from the matching user mention entity."""
    match = RETWEET_RE.match(clean(row.text))
    if not match:
        return None
    expected_handle = match.group(1).casefold()
    try:
        entities = json.loads(clean(row.entities_str))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed entities JSON for event {row.id_str}") from error
    mentions = entities.get("user_mentions", [])
    for mention in mentions:
        handle = clean(mention.get("screen_name"))
        if handle.casefold() == expected_handle:
            return account_key(mention.get("id_str", mention.get("id")), handle)
    raise ValueError(f"Retweet target account unresolved for event {row.id_str}")


def build_relation_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Load retweet target events and primary-emotion inheritance targets."""
    network = pd.read_csv(NETWORK_EVENTS, dtype=str).fillna("")
    retweet_targets = {
        clean(row.post_id): clean(row.parent_id)
        for row in network.itertuples(index=False)
        if clean(row.type) == "retweet" and clean(row.parent_id) not in {"", "-1"}
    }
    deduplicated = pd.read_csv(DEDUPLICATED_EVENTS, dtype=str).fillna("")
    emotion_targets = {
        clean(row.id_str): clean(row.target_id_str)
        for row in deduplicated.itertuples(index=False)
        if clean(row.id_str) and clean(row.target_id_str)
    }
    return retweet_targets, emotion_targets


def load_annotations() -> tuple[dict[str, object], dict[str, int]]:
    """Load every model and manual annotation selected for public release."""
    manual, manual_metrics = load_manual_labels(
        MANUAL_RESOLUTION,
        MOTIVE_ANNOTATIONS,
        MOTIVE_CODEBOOK,
    )
    annotations: dict[str, object] = {
        "gemma": load_jsonl_labels(GEMMA_LABELS, "id"),
        "gpt5": load_jsonl_labels(GPT5_LABELS, "id_str"),
        "gpt54": load_jsonl_labels(GPT54_LABELS, "id_str"),
        "human": load_human_labels(HUMAN_LABELS),
        "institution": load_institution_labels(INSTITUTION_LABELS),
        "manual": manual,
    }
    return annotations, manual_metrics


def assign_token(
    mapping: dict[object, str],
    source_key: object,
    prefix: str,
    make_token: TokenFactory,
) -> str:
    """Return one stable in-run opaque token for a source key."""
    if source_key not in mapping:
        token = make_token(prefix)
        if not TOKEN_RE.fullmatch(token):
            raise ValueError(f"Token factory returned an invalid {prefix!r} token")
        if token in mapping.values():
            raise ValueError(f"Token collision for {token}")
        mapping[source_key] = token
    return mapping[source_key]


def build_event_row(
    row: object,
    context: BuildContext,
) -> dict[str, str]:
    """Transform one source event into one release row."""
    event_id = clean(row.id_str)
    timestamp = pd.to_datetime(clean(row.time), utc=True, errors="raise")
    phase = phase_for(timestamp, context.phases)
    post_type = classify_event(row)
    author_key = account_key(row.from_user_id_str, row.from_user)
    if author_key is None:
        raise ValueError(f"Missing author account for event {event_id}")

    target_event = ""
    target_account = ""
    if post_type == "retweet":
        target_source_id = context.retweet_targets.get(event_id, "")
        if target_source_id:
            target_event = assign_token(
                context.event_tokens, target_source_id, "e", context.make_token
            )
        target_key = retweet_target_account(row)
        if target_key is None:
            raise ValueError(f"Missing retweet target account for event {event_id}")
        target_account = assign_token(
            context.account_tokens, target_key, "u", context.make_token
        )
    elif post_type == "reply":
        target_source_id = clean(row.in_reply_to_status_id_str)
        target_event = assign_token(
            context.event_tokens, target_source_id, "e", context.make_token
        )
        target_key = account_key(
            row.in_reply_to_user_id_str,
            row.in_reply_to_screen_name,
        )
        if target_key is None:
            raise ValueError(f"Missing reply target account for event {event_id}")
        target_account = assign_token(
            context.account_tokens, target_key, "u", context.make_token
        )

    gemma_labels = context.annotations["gemma"]
    gpt5_labels = context.annotations["gpt5"]
    gpt54_labels = context.annotations["gpt54"]
    human_labels = context.annotations["human"]
    institution_labels = context.annotations["institution"]
    manual_labels = context.annotations["manual"]
    emotion_target = context.emotion_targets.get(event_id, event_id)
    human_array, human_modal = human_labels.get(event_id, ("[]", ""))
    institution_gpt, institution_final = institution_labels.get(
        clean(row.from_user).casefold(),
        ("", ""),
    )
    manual = manual_labels.get(event_id, {})

    return {
        "id": assign_token(context.event_tokens, event_id, "e", context.make_token),
        "timestamp_utc": jitter_timestamp(
            timestamp, phase, context.phases, context.choice
        ),
        "anon_user_id": assign_token(
            context.account_tokens, author_key, "u", context.make_token
        ),
        "phase": phase,
        "post_type": post_type,
        "target_event_id": target_event,
        "target_anon_user_id": target_account,
        "emotion_gemma_4_31b": gemma_labels.get(emotion_target, ""),
        "emotion_gpt_5": gpt5_labels.get(event_id, ""),
        "emotion_gpt_5_4": gpt54_labels.get(event_id, ""),
        "human_emotion_labels": human_array,
        "human_emotion_modal": human_modal,
        "institution_type_gpt5mini": institution_gpt,
        "institution_type_final": institution_final,
        "manual_reporting_status": manual.get("manual_reporting_status", ""),
        "manual_first_person": manual.get("manual_first_person", ""),
        "manual_motive_codes": manual.get("manual_motive_codes", "[]"),
        "manual_motive_parent_categories": manual.get(
            "manual_motive_parent_categories", "[]"
        ),
    }


def build_release(
    *,
    choice: Choice = secrets.choice,
    make_token: TokenFactory = token_factory,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the complete release table and keep crosswalks in memory only."""
    raw = pd.read_csv(RAW_EVENTS, dtype=str).fillna("")
    raw = raw.drop_duplicates("id_str", keep="first").copy()
    phases = load_phases(PHASE_BOUNDARIES)
    retweet_targets, emotion_targets = build_relation_maps()
    annotations, manual_metrics = load_annotations()

    event_tokens: dict[object, str] = {}
    account_tokens: dict[object, str] = {}
    for event_id in raw["id_str"]:
        assign_token(event_tokens, clean(event_id), "e", make_token)

    context = BuildContext(
        phases=phases,
        event_tokens=event_tokens,
        account_tokens=account_tokens,
        retweet_targets=retweet_targets,
        emotion_targets=emotion_targets,
        annotations=annotations,
        choice=choice,
        make_token=make_token,
    )
    rows = [build_event_row(row, context) for row in raw.itertuples(index=False)]
    release = pd.DataFrame(rows, columns=COLUMNS)
    return release, manual_metrics


def collect_metrics(
    frame: pd.DataFrame, manual_metrics: dict[str, int]
) -> dict[str, object]:
    """Collect release counts used for validation and provenance."""
    metrics: dict[str, object] = {
        "rows": len(frame),
        "unique_event_tokens": int(frame["id"].nunique()),
        "unique_author_tokens": int(frame["anon_user_id"].nunique()),
        "post_type_counts": frame["post_type"].value_counts().sort_index().to_dict(),
        "phase_counts": frame["phase"].value_counts().sort_index().to_dict(),
        "target_event_counts": {
            kind: int(
                frame.loc[frame["post_type"] == kind, "target_event_id"].ne("").sum()
            )
            for kind in ("retweet", "reply")
        },
        "target_user_counts": {
            kind: int(
                frame.loc[frame["post_type"] == kind, "target_anon_user_id"]
                .ne("")
                .sum()
            )
            for kind in ("retweet", "reply")
        },
        "annotation_nonempty_counts": {
            column: int(frame[column].ne("").sum())
            for column in (
                "emotion_gemma_4_31b",
                "emotion_gpt_5",
                "emotion_gpt_5_4",
                "human_emotion_modal",
                "institution_type_gpt5mini",
                "institution_type_final",
                "manual_reporting_status",
                "manual_first_person",
            )
        },
        "manual_source_counts": manual_metrics,
    }
    return metrics


def validate_release(
    frame: pd.DataFrame, manual_metrics: dict[str, int]
) -> dict[str, object]:
    """Fail closed when the canonical release differs from expected manuscript data."""
    if tuple(frame.columns) != COLUMNS:
        raise ValueError("Release columns differ from the approved schema")
    if frame["id"].duplicated().any():
        raise ValueError("Release event tokens are not unique")
    token_columns = ["id", "anon_user_id"]
    for column in token_columns:
        if not frame[column].map(lambda value: bool(TOKEN_RE.fullmatch(value))).all():
            raise ValueError(f"Invalid token in {column}")
    for column in ("target_event_id", "target_anon_user_id"):
        valid = frame[column].eq("") | frame[column].map(
            lambda value: bool(TOKEN_RE.fullmatch(value))
        )
        if not valid.all():
            raise ValueError(f"Invalid token in {column}")
    for column in (
        "human_emotion_labels",
        "manual_motive_codes",
        "manual_motive_parent_categories",
    ):
        if (
            not frame[column]
            .map(lambda value: isinstance(json.loads(value), list))
            .all()
        ):
            raise ValueError(f"Invalid JSON array in {column}")

    metrics = collect_metrics(frame, manual_metrics)
    expected_scalars = {
        "rows": EXPECTED_ROWS,
        "unique_event_tokens": EXPECTED_ROWS,
    }
    for key, expected in expected_scalars.items():
        if metrics[key] != expected:
            raise ValueError(f"Unexpected {key}: {metrics[key]} != {expected}")
    expected_mappings = {
        "post_type_counts": EXPECTED_TYPES,
        "phase_counts": EXPECTED_PHASES,
        "target_event_counts": EXPECTED_TARGET_EVENTS,
        "target_user_counts": EXPECTED_TARGET_USERS,
    }
    for key, expected in expected_mappings.items():
        if metrics[key] != expected:
            raise ValueError(f"Unexpected {key}: {metrics[key]} != {expected}")
    annotation_counts = metrics["annotation_nonempty_counts"]
    expected_annotations = {
        "emotion_gemma_4_31b": EXPECTED_GEMMA,
        "emotion_gpt_5": EXPECTED_VALIDATION,
        "emotion_gpt_5_4": EXPECTED_VALIDATION,
        "human_emotion_modal": EXPECTED_VALIDATION,
        "institution_type_gpt5mini": EXPECTED_ROWS,
        "institution_type_final": EXPECTED_ROWS,
        "manual_first_person": EXPECTED_MANUAL_UNITS,
    }
    for key, expected in expected_annotations.items():
        if annotation_counts[key] != expected:
            raise ValueError(
                f"Unexpected annotation count for {key}: {annotation_counts[key]} != {expected}"
            )
    expected_manual = {
        "manual_units": EXPECTED_MANUAL_UNITS,
        "motive_annotation_rows": 1_565,
        "motive_events": EXPECTED_MOTIVE_EVENTS,
        "analytic_motive_events": EXPECTED_ANALYTIC_MOTIVE_EVENTS,
        "analytic_detailed_assignments": EXPECTED_DETAILED_ASSIGNMENTS,
        "analytic_parent_assignments": EXPECTED_PARENT_ASSIGNMENTS,
    }
    if manual_metrics != expected_manual:
        raise ValueError(f"Unexpected manual source counts: {manual_metrics}")
    return metrics


def write_frame(frame: pd.DataFrame, output: Path) -> None:
    """Write a CSV through a temporary file and replace the destination."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        dir=temp_root,
        encoding="utf-8",
        newline="",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False, lineterminator="\n")
    staged = output.with_suffix(output.suffix + ".tmp")
    try:
        shutil.copyfile(temporary, staged)
        staged.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def write_manifest(
    output: Path,
    manifest_path: Path,
    metrics: dict[str, object],
) -> None:
    """Write hashes, counts, and transformation policy for one release run."""
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "privacy_classification": "de-identified pseudonymous event-level research data",
        "privacy_limit": (
            "Event timing, interaction structure, and rare annotations can permit linkage. "
            "The artifact is not claimed to be anonymous."
        ),
        "transformations": {
            "event_ids": "fresh random UUIDv4 tokens; crosswalk never persisted",
            "account_ids": (
                "fresh random UUIDv4 tokens for observed numeric-id and handle pairs; "
                "crosswalk never persisted"
            ),
            "timestamps": (
                "seconds removed; random integer jitter from -15 to +15 minutes, "
                "constrained to preserve UTC date and manuscript phase"
            ),
            "text": "excluded",
            "multi_label_fields": "compact JSON arrays",
        },
        "inputs": {
            str(path.relative_to(REPO)): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in INPUT_PATHS
        },
        "output": {
            "path": str(output.relative_to(REPO)),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
        },
        "counts": metrics,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        dir=temp_root,
        encoding="utf-8",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    staged = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        shutil.copyfile(temporary, staged)
        staged.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def main(output: Path, manifest: Path, force: bool) -> int:
    """Build, validate, and write the canonical release artifacts."""
    existing = [path for path in (output, manifest) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path.relative_to(REPO)) for path in existing)
        raise SystemExit(
            f"refusing to overwrite existing artifact(s): {names}; pass --force"
        )
    for path in INPUT_PATHS:
        if not path.exists():
            raise SystemExit(f"missing required input: {path.relative_to(REPO)}")

    frame, manual_metrics = build_release()
    metrics = validate_release(frame, manual_metrics)
    write_frame(frame, output)
    write_manifest(output, manifest, metrics)
    print(f"wrote {output.relative_to(REPO)} ({len(frame):,} rows)")
    print(f"wrote {manifest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    raise SystemExit(main(**vars(arguments)))
