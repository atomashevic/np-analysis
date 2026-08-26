"""Assemble the public OSF reproducibility package.

Collects the de-identified event table, documentation, privacy-screened code,
and aggregate outputs into ``nisamprijavila-osf-deidentified.zip`` at the
repository root.

The builder withholds files carrying disclosure text or source identities. A
second content audit compares every staged text file with identifiers observed
in the restricted source archive and fails before creating the ZIP if a match
survives screening.

Usage:
    uv run python scripts/build_osf_package.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Column names that make a table unsafe to publish.
FREE_TEXT = {
    "text",
    "full_text",
    "tweet_text",
    "content",
    "body",
    "excerpt",
    "quote",
    "nonreporting_motives_raw",
}
ACCOUNT = {
    "author_username",
    "username",
    "screen_name",
    "user_id",
    "author_id",
    "from_user",
    "from_user_id_str",
}
CODED = {
    "confession_flag",
    "experience_owner",
    "victim_gender",
    "violence_sn",
    "violence_su",
    "reporting_status",
}
PER_TWEET_ID = {
    "tweet_id",
    "post_id",
    "source_tweet_id",
    "source_post_id",
    "resolved_tweet_id",
    "workbook_tweet_id",
    "workbook_row",
}

PUBLISHABLE_SUFFIXES = {".csv"}
AUDITABLE_SUFFIXES = {".csv", ".json", ".md", ".py", ".svg", ".toml", ".txt"}
HANDLE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9_]{3,15})(?![A-Za-z0-9_])")
NUMBER_TOKEN = re.compile(r"(?<![0-9])([0-9]{6,20})(?![0-9])")
HANDLE_AUDIT_EXCEPTIONS = {"nisamprijavila"}
SCRIPT_EXCLUDES = {
    Path("manual_coding_reconciliation.py"),
    Path("retweet_phase_analysis.py"),
    Path("build_phase_network_composite.py"),
}


@dataclass(frozen=True)
class SourceIdentities:
    """Restricted identifiers used only to audit the staged public package."""

    numeric_ids: frozenset[str]
    handles: frozenset[str]


def load_source_identities(path: Path) -> SourceIdentities:
    """Load source IDs and handles without writing an identity vocabulary."""
    numeric_ids: set[str] = set()
    handles: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for row in csv.DictReader(stream):
            for column in (
                "id_str",
                "from_user_id_str",
                "in_reply_to_status_id_str",
                "in_reply_to_user_id_str",
            ):
                value = str(row.get(column, "")).strip()
                if value:
                    numeric_ids.add(value)
            for column in ("from_user", "in_reply_to_screen_name"):
                value = str(row.get(column, "")).strip().casefold()
                if value:
                    handles.add(value)
            try:
                entities = json.loads(str(row.get("entities_str", "")))
            except (json.JSONDecodeError, TypeError):
                continue
            for mention in entities.get("user_mentions", []):
                numeric = str(mention.get("id_str", mention.get("id", ""))).strip()
                handle = str(mention.get("screen_name", "")).strip().casefold()
                if numeric:
                    numeric_ids.add(numeric)
                if handle:
                    handles.add(handle)
    handles.difference_update(HANDLE_AUDIT_EXCEPTIONS)
    return SourceIdentities(frozenset(numeric_ids), frozenset(handles))


def identity_hits(path: Path, identities: SourceIdentities) -> list[str]:
    """Return source identifiers found as complete tokens in a text file."""
    if path.suffix.lower() not in AUDITABLE_SUFFIXES:
        return []
    content = path.read_text(encoding="utf-8", errors="ignore")
    numeric_hits = {
        value
        for value in NUMBER_TOKEN.findall(content)
        if value in identities.numeric_ids
    }
    handle_hits = {
        value
        for value in HANDLE_TOKEN.findall(content)
        if value.casefold() in identities.handles
    }
    return sorted(numeric_hits.union(handle_hits), key=str.casefold)


def screen(path: Path, identities: SourceIdentities) -> tuple[bool, str]:
    """Return (is_publishable, reason_if_withheld) for one aggregate table."""
    if path.suffix.lower() == ".csv":
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                header = next(csv.reader(fh), [])
        except OSError:
            return False, "unreadable"
        cols = {c.strip().lower() for c in header}
        for group, reason in (
            (FREE_TEXT, "free-text disclosure content"),
            (ACCOUNT, "identifies accounts"),
            (CODED, "per-event sensitive annotations"),
            (PER_TWEET_ID, "source post identifier"),
        ):
            hit = cols & group
            if hit:
                return False, f"{reason} ({', '.join(sorted(hit))})"
    hits = identity_hits(path, identities)
    if hits:
        return False, f"source identities ({', '.join(hits[:3])})"
    return True, ""


def copy_public_scripts(
    destination: Path,
    identities: SourceIdentities,
) -> list[tuple[Path, str]]:
    """Copy Python scripts that pass the source-identity audit."""
    withheld: list[tuple[Path, str]] = []
    scripts_root = REPO / "scripts"
    for source in sorted(scripts_root.rglob("*.py")):
        relative = source.relative_to(scripts_root)
        if relative in SCRIPT_EXCLUDES:
            withheld.append((source, "explicit privacy or dependency exclusion"))
            continue
        hits = identity_hits(source, identities)
        if hits:
            withheld.append((source, f"source identities ({', '.join(hits[:3])})"))
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return withheld


def audit_stage(stage: Path, identities: SourceIdentities) -> None:
    """Fail if a source identifier remains in any auditable staged file."""
    failures = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            hits = identity_hits(path, identities)
            if hits:
                failures.append((path.relative_to(stage), hits[:3]))
    if failures:
        summary = "; ".join(
            f"{path}: {', '.join(hits)}" for path, hits in failures[:10]
        )
        raise RuntimeError(f"source identity audit failed: {summary}")


def build(dry_run: bool = False, force: bool = False) -> int:
    src_dir = REPO / "osf_package"
    identities = load_source_identities(REPO / "data" / "np.csv")
    # (source, destination within the package)
    STATIC = [
        (src_dir / "README.md", "README.md"),
        (src_dir / "REPRODUCE.md", "REPRODUCE.md"),
        (src_dir / "data" / "deidentified_events.csv", "data/deidentified_events.csv"),
        (
            src_dir / "data" / "deidentified_events.schema.md",
            "data/deidentified_events.schema.md",
        ),
        (
            src_dir / "data" / "deidentified_events.manifest.json",
            "data/deidentified_events.manifest.json",
        ),
        (src_dir / "data" / "motive_codebook.csv", "data/motive_codebook.csv"),
    ]
    for required, _ in STATIC:
        if not required.exists():
            sys.exit(f"missing required input: {required.relative_to(REPO)}")

    kept: list[Path] = []
    withheld: list[tuple[Path, str]] = []
    for src in sorted((REPO / "analysis").rglob("*")):
        if not src.is_file() or src.suffix.lower() not in PUBLISHABLE_SUFFIXES:
            continue
        ok, reason = screen(src, identities)
        (kept if ok else withheld).append(src if ok else (src, reason))

    print(f"derived outputs: {len(kept)} included, {len(withheld)} withheld")
    for src, reason in withheld:
        print(f"  withheld  {src.relative_to(REPO)}  ->  {reason}")
    if dry_run:
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "osf"
        (stage / "data").mkdir(parents=True)
        for src, rel in STATIC:
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        script_withheld = copy_public_scripts(stage / "code" / "scripts", identities)
        for source, reason in script_withheld:
            print(f"  withheld  {source.relative_to(REPO)}  ->  {reason}")
        for meta in ("pyproject.toml", "uv.lock"):
            if (REPO / meta).exists():
                shutil.copy2(REPO / meta, stage / "code" / meta)

        for src in kept:
            dst = stage / "derived_outputs" / src.relative_to(REPO / "analysis")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        audit_stage(stage, identities)
        out = REPO / "nisamprijavila-osf-deidentified.zip"
        if out.exists() and not force:
            raise FileExistsError(f"refusing to overwrite existing package: {out}")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(stage.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(stage.parent))

    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out.relative_to(REPO)}  ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be included and withheld",
    )
    ap.add_argument("--force", action="store_true", help="replace the output ZIP")
    raise SystemExit(build(**vars(ap.parse_args())))
