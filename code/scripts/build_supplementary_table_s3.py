"""Build the reconciled Supplementary Table S3 as Pandoc-ready Markdown."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "manual_coding_motives"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "manuscript_revision"
    / "supplementary_table_s3_proposed.md"
)


EXCERPTS = {
    "1": "When I did report, the police could do nothing; the abuser faced no consequences while I was left exposed.",
    "2": "I knew our system would not protect me and that they would say I had chosen it myself.",
    "3": "I thought nobody would believe me and that they would say I was exaggerating or had misunderstood.",
    "3.1": "People asked how a boyfriend you live with could rape you.",
    "3.2": "I thought they would say I was exaggerating and that it was not that serious.",
    "4": "I repressed the trauma and moved on, thinking that perhaps it was not such a big deal.",
    "5": "I did not understand that what had happened to me was violence.",
    "5.1": "I realised only later that I had not known I was supposed to report it.",
    "5.2": "I did not report because I did not understand what had happened to me.",
    "5.3": "I did not report because I had no one to turn to and did not know whom to contact.",
    "5.4": "I was a child and did not know what I was supposed to do except try to survive.",
    "5.5": "A boy cannot be a victim of domestic violence; even when he is, he must prove it repeatedly.",
    "6": "I was afraid of what he might do if he learned that I had reported him.",
    "6.1": "I still do not have the courage to write my account; accepting and disclosing it is extremely difficult.",
    "6.2": "I did not report because I feared retaliation if he found out.",
    "6.3": "I was afraid of my parents' reaction because I had agreed to let him drive me home.",
    "7": "I was ashamed and kept wondering whether I had provoked it and how much was my fault.",
    "8": "I was told that he was a respected member of society and that I should not speak about it.",
    "9": "I moved away so that he would not find me.",
    "10": "He convinced me that it was my fault and that I was worthless.",
    "11": "I was ashamed, and now I am ashamed that I felt ashamed.",
    "12": "I was alone, without support, and with no way for anyone to help me.",
    "12.1": "The club was full, including police and security, but everyone stayed silent, so I stayed silent too.",
    "12.2": "I was alone and had nobody I could tell or ask for help.",
    "14": "He had connections in the police, so I believed nobody would help me.",
    "15": "How could I prove threats, bruises that had faded, psychological torture, or carefully hidden violence?",
    "16": "I knew nobody would believe me, and later I would no longer be able to prove it.",
    "17": "My roommate blamed me, said it was my fault, and threw me out of the apartment.",
    "18": "I did not report because he was a close relative.",
    "19": "I could not bear to hurt my parents after everything they had done for me.",
    "19.1": "I did not want to break my parents' hearts or expose them to shame and pain.",
    "19.2": "I knew my father and brother would retaliate and then face criminal charges.",
    "20": "He promised that it would never happen again.",
    "22": "He was my professor, and my academic year and graduation depended on him.",
    "23": "I wanted to forget it and move on rather than be treated as a victim.",
    "24": "I did not report because I did not want to ruin someone's life.",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_count(row: dict[str, str], prefix: str) -> str:
    count = int(row[f"{prefix}_n"])
    percentage = float(row[f"{prefix}_pct"]) * 100
    return f"{count:,} ({percentage:.1f}%)"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def validate_reason_rows(rows: list[dict[str, str]]) -> None:
    parent_rows = [row for row in rows if row["row_type"] == "parent"]
    subcode_rows = [row for row in rows if row["row_type"] == "subcode"]
    if len(parent_rows) != 22 or len(subcode_rows) != 14:
        raise ValueError("Expected 22 parent categories and 14 subcodes")
    if sum(int(row["assignment_n"]) for row in parent_rows) != 1238:
        raise ValueError("Parent-category assignments must sum to 1,238")
    if any(int(row["assignment_denominator"]) != 1238 for row in parent_rows):
        raise ValueError("Unexpected parent-category assignment denominator")
    if any(int(row["assignment_denominator"]) != 1266 for row in subcode_rows):
        raise ValueError("Unexpected detailed-code assignment denominator")
    if any(int(row["unique_tweets_denominator"]) != 824 for row in rows):
        raise ValueError("Unexpected unique-tweet denominator")
    missing = [row["code"] for row in rows if row["code"] not in EXCERPTS]
    if missing:
        raise ValueError(f"Missing illustrative excerpts for: {missing}")


def reason_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| Code | Analytic category or subcode | Assignments, n (%) | Unique tweets, n (% of 824) | Translated illustrative excerpt |",
        "|:---:|---|---:|---:|---|",
    ]
    for row in rows:
        label = row["label"]
        if row["row_type"] == "subcode":
            label = f"Subcode of {row['parent_code']}: {label}"
        values = [
            row["code"],
            label,
            format_count(row, "assignment"),
            format_count(row, "unique_tweets"),
            EXCERPTS[row["code"]],
        ]
        lines.append("| " + " | ".join(escape_cell(value) for value in values) + " |")
    return lines


def exclusion_table(rows: list[dict[str, str]]) -> list[str]:
    labels = {
        "x": "Another person's account",
        "0": "No stated reason",
        "unresolved": "Unresolved code",
        "unlisted": "Unlisted code (21)",
        "blank": "Blank reason field",
    }
    lines = [
        "| Non-analytic exclusion | Coding units, n | Parsed assignments, n |",
        "|---|---:|---:|",
    ]
    for row in rows:
        label = labels[row["exclusion_type"]]
        lines.append(
            f"| {label} | {int(row['coding_units_n']):,} | "
            f"{int(row['parsed_assignments_n']):,} |"
        )
    return lines


def build_markdown(
    reason_rows: list[dict[str, str]], exclusion_rows: list[dict[str, str]]
) -> str:
    lines = [
        "# Supplementary Table S3. Reconciled manual reasons for non-reporting",
        "",
        "Parent categories and subcodes are shown with reconciled frequencies. Parent-category counts define the manuscript analysis and count each tweet once per parent, even when several detailed codes map to the same parent. Subcode rows describe the more specific codes retained in the workbook. Parent percentages use 1,238 tweet-parent assignments; subcode percentages use 1,266 detailed-code assignments. Unique-tweet percentages use 824 analytic reason tweets.",
        "",
        *reason_table(reason_rows),
        "",
        "## Non-analytic exclusions",
        "",
        *exclusion_table(exclusion_rows),
        "",
        "*Note.* Rows coded `x` and `0`, unresolved codes, and the unlisted code `21` are excluded from all reason analyses. The 64 `x` rows remain excluded even when first-person ownership was retained for the separate disclosure-composition analysis. Manual adjudication of the 28 `x`–`experience_owner = 1` conflicts retained 19 as first-person or mixed accounts and reclassified nine; that decision does not change the 824-tweet denominator, the 1,266 detailed-code assignments, or the 1,238 deduplicated tweet-parent assignments. Direct parent codes mean that subcode counts need not sum to their parent count.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    reason_rows = read_rows(ANALYSIS_DIR / "supplementary_table_s3_reason_codes.csv")
    exclusion_rows = read_rows(ANALYSIS_DIR / "supplementary_table_s3_exclusions.csv")
    validate_reason_rows(reason_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        build_markdown(reason_rows, exclusion_rows), encoding="utf-8"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
