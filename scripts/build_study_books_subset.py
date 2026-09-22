"""Build the seven-book study corpus from the validated full corpus.

Run from the oTree project directory:

    python -m scripts.build_study_books_subset

The output contains every available direct-source record for Romans,
1-2 Kings, 1-2 Samuel, and 1-2 Corinthians. It also includes only the
unfoldingWord topics and STEP lexicon entries linked from those records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from numpy import record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "corpus"
    / "processed"
    / "study_books"
    / "seven_books.jsonl"
)
DEFAULT_REPORT = DEFAULT_OUTPUT.with_name("seven_books_report.json")

STUDY_BOOKS = (
    "Romans",
    "1 Kings",
    "2 Kings",
    "1 Samuel",
    "2 Samuel",
    "1 Corinthians",
    "2 Corinthians",
)

EXPECTED_CHAPTERS = {
    "Romans": 16,
    "1 Kings": 22,
    "2 Kings": 25,
    "1 Samuel": 31,
    "2 Samuel": 24,
    "1 Corinthians": 16,
    "2 Corinthians": 13,
}

OLD_TESTAMENT_STUDY_BOOKS = {
    "1 Kings",
    "2 Kings",
    "1 Samuel",
    "2 Samuel",
}

PDF_COMMENTARY_SOURCE_IDS = {
    "COMM_ROMANS_SANDAY_HEADLAM",
    "COMM_KINGS_KEIL_DELITZSCH",
    "COMM_SAMUEL_HP_SMITH",
    "COMM_1COR_ROBERTSON_PLUMMER",
    "COMM_2COR_PLUMMER",
}

DIRECT_SOURCE_IDS = {
    "WEB",
    "OSHB",
    "UTN",
    "TOW",
    "OPENBIBLE_XREF",
    "BIBLE_ODYSSEY",
} | PDF_COMMENTARY_SOURCE_IDS

TW_REFERENCE_RE = re.compile(
    r"(?:rc://[^\s/]+/tw/dict/)?"
    r"(bible/(?:kt|names|other)/[A-Za-z0-9_-]+)",
    flags=re.I,
)
STRONG_RE = re.compile(r"\b[GH]\d{3,5}[a-z]?\b", flags=re.I)
UNAPPROVED_PERMISSION_VALUES = {
    "pending",
    "tbd",
    "todo",
    "written permission required",
}


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not record.get("chunk_id") or not str(record.get("text", "")).strip():
                raise ValueError(
                    f"Invalid corpus record on line {line_number} of {path}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Corpus contains no records: {path}")
    return records


def normalized_book(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def values_list(value: object) -> list[object]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def passage_tag_book(value: object) -> str | None:
    compact = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    for book in sorted(STUDY_BOOKS, key=len, reverse=True):
        book_key = book.casefold()
        if compact == book_key or compact.startswith(f"{book_key} "):
            return book
    return None


def matched_study_books(record: dict[str, Any]) -> tuple[str, ...]:
    """Return every selected book explicitly attached to one record."""
    keys: set[str] = set()

    if record.get("book"):
        keys.add(normalized_book(record["book"]))

    for tag in values_list(record.get("book_tags")):
        keys.add(normalized_book(tag))

    for tag in values_list(record.get("passage_tags")):
        book = passage_tag_book(tag)
        if book:
            keys.add(normalized_book(book))

    return tuple(
        book
        for book in STUDY_BOOKS
        if normalized_book(book) in keys
    )


def bible_odyssey_is_approved(record: dict[str, Any]) -> bool:
    """Require a substantive written-permission record before inclusion."""
    permission = re.sub(
        r"\s+",
        " ",
        str(record.get("permission_record", "")).strip().casefold(),
    )
    if not permission or permission in UNAPPROVED_PERMISSION_VALUES:
        return False
    if re.search(r"\b(?:pending|tbd|todo|awaiting)\b", permission):
        return False
    if re.search(r"\bnot\s+(?:yet\s+)?(?:approved|granted)\b", permission):
        return False
    return not permission.startswith("written permission required")

def normalize_tw_topic_path(value: object) -> str:
    path = str(value or "").strip().casefold().replace("\\", "/").rstrip("/")
    marker = "bible/"
    position = path.find(marker)
    return path[position:] if position >= 0 else path

def linked_topic_paths(records: Iterable[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for record in records:
        if record.get("source_id") != "UTN":
            continue
        for match in TW_REFERENCE_RE.finditer(str(record.get("text", ""))):
            paths.add(normalize_tw_topic_path(match.group(1)))
    return paths


def linked_strong_numbers(records: Iterable[dict[str, Any]]) -> set[str]:
    numbers: set[str] = set()
    for record in records:
        numbers.update(
            match.group(0).upper()
            for match in STRONG_RE.finditer(str(record.get("text", "")))
        )
    return numbers


def source_chapters(
    records: Iterable[dict[str, Any]],
    source_id: str,
    book: str,
) -> set[int]:
    book_key = normalized_book(book)
    chapters: set[int] = set()
    for record in records:
        if record.get("source_id") != source_id:
            continue
        if normalized_book(record.get("book")) != book_key:
            continue
        try:
            chapter = int(record.get("chapter"))
        except (TypeError, ValueError):
            continue
        if chapter >= 1:
            chapters.add(chapter)
    return chapters


def select_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_ids: set[str] = set()
    direct_counts_by_book = {
        book: Counter()
        for book in STUDY_BOOKS
    }
    excluded_unapproved_bible_odyssey = 0

    for record in records:
        source_id = str(record.get("source_id", ""))
        if source_id not in DIRECT_SOURCE_IDS:
            continue
        books = matched_study_books(record)
        if not books:
            continue
        if source_id == "BIBLE_ODYSSEY" and not bible_odyssey_is_approved(record):
            excluded_unapproved_bible_odyssey += 1
            continue
        selected_ids.add(str(record["chunk_id"]))
        for book in books:
            direct_counts_by_book[book][source_id] += 1

    directly_selected = [
        record
        for record in records
        if str(record.get("chunk_id")) in selected_ids
    ]
    topic_paths = linked_topic_paths(directly_selected)
    strong_numbers = linked_strong_numbers(directly_selected)

    for record in records:
        source_id = record.get("source_id")
        if source_id == "UTW":
            source_file = normalize_tw_topic_path(record.get("source_file", ""))            
            if source_file in topic_paths:
                selected_ids.add(str(record["chunk_id"]))
        elif source_id == "STEP":
            strong = str(record.get("strong_number", "")).upper()
            if strong in strong_numbers:
                selected_ids.add(str(record["chunk_id"]))

    selected = [
        record
        for record in records
        if str(record.get("chunk_id")) in selected_ids
    ]
    counts_by_source = Counter(
        str(record.get("source_id", ""))
        for record in selected
    )

    coverage: dict[str, dict[str, Any]] = {}
    coverage_problems: list[str] = []
    for book in STUDY_BOOKS:
        expected = set(range(1, EXPECTED_CHAPTERS[book] + 1))
        web_chapters = source_chapters(selected, "WEB", book)
        oshb_chapters = source_chapters(selected, "OSHB", book)
        missing_web = sorted(expected - web_chapters)
        missing_oshb = (
            sorted(expected - oshb_chapters)
            if book in OLD_TESTAMENT_STUDY_BOOKS
            else []
        )
        if missing_web:
            coverage_problems.append(
                f"WEB is missing chapters for {book}: {missing_web}"
            )
        if missing_oshb:
            coverage_problems.append(
                f"OSHB is missing chapters for {book}: {missing_oshb}"
            )
        coverage[book] = {
            "expected_chapters": EXPECTED_CHAPTERS[book],
            "web_chapters": sorted(web_chapters),
            "web_missing_chapters": missing_web,
            "oshb_chapters": sorted(oshb_chapters),
            "oshb_missing_chapters": missing_oshb,
        }

    if coverage_problems:
        raise ValueError(
            "Study-book corpus coverage failed:\n- "
            + "\n- ".join(coverage_problems)
        )

    report = {
        "study_books": list(STUDY_BOOKS),
        "source_corpus_chunks": len(records),
        "selected_chunks": len(selected),
        "chunks_by_source": dict(sorted(counts_by_source.items())),
        "direct_chunks_by_book": {
            book: dict(sorted(counts.items()))
            for book, counts in direct_counts_by_book.items()
        },
        "chapter_coverage": coverage,
        "linked_translation_word_topics": sorted(topic_paths),
        "linked_strong_numbers": sorted(strong_numbers),
        "excluded_unapproved_bible_odyssey_chunks": (
            excluded_unapproved_bible_odyssey
        ),
        "selection_policy": {
            "direct_sources": sorted(DIRECT_SOURCE_IDS),
            "full_selected_books": True,
            "linked_utw_topics": True,
            "linked_step_entries": True,
            "unlinked_global_lexicon_records_excluded": True,
            "bible_odyssey_requires_permission": True,
        },
    }
    return selected, report


def write_outputs(
    records: list[dict[str, Any]],
    output: Path,
    report_path: Path,
    report: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(output)

    report["output_file"] = str(output)
    report["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = project_path(args.source)
    output = project_path(args.output)
    report_path = project_path(args.report)
    if not source.exists():
        raise SystemExit(f"ERROR: source corpus not found: {source}")

    records = read_jsonl(source)
    selected, report = select_records(records)
    write_outputs(selected, output, report_path, report)

    print("STUDY-BOOK CORPUS COMPLETE")
    print(f"Full corpus chunks: {len(records):,}")
    print(f"Selected study-book chunks: {len(selected):,}")
    print("Books: " + ", ".join(STUDY_BOOKS))
    for source_id, count in report["chunks_by_source"].items():
        print(f"  {source_id}: {count:,}")
    print(f"Output: {output}")
    print(f"Report: {report_path}")
    print(f"SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
