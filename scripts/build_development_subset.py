"""Build a passage-focused development corpus from the validated full corpus.

Run from the oTree project directory:

    python -m scripts.build_development_subset

The output contains the five study/practice passages, directly overlapping
Bible text/notes/commentary/morphology, Theology of Work book-level summaries,
and lexical/topic records explicitly linked from those selected chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "corpus"
    / "processed"
    / "development"
    / "five_passages.jsonl"
)
DEFAULT_REPORT = DEFAULT_OUTPUT.with_name("five_passages_report.json")

PASSAGES = (
    ("2 Kings", 5, 9, 14, "Block 1 — 2 Kings 5:9–14"),
    ("Romans", 12, 1, 5, "Block 1 — Romans 12:1–5"),
    ("1 Samuel", 8, 4, 9, "Block 2 — 1 Samuel 8:4–9"),
    ("1 Corinthians", 8, 1, 6, "Block 2 — 1 Corinthians 8:1–6"),
    ("Mark", 4, 35, 41, "Practice — Mark 4:35–41"),
)

DIRECT_SOURCE_IDS = {"WEB", "OSHB", "UTN", "TOW"}
TW_REFERENCE_RE = re.compile(
    r"(?:rc://[^\s/]+/tw/dict/)?"
    r"(bible/(?:kt|names|other)/[A-Za-z0-9_-]+)",
    flags=re.I,
)
STRONG_RE = re.compile(r"\b[GH]\d{3,5}[a-z]?\b", flags=re.I)


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
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
                raise ValueError(f"Invalid corpus record on line {line_number} of {path}")
            records.append(record)
    if not records:
        raise ValueError(f"Corpus contains no records: {path}")
    return records


def normalized_book(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def verse_overlap(record: dict, start: int, end: int) -> bool:
    record_start = record.get("verse_start")
    record_end = record.get("verse_end")
    if record_start is None:
        # Chapter-level introductions or summaries are relevant to the chapter.
        return True
    try:
        record_start = int(record_start)
        record_end = int(record_end if record_end is not None else record_start)
    except (TypeError, ValueError):
        return False
    return record_start <= end and record_end >= start


def matches_passage(record: dict, passage: tuple) -> bool:
    book, chapter, start, end, _label = passage
    if normalized_book(record.get("book")) != normalized_book(book):
        return False
    try:
        record_chapter = int(record.get("chapter"))
    except (TypeError, ValueError):
        return False
    return record_chapter == chapter and verse_overlap(record, start, end)


def is_tow_book_summary(record: dict) -> bool:
    if record.get("source_id") != "TOW" or record.get("chapter") is not None:
        return False
    target_books = {normalized_book(item[0]) for item in PASSAGES}
    return normalized_book(record.get("book")) in target_books


def linked_topic_paths(records: list[dict]) -> set[str]:
    paths: set[str] = set()
    for record in records:
        if record.get("source_id") != "UTN":
            continue
        for match in TW_REFERENCE_RE.finditer(str(record.get("text", ""))):
            paths.add(match.group(1).casefold().rstrip("/"))
    return paths


def linked_strong_numbers(records: list[dict]) -> set[str]:
    numbers: set[str] = set()
    for record in records:
        if record.get("source_id") != "OSHB":
            continue
        numbers.update(match.group(0).upper() for match in STRONG_RE.finditer(record["text"]))
    return numbers


def select_records(records: list[dict]) -> tuple[list[dict], dict]:
    selected_ids: set[str] = set()
    passage_counts: dict[str, Counter] = {}

    for passage in PASSAGES:
        label = passage[4]
        counts: Counter = Counter()
        for record in records:
            if record.get("source_id") not in DIRECT_SOURCE_IDS:
                continue
            if matches_passage(record, passage):
                selected_ids.add(record["chunk_id"])
                counts[str(record.get("source_id"))] += 1
        passage_counts[label] = counts

    for record in records:
        if is_tow_book_summary(record):
            selected_ids.add(record["chunk_id"])

    directly_selected = [
        record for record in records if record.get("chunk_id") in selected_ids
    ]
    topic_paths = linked_topic_paths(directly_selected)
    strong_numbers = linked_strong_numbers(directly_selected)

    for record in records:
        source_id = record.get("source_id")
        if source_id == "UTW":
            source_file = str(record.get("source_file", "")).casefold().rstrip("/")
            if source_file in topic_paths:
                selected_ids.add(record["chunk_id"])
        elif source_id == "STEP":
            strong = str(record.get("strong_number", "")).upper()
            if strong in strong_numbers:
                selected_ids.add(record["chunk_id"])

    selected = [record for record in records if record["chunk_id"] in selected_ids]
    counts_by_source = Counter(str(record.get("source_id")) for record in selected)

    missing_web = [
        passage[4]
        for passage in PASSAGES
        if passage_counts[passage[4]].get("WEB", 0) == 0
    ]
    if missing_web:
        raise ValueError(
            "Development subset is missing WEB coverage for: " + ", ".join(missing_web)
        )

    report = {
        "source_corpus_chunks": len(records),
        "selected_chunks": len(selected),
        "chunks_by_source": dict(sorted(counts_by_source.items())),
        "passage_direct_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in passage_counts.items()
        },
        "linked_translation_word_topics": sorted(topic_paths),
        "linked_strong_numbers": sorted(strong_numbers),
        "selection_policy": {
            "direct_sources": sorted(DIRECT_SOURCE_IDS),
            "tow_book_level_summaries": True,
            "linked_utw_topics": True,
            "linked_step_entries_from_oshb": True,
            "unlinked_global_lexicon_records_excluded": True,
        },
    }
    return selected, report


def write_outputs(records: list[dict], output: Path, report_path: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
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

    print("DEVELOPMENT SUBSET COMPLETE")
    print(f"Full corpus chunks: {len(records):,}")
    print(f"Development chunks: {len(selected):,}")
    for source_id, count in report["chunks_by_source"].items():
        print(f"  {source_id}: {count:,}")
    print(f"Output: {output}")
    print(f"Report: {report_path}")
    print(f"SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
