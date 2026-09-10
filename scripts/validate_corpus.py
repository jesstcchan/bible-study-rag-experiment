"""Validate corpus/processed/chunks.jsonl before creating embeddings.

Run from the oTree project directory:

    python -m scripts.validate_corpus
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
TOW_RAW_FILE = PROJECT_ROOT / "corpus" / "raw" / "theology_of_work" / "pages.jsonl"
TOW_CRAWL_MANIFEST = (
    PROJECT_ROOT / "corpus" / "raw" / "theology_of_work" / "crawl_manifest.json"
)
BIBLE_ODYSSEY_RAW_FILE = (
    PROJECT_ROOT / "corpus" / "raw" / "bible_odyssey" / "pages.jsonl"
)
SOURCE_MANIFEST = PROJECT_ROOT / "corpus" / "source_manifest.csv"
REQUIRED_SOURCES = {
    "WEB",
    "STEP",
    "OSHB",
    "UTN",
    "UTW",
    "TOW",
    "OPENBIBLE_XREF",
}
OPTIONAL_SOURCES = {"BIBLE_ODYSSEY"}
REQUIRED_FIELDS = {
    "chunk_id",
    "source_id",
    "source_name",
    "source_type",
    "title",
    "text",
    "version",
    "license",
    "url",
}
MAX_EXPECTED_CHARS = 3_500
REQUIRED_TOW_FIELDS = {
    "source_page_id",
    "canonical_url",
    "page_chunk_number",
    "page_chunk_count",
    "book_tags",
    "hierarchy_role",
    "producer",
    "license_url",
    "accessed_at",
    "scraper_version",
    "original_text_sha256",
    "inline_quoted_spans_removed_from_page",
    "table_source_cells_removed_from_page",
}
REQUIRED_OPENBIBLE_FIELDS = {
    "from_reference",
    "cross_reference_count",
    "highest_vote_count",
    "scripture_text_source",
}
REQUIRED_BIBLE_ODYSSEY_FIELDS = {
    "source_page_id",
    "canonical_url",
    "book_tags",
    "passage_tags",
    "accessed_at",
    "original_html_sha256",
    "original_text_sha256",
    "license_url",
    "permission_record",
}

STUDY_PASSAGES = [
    {
        "block": 1,
        "book": "2 Kings",
        "chapter": 5,
        "verse_start": 9,
        "verse_end": 14,
        "require_oshb": True,
        "require_tow": True,
    },
    {
        "block": 1,
        "book": "Romans",
        "chapter": 12,
        "verse_start": 1,
        "verse_end": 5,
        "require_oshb": False,
        "require_tow": True,
    },
    {
        "block": 2,
        "book": "1 Samuel",
        "chapter": 8,
        "verse_start": 4,
        "verse_end": 9,
        "require_oshb": True,
        "require_tow": True,
    },
    {
        "block": 2,
        "book": "1 Corinthians",
        "chapter": 8,
        "verse_start": 1,
        "verse_end": 6,
        "require_oshb": False,
        # The scraped TOW Bible Commentary contains 1 Corinthians material,
        # but no page explicitly covering 1 Corinthians 8:1-6.
        "require_tow": False,
    },
    {
        "block": "Practice",
        "book": "Mark",
        "chapter": 4,
        "verse_start": 35,
        "verse_end": 41,
        "require_oshb": False,
        "require_tow": True,
    },
]


def overlapping_records(records: list[dict], passage: dict, source_id: str) -> list[dict]:
    """Return source records whose verse interval overlaps a study passage."""
    matches = []
    for record in records:
        if record.get("source_id") != source_id:
            continue
        if record.get("book") != passage["book"]:
            continue
        if record.get("chapter") != passage["chapter"]:
            continue
        start = record.get("verse_start")
        end = record.get("verse_end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end >= passage["verse_start"] and start <= passage["verse_end"]:
            matches.append(record)
    return matches


def covered_verses(records: list[dict], passage: dict, source_id: str) -> set[int]:
    """Return the target verse numbers covered by a source."""
    target = set(range(passage["verse_start"], passage["verse_end"] + 1))
    covered: set[int] = set()
    for record in overlapping_records(records, passage, source_id):
        start = max(record["verse_start"], passage["verse_start"])
        end = min(record["verse_end"], passage["verse_end"])
        covered.update(range(start, end + 1))
    return covered & target


def read_tow_crawl_manifest() -> dict:
    if not TOW_CRAWL_MANIFEST.exists():
        raise SystemExit(f"ERROR: file not found: {TOW_CRAWL_MANIFEST}")
    try:
        return json.loads(TOW_CRAWL_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"ERROR: invalid TOW crawl manifest: {error}") from error


def read_tow_source_manifest_row() -> dict[str, str] | None:
    if not SOURCE_MANIFEST.exists():
        return None
    with SOURCE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_id") == "TOW":
                return row
    return None


def main() -> None:
    if not CORPUS_FILE.exists():
        raise SystemExit(f"ERROR: file not found: {CORPUS_FILE}")

    records: list[dict] = []
    invalid_json: list[int] = []
    missing_fields: list[tuple[int, list[str]]] = []
    empty_text: list[int] = []
    oversized: list[tuple[int, int]] = []

    with CORPUS_FILE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json.append(line_number)
                continue

            records.append(record)
            missing = sorted(REQUIRED_FIELDS - record.keys())
            if missing:
                missing_fields.append((line_number, missing))
            text = str(record.get("text", "")).strip()
            if not text:
                empty_text.append(line_number)
            if len(text) > MAX_EXPECTED_CHARS:
                oversized.append((line_number, len(text)))

    id_counts = Counter(record.get("chunk_id") for record in records)
    duplicate_ids = sorted(
        chunk_id for chunk_id, count in id_counts.items()
        if chunk_id is not None and count > 1
    )
    source_counts = Counter(record.get("source_id") for record in records)
    bible_odyssey_raw_available = (
        BIBLE_ODYSSEY_RAW_FILE.is_file()
        and BIBLE_ODYSSEY_RAW_FILE.stat().st_size > 0
    )
    expected_sources = set(REQUIRED_SOURCES)
    if bible_odyssey_raw_available:
        expected_sources.add("BIBLE_ODYSSEY")
    known_sources = REQUIRED_SOURCES | OPTIONAL_SOURCES
    missing_sources = sorted(expected_sources - set(source_counts))
    unexpected_sources = sorted(set(source_counts) - known_sources)

    openbible_chunks = [
        record for record in records
        if record.get("source_id") == "OPENBIBLE_XREF"
    ]
    openbible_missing_fields = [
        (str(record.get("chunk_id")), sorted(REQUIRED_OPENBIBLE_FIELDS - record.keys()))
        for record in openbible_chunks
        if REQUIRED_OPENBIBLE_FIELDS - record.keys()
    ]

    bible_odyssey_chunks = [
        record for record in records
        if record.get("source_id") == "BIBLE_ODYSSEY"
    ]
    bible_odyssey_missing_fields = [
        (
            str(record.get("chunk_id")),
            sorted(REQUIRED_BIBLE_ODYSSEY_FIELDS - record.keys()),
        )
        for record in bible_odyssey_chunks
        if REQUIRED_BIBLE_ODYSSEY_FIELDS - record.keys()
    ]
    bible_odyssey_missing_permission = [
        str(record.get("chunk_id"))
        for record in bible_odyssey_chunks
        if not str(record.get("permission_record", "")).strip()
    ]

    tow_chunks = [
        record for record in records
        if record.get("source_id") == "TOW"
    ]
    tow_page_ids = {
        str(record.get("source_page_id"))
        for record in tow_chunks
        if record.get("source_page_id")
    }
    tow_missing_fields = [
        (str(record.get("chunk_id")), sorted(REQUIRED_TOW_FIELDS - record.keys()))
        for record in tow_chunks
        if REQUIRED_TOW_FIELDS - record.keys()
    ]
    tow_invalid_licenses = [
        str(record.get("chunk_id"))
        for record in tow_chunks
        if record.get("license") != "CC BY-NC 4.0"
    ]
    tow_invalid_urls = [
        str(record.get("chunk_id"))
        for record in tow_chunks
        if not str(record.get("canonical_url", "")).startswith(
            "https://www.theologyofwork.org/"
        )
    ]
    tow_page_chunks: dict[str, list[dict]] = {}
    for record in tow_chunks:
        page_id = str(record.get("source_page_id", ""))
        if page_id:
            tow_page_chunks.setdefault(page_id, []).append(record)
    tow_sequence_errors: list[str] = []
    for page_id, page_records in tow_page_chunks.items():
        totals = {record.get("page_chunk_count") for record in page_records}
        numbers = {record.get("page_chunk_number") for record in page_records}
        if len(totals) != 1 or not all(isinstance(value, int) for value in totals):
            tow_sequence_errors.append(page_id)
            continue
        expected_numbers = set(range(1, int(next(iter(totals))) + 1))
        if numbers != expected_numbers:
            tow_sequence_errors.append(page_id)

    tow_crawl_manifest = read_tow_crawl_manifest()
    expected_tow_pages = int(tow_crawl_manifest.get("accepted_count", 0))
    tow_source_row = read_tow_source_manifest_row()
    raw_tow_sha256 = (
        hashlib.sha256(TOW_RAW_FILE.read_bytes()).hexdigest()
        if TOW_RAW_FILE.exists()
        else ""
    )

    print(f"Corpus file: {CORPUS_FILE}")
    print(f"Total chunks: {len(records):,}")
    print("Chunks by source:")
    for source_id, count in sorted(source_counts.items()):
        print(f"  {source_id}: {count:,}")
    if not bible_odyssey_raw_available:
        print(
            "  BIBLE_ODYSSEY: not yet expected because "
            "corpus/raw/bible_odyssey/pages.jsonl is empty"
        )

    print("Study-passage coverage checks:")
    passage_results = []
    for passage in STUDY_PASSAGES:
        label = (
            f"{passage['book']} {passage['chapter']}:"
            f"{passage['verse_start']}-{passage['verse_end']}"
        )
        target = set(range(passage["verse_start"], passage["verse_end"] + 1))
        web_covered = covered_verses(records, passage, "WEB")
        utn_matches = overlapping_records(records, passage, "UTN")
        oshb_covered = covered_verses(records, passage, "OSHB")
        tow_matches = overlapping_records(records, passage, "TOW")
        book_tag = passage["book"].lower().replace(" ", "-")
        tow_book_chunks = [
            record for record in tow_chunks
            if book_tag in record.get("book_tags", [])
        ]
        passage_results.append(
            (
                passage,
                label,
                target,
                web_covered,
                utn_matches,
                oshb_covered,
                tow_matches,
                tow_book_chunks,
            )
        )
        print(f"  Block {passage['block']} — {label}")
        print(f"    WEB verses: {len(web_covered)}/{len(target)}")
        print(f"    UTN overlapping chunks: {len(utn_matches)}")
        if passage["require_oshb"]:
            print(f"    OSHB verses: {len(oshb_covered)}/{len(target)}")
        else:
            print("    OSHB: not applicable (New Testament passage)")
        if tow_matches:
            print(f"    TOW directly overlapping chunks: {len(tow_matches)}")
        elif passage["require_tow"]:
            print("    TOW directly overlapping chunks: 0 (required)")
        else:
            print(
                "    TOW directly overlapping chunks: 0 "
                f"(not required; {len(tow_book_chunks)} book-level chunks available)"
            )
    print("Theology of Work corpus checks:")
    print(f"  Source pages represented: {len(tow_page_ids):,}/{expected_tow_pages:,}")
    print(f"  TOW chunks: {len(tow_chunks):,}")
    print(f"  Crawl status: {tow_crawl_manifest.get('status')}")
    print(f"  Crawl pending pages: {tow_crawl_manifest.get('pending_count')}")

    problems: list[str] = []
    if invalid_json:
        problems.append(f"invalid JSON lines: {invalid_json[:10]}")
    if missing_fields:
        problems.append(f"records with missing fields: {missing_fields[:10]}")
    if empty_text:
        problems.append(f"empty-text lines: {empty_text[:10]}")
    if oversized:
        problems.append(f"oversized chunks: {oversized[:10]}")
    if duplicate_ids:
        problems.append(f"duplicate chunk IDs: {duplicate_ids[:10]}")
    if missing_sources:
        problems.append(f"missing expected sources: {missing_sources}")
    if unexpected_sources:
        problems.append(f"unexpected sources: {unexpected_sources}")
    if openbible_missing_fields:
        problems.append(
            "OpenBible chunks with missing metadata: "
            f"{openbible_missing_fields[:10]}"
        )
    if bible_odyssey_missing_fields:
        problems.append(
            "Bible Odyssey chunks with missing metadata: "
            f"{bible_odyssey_missing_fields[:10]}"
        )
    if bible_odyssey_missing_permission:
        problems.append(
            "Bible Odyssey chunks without a written-permission record: "
            f"{bible_odyssey_missing_permission[:10]}"
        )
    if tow_missing_fields:
        problems.append(f"TOW chunks with missing metadata: {tow_missing_fields[:10]}")
    if tow_invalid_licenses:
        problems.append(f"TOW chunks with unexpected licences: {tow_invalid_licenses[:10]}")
    if tow_invalid_urls:
        problems.append(f"TOW chunks with invalid canonical URLs: {tow_invalid_urls[:10]}")
    if tow_sequence_errors:
        problems.append(f"TOW page chunk sequences are invalid: {tow_sequence_errors[:10]}")
    if tow_crawl_manifest.get("status") not in {"complete", "complete_with_rejections"}:
        problems.append(f"TOW crawl is not complete: {tow_crawl_manifest.get('status')}")
    if tow_crawl_manifest.get("pending_count") != 0:
        problems.append(
            f"TOW crawl still has pending pages: {tow_crawl_manifest.get('pending_count')}"
        )
    if expected_tow_pages <= 0:
        problems.append("TOW crawl manifest has no accepted pages")
    if len(tow_page_ids) != expected_tow_pages:
        problems.append(
            "processed TOW page count does not match crawl manifest: "
            f"{len(tow_page_ids)} != {expected_tow_pages}"
        )
    if tow_source_row is None:
        problems.append("source_manifest.csv has no TOW row")
    else:
        if tow_source_row.get("license") != "CC BY-NC 4.0":
            problems.append("source_manifest.csv has an unexpected TOW licence")
        if tow_source_row.get("file_hash") != raw_tow_sha256:
            problems.append("TOW pages.jsonl hash does not match source_manifest.csv")
    for (
        passage,
        label,
        target,
        web_covered,
        utn_matches,
        oshb_covered,
        tow_matches,
        tow_book_chunks,
    ) in passage_results:
        missing_web = sorted(target - web_covered)
        if missing_web:
            problems.append(f"WEB is missing verses for {label}: {missing_web}")
        if not utn_matches:
            problems.append(f"UTN has no overlapping notes for {label}")
        if passage["require_oshb"]:
            missing_oshb = sorted(target - oshb_covered)
            if missing_oshb:
                problems.append(f"OSHB is missing verses for {label}: {missing_oshb}")
        if passage["require_tow"] and not tow_matches:
            problems.append(f"TOW has no directly overlapping commentary for {label}")
        if not passage["require_tow"] and not tow_book_chunks:
            problems.append(f"TOW has no book-level commentary for {passage['book']}")

    if problems:
        print("\nVALIDATION FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("\nVALIDATION PASSED")
    print("The normalized corpus is structurally ready for retrieval development.")
    print(
        "Before the final experimental index, document a manual sample review "
        "of TOW commentary for any residual third-party or Scripture quotations."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nValidation interrupted.", file=sys.stderr)
        raise SystemExit(130)
