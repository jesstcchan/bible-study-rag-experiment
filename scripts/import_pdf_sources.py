"""Create page-level, provenance-preserving records from downloaded PDFs.

Run from the project root after adding hashes to corpus/source_manifest.csv:

    python -m scripts.import_pdf_sources

This step only extracts locally downloaded PDF text.  It does not create
embeddings or change the RAG index.  Each accepted record retains its source
file, SHA-256 hash, page number, publication metadata, source layer, and
retrieval policy for the later corpus builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "corpus" / "raw"
MANIFEST_FILE = PROJECT_ROOT / "corpus" / "source_manifest.csv"
OUTPUT_DIR = RAW_DIR / "pdf_sources"
OUTPUT_FILE = OUTPUT_DIR / "pages.jsonl"
REPORT_FILE = OUTPUT_DIR / "import_report.json"
IMPORTER_VERSION = "1.0.0"
MIN_TEXT_CHARS = 120


# One authoritative mapping from every local PDF to its source-manifest row.
# The source IDs must exactly match corpus/source_manifest.csv.
PDF_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "CONF_LUTHERAN_AUGSBURG",
        "source_name": "Augsburg Confession",
        "source_type": "confessional_document",
        "source_layer": "confessional_document",
        "claim_status": "denominational_confession",
        "tradition_scope": "Lutheran",
        "retrieval_policy": "explicit_tradition_only",
        "index_eligible": False,
        "license_status": "review_required",
        "source_file": "confessions/LCMS-The-Augsburg-Confession.pdf",
        "source_url": "https://files.lcms.org/dl/f/the-augsburg-confession",
        "version": "LCMS PDF edition",
        "publisher": "The Lutheran Church—Missouri Synod (PDF edition)",
        "edition_note": "Historical confession; verify PDF/transcription terms before indexing.",
        "book_tags": [],
    },
    {
        "source_id": "CONF_REFORMED_WESTMINSTER",
        "source_name": "Westminster Confession of Faith",
        "source_type": "confessional_document",
        "source_layer": "confessional_document",
        "claim_status": "denominational_confession",
        "tradition_scope": "Reformed/Presbyterian",
        "retrieval_policy": "explicit_tradition_only",
        "index_eligible": False,
        "license_status": "review_required",
        "source_file": "confessions/carruthers.pdf",
        "source_url": "https://www.pcahistory.org/HCLibrary/westminster/wcf/tercentenary/carruthers.pdf",
        "version": "Carruthers edition, 1946",
        "editor": "S. W. Carruthers",
        "publisher": "Publishing Office of the Presbyterian Church of England",
        "edition_note": "Original 1646 text; verify the 1946 edited PDF terms before indexing.",
        "book_tags": [],
    },
    {
        "source_id": "CONF_ANGLICAN_39_ARTICLES",
        "source_name": "Thirty-Nine Articles of Religion",
        "source_type": "confessional_document",
        "source_layer": "confessional_document",
        "claim_status": "denominational_confession",
        "tradition_scope": "Anglican",
        "retrieval_policy": "explicit_tradition_only",
        "index_eligible": False,
        "license_status": "review_required",
        "source_file": "confessions/Thirty-Nine-Articles-of-Religion.pdf",
        "source_url": "https://www.anglicancommunion.org/wp-content/uploads/2026/02/Thirty-Nine-Articles-of-Religion.pdf",
        "version": "Anglican Communion PDF edition",
        "publisher": "Anglican Communion (PDF host)",
        "edition_note": "Historical text; verify host/PDF terms before indexing.",
        "book_tags": [],
    },
    {
        "source_id": "COMM_ROMANS_SANDAY_HEADLAM",
        "source_name": "A Critical and Exegetical Commentary on the Epistle to the Romans",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "source_file": "commentaries/epistlecommentar00sanduoft.pdf",
        "source_url": "https://archive.org/details/epistlecommentar00sanduoft",
        "version": "1895 edition",
        "author": "William Sanday; Arthur C. Headlam",
        "publisher": "Charles Scribner's Sons",
        "publication_year": 1895,
        "book_tags": ["Romans"],
    },
    {
        "source_id": "COMM_KINGS_KEIL_DELITZSCH",
        "source_name": "The Books of the Kings",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "source_file": "commentaries/The_Books_of_the_Kings;_(IA_booksofkings00keil).pdf",
        "source_url": "https://archive.org/details/booksofkings00keil",
        "version": "1872 English translation",
        "author": "C. F. Keil",
        "translator": "James Martin",
        "publisher": "T. & T. Clark",
        "publication_year": 1872,
        "book_tags": ["1 Kings", "2 Kings"],
    },
    {
        "source_id": "COMM_SAMUEL_HP_SMITH",
        "source_name": "A Critical and Exegetical Commentary on the Books of Samuel",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "source_file": "commentaries/cr00iticalexegeticsmitrich.pdf",
        "source_url": "https://archive.org/details/cr00iticalexegeticsmitrich",
        "version": "Internet Archive scan; catalogued 1902",
        "author": "Henry Preserved Smith",
        "publisher": "Charles Scribner's Sons",
        "publication_year": 1902,
        "book_tags": ["1 Samuel", "2 Samuel"],
    },
    {
        "source_id": "COMM_1COR_ROBERTSON_PLUMMER",
        "source_name": "A Critical and Exegetical Commentary on the First Epistle of St. Paul to the Corinthians",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "source_file": "commentaries/1-corinthians_plummer.pdf",
        "source_url": "https://archive.org/details/in.ernet.dli.2015.88459",
        "version": "1911 edition",
        "author": "Archibald Robertson; Alfred Plummer",
        "publisher": "Charles Scribner's Sons",
        "publication_year": 1911,
        "book_tags": ["1 Corinthians"],
    },
    {
        "source_id": "COMM_2COR_PLUMMER",
        "source_name": "A Critical and Exegetical Commentary on the Second Epistle of St. Paul to the Corinthians",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "source_file": "commentaries/corinthiexegetic00plumrich.pdf",
        "source_url": "https://archive.org/details/corinthiexegetic00plumrich",
        "version": "1915 edition",
        "author": "Alfred Plummer",
        "publisher": "Charles Scribner's Sons",
        "publication_year": 1915,
        "book_tags": ["2 Corinthians"],
    },
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    """Keep paragraph boundaries while removing extraction artefacts."""
    lines = []
    for line in value.replace("\r", "\n").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()


def read_manifest_hashes() -> dict[str, str]:
    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(f"No source manifest found at {MANIFEST_FILE}")
    with MANIFEST_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hashes: dict[str, str] = {}
    for row in rows:
        source_id = str(row.get("source_id", "")).strip()
        file_hash = str(row.get("file_hash", "")).strip().casefold()
        if source_id in hashes:
            raise ValueError(f"Duplicate source_id in source manifest: {source_id}")
        hashes[source_id] = file_hash
    return hashes


def require_manifest_hash(source_id: str, actual_hash: str, hashes: dict[str, str]) -> None:
    expected_hash = hashes.get(source_id, "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError(
            f"source_manifest.csv has no valid SHA-256 for {source_id}. "
            "Paste the value printed by shasum into its file_hash cell."
        )
    if actual_hash != expected_hash:
        raise ValueError(
            f"Hash mismatch for {source_id}: manifest={expected_hash}, actual={actual_hash}. "
            "Do not import until the source file and manifest agree."
        )


def page_record(source: dict[str, Any], path: Path, file_hash: str, page_number: int,
                page_count: int, extracted_text: str, imported_at: str) -> dict[str, Any]:
    record = {
        "id": f"{source['source_id']}_page_{page_number:04d}",
        "source_id": source["source_id"],
        "source_name": source["source_name"],
        "source_type": source["source_type"],
        "source_layer": source["source_layer"],
        "claim_status": source["claim_status"],
        "tradition_scope": source["tradition_scope"],
        "retrieval_policy": source["retrieval_policy"],
        "index_eligible": bool(source["index_eligible"]),
        "license_status": source["license_status"],
        "title": f"{source['source_name']} — PDF page {page_number}",
        "text": extracted_text,
        "url": source["source_url"],
        "canonical_url": source["source_url"],
        "version": source["version"],
        "source_file": path.relative_to(RAW_DIR).as_posix(),
        "source_file_sha256": file_hash,
        "page_number": page_number,
        "pdf_page_count": page_count,
        "original_text_sha256": sha256_text(extracted_text),
        "book_tags": list(source.get("book_tags", [])),
        "imported_at": imported_at,
        "importer_version": IMPORTER_VERSION,
    }
    for key in (
        "author", "editor", "translator", "publisher", "publication_year", "edition_note",
    ):
        if source.get(key) not in (None, ""):
            record[key] = source[key]
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and validate PDFs without writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hashes = read_manifest_hashes()
    imported_at = utc_now()
    records: list[dict[str, Any]] = []
    report_sources: list[dict[str, Any]] = []

    for source in PDF_SOURCES:
        path = RAW_DIR / source["source_file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing PDF for {source['source_id']}: {path}"
            )
        file_hash = sha256_file(path)
        require_manifest_hash(source["source_id"], file_hash, hashes)
        try:
            reader = PdfReader(str(path))
        except Exception as error:
            raise RuntimeError(f"Could not open {path}: {error}") from error

        source_records: list[dict[str, Any]] = []
        skipped_pages: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted_text = normalize_text(page.extract_text() or "")
            except Exception as error:
                raise RuntimeError(
                    f"Could not extract page {page_number} from {path}: {error}"
                ) from error
            if len(extracted_text) < MIN_TEXT_CHARS:
                skipped_pages.append(page_number)
                continue
            source_records.append(
                page_record(
                    source,
                    path,
                    file_hash,
                    page_number,
                    len(reader.pages),
                    extracted_text,
                    imported_at,
                )
            )

        if not source_records:
            raise ValueError(
                f"No usable text was extracted from {path}. The file may be image-only "
                "and require OCR before it can be used."
            )
        records.extend(source_records)
        report_sources.append(
            {
                "source_id": source["source_id"],
                "source_file": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_file_sha256": file_hash,
                "pdf_pages": len(reader.pages),
                "accepted_pages": len(source_records),
                "skipped_short_pages": skipped_pages,
                "index_eligible": bool(source["index_eligible"]),
                "license_status": source["license_status"],
            }
        )
        print(
            f"  {source['source_id']}: {len(source_records):,}/{len(reader.pages):,} "
            "PDF pages accepted"
        )

    report = {
        "importer_version": IMPORTER_VERSION,
        "imported_at": imported_at,
        "total_records": len(records),
        "sources": report_sources,
        "notes": [
            "This output is a raw, page-level extraction. It has not been embedded.",
            "Confessional sources remain index_eligible=false until their PDF/transcription terms are reviewed.",
        ],
    }
    if args.dry_run:
        print(f"DRY RUN COMPLETE: {len(records):,} page records would be written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(OUTPUT_FILE)
    report["output_file"] = OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix()
    report["sha256"] = sha256_file(OUTPUT_FILE)
    report_temporary = REPORT_FILE.with_suffix(REPORT_FILE.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_temporary.replace(REPORT_FILE)

    print("PDF IMPORT COMPLETE")
    print(f"Records: {len(records):,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print(f"SHA-256: {report['sha256']}")
    print("No embeddings or index files were changed.")


if __name__ == "__main__":
    main()
