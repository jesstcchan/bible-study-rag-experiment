"""Build a normalized JSONL corpus for the Bible-study RAG system.

Run this file from the oTree project directory with:

    python -m scripts.build_corpus

Expected project layout:

    corpus/raw/bible_odyssey/pages.jsonl
    corpus/raw/openbible/cross_references.txt
    corpus/raw/oshb/
    corpus/raw/pdf_sources/pages.jsonl
    corpus/raw/confessions/1689/pages.jsonl
    corpus/raw/stepbible/
    corpus/raw/theology_of_work/pages.jsonl
    corpus/raw/unfoldingword_notes/
    corpus/raw/unfoldingword_words/
    corpus/raw/world_english_bible/
    corpus/processed/

The script uses only Python's standard library. It deliberately excludes
licence/readme/repository files and macOS ``__MACOSX`` metadata from the RAG
content. Theology of Work is imported from the controlled, licensed Bible
Commentary crawl. Standalone quotation blocks were removed by the scraper;
the builder also removes remaining inline quoted spans so that the RAG corpus
uses WEB for Scripture text rather than separately copyrighted translations.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "corpus" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
REPORT_FILE = OUTPUT_DIR / "build_report.json"
PDF_PAGES_FILE = RAW_DIR / "pdf_sources" / "pages.jsonl"
CONF_1689_PAGES_FILE = RAW_DIR / "confessions" / "1689" / "pages.jsonl"

# Character based splitting is used here so no tokenizer dependency is needed.
# This remains comfortably below the embedding model's input limit.
MAX_CHARS = 3_500

# Only these five historic commentaries are allowed through the default
# Protestant study-book pipeline. Confessional documents remain present in
# raw storage but are intentionally held out until their PDF/transcription
# terms have been reviewed and an explicit tradition-selection flow exists.
PDF_COMMENTARY_SOURCE_IDS = {
    "COMM_ROMANS_SANDAY_HEADLAM",
    "COMM_KINGS_KEIL_DELITZSCH",
    "COMM_SAMUEL_HP_SMITH",
    "COMM_1COR_ROBERTSON_PLUMMER",
    "COMM_2COR_PLUMMER",
}

PDF_PAGE_REQUIRED_FIELDS = {
    "id", "source_id", "source_name", "source_type", "source_layer",
    "claim_status", "tradition_scope", "retrieval_policy", "index_eligible",
    "license_status", "title", "text", "url", "canonical_url", "version",
    "source_file", "source_file_sha256", "page_number", "pdf_page_count",
    "original_text_sha256", "book_tags",
}

# These are provenance/download pages, not historical commentary. They stay in
# the raw JSONL for auditability but never become searchable corpus chunks.
PDF_NONCONTENT_MARKERS = (
    "this document was supplied for free educational purposes",
    "buymecoffee.com/theology",
    "patreon.com/theologyontheweb",
    "paypal.me/",
)


BOOK_NAMES = {
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus",
    "NUM": "Numbers", "DEU": "Deuteronomy", "JOS": "Joshua",
    "JDG": "Judges", "RUT": "Ruth", "1SA": "1 Samuel",
    "2SA": "2 Samuel", "1KI": "1 Kings", "2KI": "2 Kings",
    "1CH": "1 Chronicles", "2CH": "2 Chronicles", "EZR": "Ezra",
    "NEH": "Nehemiah", "EST": "Esther", "JOB": "Job",
    "PSA": "Psalms", "PRO": "Proverbs", "ECC": "Ecclesiastes",
    "SNG": "Song of Songs", "ISA": "Isaiah", "JER": "Jeremiah",
    "LAM": "Lamentations", "EZK": "Ezekiel", "DAN": "Daniel",
    "HOS": "Hosea", "JOL": "Joel", "AMO": "Amos",
    "OBA": "Obadiah", "JON": "Jonah", "MIC": "Micah",
    "NAM": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah",
    "HAG": "Haggai", "ZEC": "Zechariah", "MAL": "Malachi",
    "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke",
    "JHN": "John", "ACT": "Acts", "ROM": "Romans",
    "1CO": "1 Corinthians", "2CO": "2 Corinthians",
    "GAL": "Galatians", "EPH": "Ephesians", "PHP": "Philippians",
    "COL": "Colossians", "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians", "1TI": "1 Timothy",
    "2TI": "2 Timothy", "TIT": "Titus", "PHM": "Philemon",
    "HEB": "Hebrews", "JAS": "James", "1PE": "1 Peter",
    "2PE": "2 Peter", "1JN": "1 John", "2JN": "2 John",
    "3JN": "3 John", "JUD": "Jude", "REV": "Revelation",
}

OSIS_TO_USFM = {
    "Gen": "GEN", "Exod": "EXO", "Lev": "LEV", "Num": "NUM",
    "Deut": "DEU", "Josh": "JOS", "Judg": "JDG", "Ruth": "RUT",
    "1Sam": "1SA", "2Sam": "2SA", "1Kgs": "1KI", "2Kgs": "2KI",
    "1Chr": "1CH", "2Chr": "2CH", "Ezra": "EZR", "Neh": "NEH",
    "Esth": "EST", "Job": "JOB", "Ps": "PSA", "Prov": "PRO",
    "Eccl": "ECC", "Song": "SNG", "Isa": "ISA", "Jer": "JER",
    "Lam": "LAM", "Ezek": "EZK", "Dan": "DAN", "Hos": "HOS",
    "Joel": "JOL", "Amos": "AMO", "Obad": "OBA", "Jonah": "JON",
    "Mic": "MIC", "Nah": "NAM", "Hab": "HAB", "Zeph": "ZEP",
    "Hag": "HAG", "Zech": "ZEC", "Mal": "MAL",
    "Matt": "MAT", "Mark": "MRK", "Luke": "LUK", "John": "JHN",
    "Acts": "ACT", "Rom": "ROM", "1Cor": "1CO", "2Cor": "2CO",
    "Gal": "GAL", "Eph": "EPH", "Phil": "PHP", "Col": "COL",
    "1Thess": "1TH", "2Thess": "2TH", "1Tim": "1TI",
    "2Tim": "2TI", "Titus": "TIT", "Phlm": "PHM", "Phm": "PHM",
    "Heb": "HEB", "Jas": "JAS", "1Pet": "1PE", "2Pet": "2PE",
    "1John": "1JN", "2John": "2JN", "3John": "3JN",
    "Jude": "JUD", "Rev": "REV",
}


SOURCE_META = {
    "WEB": {
        "source_name": "World English Bible, Protestant edition",
        "source_type": "bible_text",
        "source_layer": "biblical_text",
        "claim_status": "biblical_text",
        "tradition_scope": "Protestant canon",
        "retrieval_policy": "always_available",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "engwebp",
        "license": "Public domain",
        "url": "https://ebible.org/find/details.php?id=engwebp",
    },
    "STEP": {
        "source_name": "STEPBible TBESH/TBESG brief lexicons",
        "source_type": "lexical_data",
        "source_layer": "hebrew_and_greek_lexical_data",
        "claim_status": "lexical_reference",
        "tradition_scope": "Hebrew and Greek biblical languages",
        "retrieval_policy": "supporting_reference",
        "index_eligible": True,
        "license_status": "cc_by_4_0",
        "version": "downloaded repository snapshot",
        "license": "CC BY 4.0",
        "url": "https://github.com/STEPBible/STEPBible-Data/tree/master/Lexicons",
    },
    "OSHB": {
        "source_name": "Open Scriptures Hebrew Bible",
        "source_type": "hebrew_morphology",
        "source_layer": "hebrew_morphology_and_lemma_data",
        "claim_status": "linguistic_data",
        "tradition_scope": "Hebrew Bible / Protestant Old Testament",
        "retrieval_policy": "supporting_reference",
        "index_eligible": True,
        "license_status": "public_domain_and_cc_by_4_0",
        "version": "v2.2",
        "license": "WLC text public domain; lemma and morphology CC BY 4.0",
        "url": "https://github.com/openscriptures/morphhb/releases/tag/v.2.2",
    },
    "UTN": {
        "source_name": "unfoldingWord Translation Notes",
        "source_type": "translation_note",
        "source_layer": "translation_and_contextual_notes",
        "claim_status": "translation_note",
        "tradition_scope": "Protestant translation aid",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "cc_by_sa_4_0",
        "version": "v90",
        "license": "CC BY-SA 4.0",
        "url": "https://git.door43.org/unfoldingWord/en_tn/releases/tag/v90",
    },
    "UTW": {
        "source_name": "unfoldingWord Translation Words",
        "source_type": "biblical_term",
        "source_layer": "biblical_terms_and_concepts",
        "claim_status": "thematic_reference",
        "tradition_scope": "Protestant translation aid",
        "retrieval_policy": "supporting_reference",
        "index_eligible": True,
        "license_status": "cc_by_sa_4_0",
        "version": "v90",
        "license": "CC BY-SA 4.0",
        "url": "https://git.door43.org/unfoldingWord/en_tw/releases/tag/v90",
    },
    "TOW": {
        "source_name": "Theology of Work Bible Commentary",
        "source_type": "commentary",
        "source_layer": "contemporary_application_commentary",
        "claim_status": "contemporary_commentary",
        "tradition_scope": "Protestant workplace theology",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "cc_by_nc_4_0",
        "version": "Website snapshot 2026-09-22 (scraper v2.1.0)",
        "license": "CC BY-NC 4.0",
        "url": "https://www.theologyofwork.org/resources/the-theology-of-work-bible-commentary",
    },

    "OPENBIBLE_XREF": {
        "source_name": "OpenBible.info Bible Cross References",
        "source_type": "cross_reference",
        "source_layer": "cross_reference_data",
        "claim_status": "cross_reference",
        "tradition_scope": "Protestant canon",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "cc_by_4_0",
        "version": "Dataset snapshot 2026-09-07",
        "license": "CC BY 4.0",
        "url": "https://www.openbible.info/labs/cross-references/",
    },
    "BIBLE_ODYSSEY": {
        "source_name": "Bible Odyssey",
        "source_type": "scholarly_article",
        "source_layer": "scholarly_articles_and_dictionary_entries",
        "claim_status": "scholarly_article",
        "tradition_scope": "academic_multitradition",
        "retrieval_policy": "held_pending_written_permission",
        "index_eligible": False,
        "license_status": "written_permission_required",
        "version": "Website snapshot 2026-09-09",
        # Replace this wording with the exact scope and date of the written
        # permission before enabling this source in the final study corpus.
        "license": "Written permission required; see source_manifest.csv",
        "url": "https://www.bibleodyssey.org/",
    },
    "COMM_ROMANS_SANDAY_HEADLAM": {
        "source_name": "A Critical and Exegetical Commentary on the Epistle to the Romans",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "1895 edition",
        "license": "Public-domain historical edition",
        "url": "https://archive.org/details/epistlecommentar00sanduoft",
    },
    "COMM_KINGS_KEIL_DELITZSCH": {
        "source_name": "The Books of the Kings",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "1872 English translation",
        "license": "Public-domain historical edition",
        "url": "https://archive.org/details/booksofkings00keil",
    },
    "COMM_SAMUEL_HP_SMITH": {
        "source_name": "A Critical and Exegetical Commentary on the Books of Samuel",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "Internet Archive scan; catalogued 1902",
        "license": "Public-domain historical edition",
        "url": "https://archive.org/details/cr00iticalexegeticsmitrich",
    },
    "COMM_1COR_ROBERTSON_PLUMMER": {
        "source_name": "A Critical and Exegetical Commentary on the First Epistle of St. Paul to the Corinthians",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "1911 edition",
        "license": "Public-domain historical edition",
        "url": "https://archive.org/details/in.ernet.dli.2015.88459",
    },
    "COMM_2COR_PLUMMER": {
        "source_name": "A Critical and Exegetical Commentary on the Second Epistle of St. Paul to the Corinthians",
        "source_type": "academic_commentary",
        "source_layer": "academic_commentary",
        "claim_status": "scholarly_commentary",
        "tradition_scope": "Historical academic commentary",
        "retrieval_policy": "study_book_default",
        "index_eligible": True,
        "license_status": "public_domain",
        "version": "1915 edition",
        "license": "Public-domain historical edition",
        "url": "https://archive.org/details/corinthiexegetic00plumrich",
    },
}


TOW_BOOK_TAG_NAMES = {
    "genesis-1-11": "Genesis",
    "genesis-12-50": "Genesis",
    "song-of-songs": "Song of Songs",
    **{
        name.lower().replace(" ", "-"): name
        for name in BOOK_NAMES.values()
        if name not in {"Genesis", "Song of Songs"}
    },
}

TOW_REFERENCE_BOOKS = sorted(set(BOOK_NAMES.values()), key=len, reverse=True)
TOW_REFERENCE_RE = re.compile(
    r"\b(" + "|".join(re.escape(book) for book in TOW_REFERENCE_BOOKS) + r")"
    r"\s+(\d+)"
    r"(?::(\d+)(?:\s*[-\u2013\u2014]\s*(?:(\d+):)?(\d+))?)?",
    flags=re.I,
)

TOW_INLINE_QUOTE_PATTERNS = (
    re.compile(r'[\u201c\u201d"][^\u201c\u201d"]+[\u201c\u201d"]', flags=re.S),
    re.compile(r"\u2018[^\u2019]+\u2019", flags=re.S),
)


def usable_file(path: Path) -> bool:
    """Return False for repository, licence and macOS metadata files."""
    parts = set(path.parts)
    return (
        "__MACOSX" not in parts
        and ".git" not in parts
        and ".gitea" not in parts
        and ".github" not in parts
        and not path.name.startswith("._")
    )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def safe_id(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return result or "NA"


def plain_html(value: str) -> str:
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value)


def normalize_text(value: str) -> str:
    value = value.replace("\ufeff", "").replace("\r\n", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_pdf_page_text(value: str) -> str:
    """Flatten physical PDF line breaks without modifying the raw extraction.

    Several supplied PDFs are visually laid out as one word per line.  That is
    appropriate to preserve in ``corpus/raw/pdf_sources/pages.jsonl`` but not
    for a retrieval chunk.  The raw text and its page hash remain unchanged;
    this function creates only the normalized representation used downstream.
    """
    value = normalize_text(value)
    value = re.sub(r"\s*\n\s*", " ", value)
    return re.sub(r"\s{2,}", " ", value).strip()


def read_jsonl_records(path: Path, label: str) -> list[dict]:
    """Load JSONL records with a useful error that names the source line."""
    if not path.is_file():
        raise FileNotFoundError(f"No {label} JSONL file found at {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid {label} JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{label} line {line_number} is not a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{label} JSONL contains no records: {path}")
    return records


def pdf_page_exclusion_reason(page: dict) -> str | None:
    """Return a reason when a raw PDF page is not commentary prose.

    This is deliberately conservative: uncertain material remains in raw
    storage and will be reviewed later; only clear contents, index, editorial,
    and third-party download pages are removed from the normalized corpus.
    """
    text = normalize_pdf_page_text(str(page.get("text", "")))
    lower = text.casefold()
    leading = lower[:450]

    if any(marker in lower for marker in PDF_NONCONTENT_MARKERS):
        return "third_party_download_notice"
    if re.search(r"\b(?:table of )?contents\b", leading):
        return "contents"
    if re.search(r"\b(?:general |scriptural |subject )?index\b", leading):
        return "index"

    page_number = page.get("page_number")
    if isinstance(page_number, int) and page_number <= 15:
        if "under the editorship" in lower or "under the present editorship" in lower:
            return "editorial_front_matter"
    return None


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = f"{current} {sentence}".strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sentence[:max_chars]
                    remainder = sentence[max_chars:]
                    while len(remainder) > max_chars:
                        chunks.append(remainder[:max_chars])
                        remainder = remainder[max_chars:]
                    if remainder:
                        current = remainder
    if current:
        chunks.append(current)
    return chunks


def make_chunks(
    source_id: str,
    base_id: str,
    title: str,
    text: str,
    *,
    book: str | None = None,
    chapter: int | None = None,
    verse_start: int | None = None,
    verse_end: int | None = None,
    extra: dict | None = None,
) -> list[dict]:
    meta = SOURCE_META[source_id]
    pieces = split_long_text(text)
    result = []
    for number, piece in enumerate(pieces, start=1):
        suffix = f"_{number:02d}" if len(pieces) > 1 else ""
        record = {
            "chunk_id": f"{source_id}_{safe_id(base_id)}{suffix}",
            "source_id": source_id,
            "source_name": meta["source_name"],
            "source_type": meta["source_type"],
            "book": book,
            "chapter": chapter,
            "verse_start": verse_start,
            "verse_end": verse_end,
            "title": title,
            "text": piece,
            "version": meta["version"],
            "license": meta["license"],
            "url": meta["url"],
        }
        # These fields make the evidence layer explicit at retrieval time.
        # In particular, a commentary chunk must never be presented as though
        # it were the biblical text itself.
        for field in (
            "source_layer", "claim_status", "tradition_scope",
            "retrieval_policy", "index_eligible", "license_status",
        ):
            if field in meta:
                record[field] = meta[field]
        if extra:
            record.update(extra)
        result.append(record)
    return result


def remove_tow_inline_quoted_spans(text: str) -> tuple[str, int]:
    """Remove typographically marked quotations from a TOW text block.

    The controlled scraper already removes blockquotes. This second,
    conservative pass removes remaining curly- or straight-quoted spans so
    separately copyrighted Scripture and third-party quotations are not sent
    to the embedding service. The surrounding TOW-authored commentary and
    references remain available.
    """
    removed = 0
    for pattern in TOW_INLINE_QUOTE_PATTERNS:
        text, count = pattern.subn(" ", text)
        removed += count
    # Discard unmatched double-quote glyphs left by inconsistent source HTML.
    text = text.replace("\u201c", " ").replace("\u201d", " ").replace('"', " ")
    return normalize_text(text), removed


def tow_reference_metadata(page: dict) -> tuple[str | None, int | None, int | None, int | None]:
    """Infer the first curator-supplied or explicit Bible reference."""
    raw_passage_tags = page.get("passage_tags") or []
    if isinstance(raw_passage_tags, str):
        raw_passage_tags = [raw_passage_tags]
    candidates = [
        normalize_text(str(page.get("passage", ""))),
        *(normalize_text(str(tag)) for tag in raw_passage_tags),
        normalize_text(str(page.get("title", ""))),
    ]
    for candidate in candidates:
        match = TOW_REFERENCE_RE.search(candidate)
        if not match:
            continue
        matched_book = match.group(1)
        book = next(
            (name for name in TOW_REFERENCE_BOOKS if name.lower() == matched_book.lower()),
            matched_book,
        )
        chapter = int(match.group(2))
        verse_start = int(match.group(3)) if match.group(3) else None
        verse_end = verse_start
        if verse_start is not None and match.group(5):
            end_chapter = int(match.group(4)) if match.group(4) else chapter
            if end_chapter == chapter:
                verse_end = int(match.group(5))
        return book, chapter, verse_start, verse_end

    raw_book_tags = page.get("book_tags") or []
    if isinstance(raw_book_tags, str):
        raw_book_tags = [raw_book_tags]
    for tag in raw_book_tags:
        tag_key = re.sub(
            r"[^a-z0-9]+",
            "-",
            str(tag).casefold(),
        ).strip("-")
        book = TOW_BOOK_TAG_NAMES.get(tag_key)
        if book:
            return book, None, None, None
    return None, None, None, None


def tow_body_pieces(page: dict, max_chars: int) -> tuple[list[str], int, int]:
    """Group sanitized page blocks while retaining their section headings."""
    items: list[str] = []
    active_heading = ""
    removed_quoted_spans = 0
    removed_table_source_cells = 0

    blocks = page.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError(f"TOW page {page.get('id', 'unknown')} has no block list")

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = normalize_text(str(block.get("type", "paragraph"))).lower()
        block_text, removed = remove_tow_inline_quoted_spans(str(block.get("text", "")))
        removed_quoted_spans += removed
        if not block_text:
            continue
        if block_type == "heading":
            active_heading = block_text
            continue
        if block_type == "table_row":
            cells = [normalize_text(cell) for cell in block_text.split(" | ") if cell.strip()]
            if len(cells) >= 2:
                table_parts: list[str] = []
                reference_match = TOW_REFERENCE_RE.search(cells[0])
                if reference_match:
                    table_parts.append(f"Table reference: {reference_match.group(0)}")
                elif len(cells[0]) <= 100:
                    table_parts.append(f"Table topic: {cells[0]}")
                table_parts.append(f"Table commentary: {cells[-1]}")
                block_text = "\n".join(table_parts)
                removed_table_source_cells += len(cells) - 1
            else:
                block_text = f"Table commentary: {block_text}"
        if active_heading:
            block_text = f"Section: {active_heading}\n{block_text}"
        items.extend(split_long_text(block_text, max_chars=max_chars))

    if active_heading and not items:
        items.append(f"Section: {active_heading}")

    pieces: list[str] = []
    current = ""
    for item in items:
        candidate = f"{current}\n\n{item}".strip() if current else item
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = item
    if current:
        pieces.append(current)
    return pieces, removed_quoted_spans, removed_table_source_cells


def parse_tow() -> list[dict]:
    """Parse the controlled Theology of Work Bible Commentary crawl."""
    path = RAW_DIR / "theology_of_work" / "pages.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No Theology of Work pages.jsonl found at {path}")

    pages: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid TOW JSON on line {line_number}: {error}") from error
            pages.append(page)

    page_ids = Counter(str(page.get("id", "")) for page in pages)
    page_urls = Counter(str(page.get("canonical_url", "")) for page in pages)
    duplicate_ids = [value for value, count in page_ids.items() if value and count > 1]
    duplicate_urls = [value for value, count in page_urls.items() if value and count > 1]
    if duplicate_ids or duplicate_urls:
        raise ValueError(
            "Duplicate TOW page identity detected: "
            f"ids={duplicate_ids[:5]}, urls={duplicate_urls[:5]}"
        )

    chunks: list[dict] = []
    for page_number, page in enumerate(pages, start=1):
        required = {"id", "title", "canonical_url", "license", "blocks", "text_sha256"}
        missing = sorted(required - page.keys())
        if missing:
            raise ValueError(f"TOW page {page_number} is missing fields: {missing}")
        if page.get("content_type") != "Bible Commentary":
            raise ValueError(
                f"TOW page {page.get('id')} is not labelled Bible Commentary"
            )
        if page.get("license") != "CC BY-NC 4.0":
            raise ValueError(f"Unsupported TOW licence on page {page.get('id')}")

        page_id = str(page["id"])
        title = normalize_text(str(page["title"]))
        prefix_title, title_quotes_removed = remove_tow_inline_quoted_spans(title)
        prefix_title = prefix_title or "Bible commentary"
        passage = normalize_text(str(page.get("passage", "")))
        book, chapter, verse_start, verse_end = tow_reference_metadata(page)
        prefix_parts = [f"Commentary page: {prefix_title}"]
        if passage:
            prefix_parts.append(f"Biblical passage: {passage}")
        if book:
            prefix_parts.append(f"Book: {book}")
        prefix = "\n".join(prefix_parts)
        body_limit = MAX_CHARS - len(prefix) - 2
        if body_limit < 500:
            raise ValueError(f"TOW metadata prefix is unexpectedly long on {page_id}")
        pieces, removed_quotes, removed_table_cells = tow_body_pieces(page, body_limit)
        removed_quotes += title_quotes_removed
        if not pieces:
            raise ValueError(f"TOW page {page_id} has no usable commentary after cleaning")

        page_chunk_count = len(pieces)
        for piece_number, piece in enumerate(pieces, start=1):
            text = f"{prefix}\n\n{piece}"
            extra = {
                "url": str(page["canonical_url"]),
                "canonical_url": str(page["canonical_url"]),
                "source_page_id": page_id,
                "page_chunk_number": piece_number,
                "page_chunk_count": page_chunk_count,
                "passage": passage or None,
                "book_tags": page.get("book_tags", []),
                "hierarchy_role": page.get("hierarchy_role"),
                "content_type": page.get("content_type"),
                "producer": page.get("producer"),
                "contributors": page.get("contributors"),
                "license_url": page.get("license_url"),
                "copyright_notice": page.get("copyright_notice"),
                "accessed_at": page.get("accessed_at"),
                "scraper_version": page.get("scraper_version"),
                "scope": page.get("scope"),
                "study_role": page.get("study_role"),
                "original_html_sha256": page.get("html_sha256"),
                "original_text_sha256": page.get("text_sha256"),
                "manual_review_required": bool(page.get("manual_review_required")),
                "scripture_translations_noted": page.get("scripture_translations_noted", []),
                "inline_quoted_spans_removed_from_page": removed_quotes,
                "table_source_cells_removed_from_page": removed_table_cells,
            }
            chunks.extend(make_chunks(
                "TOW",
                f"{page_id}_{piece_number:03d}",
                title,
                text,
                book=book,
                chapter=chapter,
                verse_start=verse_start,
                verse_end=verse_end,
                extra=extra,
            ))
    return chunks


def parse_pdf_sources() -> list[dict]:
    """Turn reviewed historical-commentary PDF pages into retrieval chunks.

    The page-level JSONL stays the immutable, provenance-preserving source.
    This parser creates a second, cleaned representation suitable for RAG:
    it flattens physical PDF line breaks, removes clear non-content pages, and
    labels every result as scholarly interpretation rather than Bible text.
    """
    pages = read_jsonl_records(PDF_PAGES_FILE, "PDF source")
    chunks: list[dict] = []
    seen_page_ids: set[str] = set()
    seen_commentary_sources: set[str] = set()
    excluded = Counter()
    held_confessional_pages = Counter()

    for line_number, page in enumerate(pages, start=1):
        missing = sorted(PDF_PAGE_REQUIRED_FIELDS - page.keys())
        if missing:
            raise ValueError(f"PDF source line {line_number} is missing fields: {missing}")

        source_id = normalize_text(str(page["source_id"]))
        page_id = normalize_text(str(page["id"]))
        if not source_id or not page_id:
            raise ValueError(f"PDF source line {line_number} has no source or page ID")
        if page_id in seen_page_ids:
            raise ValueError(f"Duplicate PDF page ID found: {page_id}")
        seen_page_ids.add(page_id)

        # The three PDF confessions are deliberately stored but not processed.
        # Their index_eligible flag is enforced here, rather than relying only
        # on a future index command to exclude them.
        if page.get("index_eligible") is not True:
            if page.get("source_type") != "confessional_document":
                raise ValueError(
                    f"PDF source {page_id} is ineligible but is not a confessional document"
                )
            if page.get("license_status") != "review_required":
                raise ValueError(
                    f"PDF confession {page_id} must remain review_required"
                )
            held_confessional_pages[source_id] += 1
            continue

        if source_id not in PDF_COMMENTARY_SOURCE_IDS:
            raise ValueError(
                f"PDF source {page_id} is index-eligible but is not an approved "
                f"historical commentary: {source_id}"
            )
        if page.get("source_type") != "academic_commentary":
            raise ValueError(f"PDF source {page_id} has the wrong source_type")
        if page.get("source_layer") != "academic_commentary":
            raise ValueError(f"PDF source {page_id} has the wrong source_layer")
        if page.get("claim_status") != "scholarly_commentary":
            raise ValueError(f"PDF source {page_id} has the wrong claim_status")
        if page.get("license_status") != "public_domain":
            raise ValueError(f"PDF commentary {page_id} is not marked public_domain")

        meta = SOURCE_META[source_id]
        for field in (
            "source_name", "source_type", "source_layer", "claim_status",
            "tradition_scope", "retrieval_policy", "version",
        ):
            if page.get(field) != meta[field]:
                raise ValueError(
                    f"PDF source {page_id} disagrees with the approved metadata "
                    f"for {source_id}: {field}"
                )
        source_file_hash = normalize_text(str(page["source_file_sha256"]))
        if not re.fullmatch(r"[0-9a-f]{64}", source_file_hash):
            raise ValueError(f"PDF source {page_id} has no valid source-file SHA-256")
        original_text_hash = normalize_text(str(page["original_text_sha256"]))
        if not re.fullmatch(r"[0-9a-f]{64}", original_text_hash):
            raise ValueError(f"PDF source {page_id} has no valid text SHA-256")

        reason = pdf_page_exclusion_reason(page)
        if reason:
            excluded[reason] += 1
            continue

        text = normalize_pdf_page_text(str(page["text"]))
        if not text:
            excluded["empty_after_normalization"] += 1
            continue

        page_number = page.get("page_number")
        page_count = page.get("pdf_page_count")
        if (
            not isinstance(page_number, int) or page_number <= 0
            or not isinstance(page_count, int) or page_count < page_number
        ):
            raise ValueError(f"PDF source {page_id} has invalid page location metadata")
        book_tags = page.get("book_tags")
        if not isinstance(book_tags, list) or not all(isinstance(tag, str) for tag in book_tags):
            raise ValueError(f"PDF source {page_id} has invalid book_tags")
        book_tags = [normalize_text(tag) for tag in book_tags if normalize_text(tag)]
        if not book_tags:
            raise ValueError(f"PDF commentary {page_id} has no study-book tags")

        source_name = normalize_text(str(page["source_name"]))
        author = normalize_text(str(page.get("author", "")))
        prefix = [
            "Interpretive layer: scholarly academic commentary.",
            "This is an interpretation; do not treat it as biblical text.",
            f"Source: {source_name}.",
            f"Biblical books: {', '.join(book_tags)}.",
            f"PDF page: {page_number} of {page_count}.",
        ]
        if author:
            prefix.insert(3, f"Author: {author}.")
        page_text = "\n".join(prefix) + "\n\n" + text
        title = f"{source_name} — PDF page {page_number}"
        page_chunks = make_chunks(
            source_id,
            page_id,
            title,
            page_text,
            # A multi-book commentary page cannot be assigned to one book
            # solely from its PDF page number. Its full book_tags are retained
            # for the study-book subset builder.
            book=book_tags[0] if len(book_tags) == 1 else None,
            extra={
                "url": str(page["canonical_url"]),
                "canonical_url": str(page["canonical_url"]),
                "source_page_id": page_id,
                "source_file": str(page["source_file"]),
                "source_file_sha256": source_file_hash,
                "page_number": page_number,
                "pdf_page_count": page_count,
                "original_text_sha256": original_text_hash,
                "book_tags": book_tags,
                "author": page.get("author"),
                "editor": page.get("editor"),
                "translator": page.get("translator"),
                "publisher": page.get("publisher"),
                "publication_year": page.get("publication_year"),
                "edition_note": page.get("edition_note"),
                "imported_at": page.get("imported_at"),
                "importer_version": page.get("importer_version"),
                "raw_page_filter": "accepted_commentary_prose",
            },
        )
        for chunk_number, chunk in enumerate(page_chunks, start=1):
            chunk["page_chunk_number"] = chunk_number
            chunk["page_chunk_count"] = len(page_chunks)
        chunks.extend(page_chunks)
        seen_commentary_sources.add(source_id)

    missing_sources = sorted(PDF_COMMENTARY_SOURCE_IDS - seen_commentary_sources)
    if missing_sources:
        raise ValueError(
            "The PDF import is missing approved commentary sources: "
            f"{', '.join(missing_sources)}"
        )
    if held_confessional_pages:
        held_summary = ", ".join(
            f"{source_id}={count}" for source_id, count in sorted(held_confessional_pages.items())
        )
        print(f"  PDF confessions held from indexing: {held_summary}")
    if excluded:
        excluded_summary = ", ".join(
            f"{reason}={count}" for reason, count in sorted(excluded.items())
        )
        print(f"  PDF non-content pages excluded from normalized corpus: {excluded_summary}")
    return chunks


def inspect_1689_held_source() -> list[dict]:
    """Verify that the scraped 1689 source is still held outside the index."""
    if not CONF_1689_PAGES_FILE.is_file():
        print(
            "  CONF_BAPTIST_1689: no raw pages file found; source remains out of the index."
        )
        return []

    pages = read_jsonl_records(CONF_1689_PAGES_FILE, "1689 confession")
    for line_number, page in enumerate(pages, start=1):
        if page.get("source_id") != "CONF_BAPTIST_1689":
            raise ValueError(f"1689 confession line {line_number} has the wrong source_id")
        if page.get("source_type") != "confessional_document":
            raise ValueError(f"1689 confession line {line_number} has the wrong source_type")
        if page.get("claim_status") != "denominational_confession":
            raise ValueError(f"1689 confession line {line_number} has the wrong claim_status")
        if page.get("license_status") != "review_required":
            raise ValueError(
                f"1689 confession line {line_number} must remain review_required"
            )
    print(
        f"  CONF_BAPTIST_1689: {len(pages):,} raw pages held pending website-transcription review."
    )
    return []


def strip_usfm(text: str) -> str:
    # Remove footnotes, cross-references and study notes before general markers.
    text = re.sub(r"\\(?:f|x|fe)\s.*?\\(?:f|x|fe)\*", " ", text)
    text = re.sub(r"\\zaln-s\s.*?\\\*", " ", text)
    text = re.sub(r"\\zaln-e\\\*", " ", text)
    text = re.sub(r"\\w\s+([^|\\]+)(?:\|[^\\]*)?\\w\*", r"\1", text)
    text = re.sub(r"\\[A-Za-z0-9+\-]+\*?", " ", text)
    return normalize_text(text)


def parse_web() -> list[dict]:
    root = RAW_DIR / "world_english_bible"
    files = sorted(p for p in root.rglob("*.usfm") if usable_file(p))
    if not files:
        raise FileNotFoundError(f"No WEB .usfm files found under {root}")

    chunks: list[dict] = []
    for path in files:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        id_match = re.search(r"(?m)^\\id\s+([A-Z0-9]{3})", content)
        file_match = re.search(r"-([1-3]?[A-Z]{2,3})engwebp", path.name)
        code = (id_match or file_match).group(1) if (id_match or file_match) else path.stem
        book = BOOK_NAMES.get(code, code)
        chapter: int | None = None
        active_verse: tuple[int, int, str] | None = None

        def flush() -> None:
            nonlocal active_verse
            if not active_verse or chapter is None:
                return
            start, end, verse_text = active_verse
            verse_text = strip_usfm(verse_text)
            if verse_text:
                chunks.extend(make_chunks(
                    "WEB", f"{code}_{chapter:03d}_{start:03d}_{end:03d}",
                    f"{book} {chapter}:{start}" + (f"-{end}" if end != start else ""),
                    verse_text, book=book, chapter=chapter,
                    verse_start=start, verse_end=end,
                ))
            active_verse = None

        for raw_line in content.splitlines():
            line = raw_line.strip()
            chapter_match = re.match(r"\\c\s+(\d+)", line)
            if chapter_match:
                flush()
                chapter = int(chapter_match.group(1))
                continue
            verse_match = re.match(r"\\v\s+(\d+)(?:-(\d+))?\s*(.*)", line)
            if verse_match:
                flush()
                start = int(verse_match.group(1))
                end = int(verse_match.group(2) or start)
                active_verse = (start, end, verse_match.group(3))
                continue
            if active_verse and line and not re.match(r"\\(?:id|ide|h|toc\d|mt\d?)\b", line):
                start, end, previous = active_verse
                active_verse = (start, end, f"{previous} {line}")
        flush()
    return chunks


OPENBIBLE_REFERENCE_RE = re.compile(
    r"^(?P<book>[1-3]?[A-Za-z]+)\.(?P<chapter>\d+)\.(?P<verse>\d+)$"
)


def parse_openbible_reference(value: str) -> tuple[str, int, int]:
    """Convert one OpenBible OSIS-style verse to USFM book/chapter/verse."""
    match = OPENBIBLE_REFERENCE_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported OpenBible reference: {value!r}")
    osis_book = match.group("book")
    try:
        book_code = OSIS_TO_USFM[osis_book]
    except KeyError as error:
        raise ValueError(f"Unknown OpenBible book code: {osis_book!r}") from error
    return book_code, int(match.group("chapter")), int(match.group("verse"))


def openbible_range_label(value: str) -> str:
    """Return a readable label for a single verse or an OSIS verse range."""
    parts = value.split("-", maxsplit=1)
    start_code, start_chapter, start_verse = parse_openbible_reference(parts[0])
    start_book = BOOK_NAMES[start_code]
    if len(parts) == 1:
        return f"{start_book} {start_chapter}:{start_verse}"

    end_code, end_chapter, end_verse = parse_openbible_reference(parts[1])
    end_book = BOOK_NAMES[end_code]
    if end_code == start_code and end_chapter == start_chapter:
        return f"{start_book} {start_chapter}:{start_verse}\u2013{end_verse}"
    if end_code == start_code:
        return f"{start_book} {start_chapter}:{start_verse}\u2013{end_chapter}:{end_verse}"
    return (
        f"{start_book} {start_chapter}:{start_verse}\u2013"
        f"{end_book} {end_chapter}:{end_verse}"
    )


def parse_openbible(web_chunks: list[dict]) -> list[dict]:
    """Parse the OpenBible cross-reference dataset without copying ESV text."""
    path = RAW_DIR / "openbible" / "cross_references.txt"
    if not path.exists():
        raise FileNotFoundError(f"No OpenBible cross-reference file found at {path}")

    book_codes = {name: code for code, name in BOOK_NAMES.items()}
    web_verses: dict[tuple[str, int, int], str] = {}
    for chunk in web_chunks:
        book_code = book_codes.get(str(chunk.get("book", "")))
        chapter = chunk.get("chapter")
        verse_start = chunk.get("verse_start")
        verse_end = chunk.get("verse_end")
        if (
            not book_code
            or not isinstance(chapter, int)
            or not isinstance(verse_start, int)
            or not isinstance(verse_end, int)
        ):
            continue
        for verse in range(verse_start, verse_end + 1):
            web_verses[(book_code, chapter, verse)] = str(chunk["text"])

    grouped: dict[tuple[str, int, int], list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("From Verse") or line.startswith("#"):
                continue
            columns = line.split()
            if len(columns) < 3:
                raise ValueError(
                    f"Invalid OpenBible row on line {line_number}: {raw_line.rstrip()!r}"
                )
            from_raw, to_raw, votes_raw = columns[:3]
            try:
                source = parse_openbible_reference(from_raw)
                # Validate both ends of a target range before storing it.
                for target_part in to_raw.split("-", maxsplit=1):
                    parse_openbible_reference(target_part)
                votes = int(votes_raw)
            except ValueError as error:
                raise ValueError(
                    f"Invalid OpenBible row on line {line_number}: {error}"
                ) from error
            grouped.setdefault(source, []).append((votes, to_raw))

    if not grouped:
        raise ValueError(f"OpenBible file contains no usable rows: {path}")

    missing_web: list[str] = []
    chunks: list[dict] = []
    for (book_code, chapter, verse), links in sorted(grouped.items()):
        source_key = (book_code, chapter, verse)
        source_text = web_verses.get(source_key)
        source_label = f"{BOOK_NAMES[book_code]} {chapter}:{verse}"
        if not source_text:
            missing_web.append(source_label)
            continue

        # Keep all links, but rank them by the dataset's vote count.
        ranked_links = sorted(links, key=lambda item: (-item[0], item[1]))
        reference_lines = [
            f"- {openbible_range_label(target)} ({votes} votes)"
            for votes, target in ranked_links
        ]
        text = "\n\n".join([
            f"Source verse: {source_label}",
            f"World English Bible text: {source_text}",
            "Cross-references ranked by OpenBible.info votes:",
            *reference_lines,
        ])
        chunks.extend(make_chunks(
            "OPENBIBLE_XREF",
            f"{book_code}_{chapter:03d}_{verse:03d}",
            f"Cross-references for {source_label}",
            text,
            book=BOOK_NAMES[book_code],
            chapter=chapter,
            verse_start=verse,
            verse_end=verse,
            extra={
                "from_reference": source_label,
                "cross_reference_count": len(ranked_links),
                "highest_vote_count": ranked_links[0][0],
                "scripture_text_source": "World English Bible (public domain)",
            },
        ))

    if missing_web:
        examples = ", ".join(missing_web[:10])
        print(
            "WARNING: skipped "
            f"{len(missing_web)} OpenBible source verses that could not "
            f"be matched to WEB text. Examples: {examples}",
            file=sys.stderr,
        )

    return chunks


def bible_odyssey_text(page: dict) -> str:
    """Extract article prose while excluding quotation/scripture blocks."""
    blocks = page.get("blocks")
    if isinstance(blocks, list):
        pieces: list[str] = []
        excluded_types = {
            "blockquote", "quotation", "quote", "scripture", "bible_text",
            "caption", "image_caption",
        }
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = normalize_text(str(block.get("type", "paragraph"))).lower()
            if block_type in excluded_types:
                continue
            block_text = normalize_text(str(block.get("text", "")))
            if not block_text:
                continue
            if block_type == "heading":
                pieces.append(f"Section: {block_text}")
            else:
                pieces.append(block_text)
        if pieces:
            return "\n\n".join(pieces)

    for field in ("text", "article_text", "content_text", "body_text"):
        value = page.get(field)
        if isinstance(value, str) and normalize_text(value):
            return normalize_text(value)
    return ""


def parse_bible_odyssey() -> list[dict]:
    """Parse pages produced by the controlled Bible Odyssey scraper."""
    path = RAW_DIR / "bible_odyssey" / "pages.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No Bible Odyssey pages.jsonl found at {path}")

    pages: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                page = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Bible Odyssey JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(page, dict):
                raise ValueError(
                    f"Bible Odyssey line {line_number} is not a JSON object"
                )
            pages.append(page)

    if not pages:
        raise ValueError(f"Bible Odyssey file contains no pages: {path}")

    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    chunks: list[dict] = []
    for page_number, page in enumerate(pages, start=1):
        status = page.get("http_status")
        if status not in (None, 200, "200"):
            raise ValueError(
                f"Bible Odyssey page {page_number} has HTTP status {status!r}"
            )

        title = normalize_text(str(page.get("title", "")))
        canonical_url = normalize_text(str(
            page.get("canonical_url")
            or page.get("final_url")
            or page.get("url")
            or ""
        ))
        body = bible_odyssey_text(page)
        if not title or not canonical_url or not body:
            raise ValueError(
                f"Bible Odyssey page {page_number} needs title, URL, and article text"
            )
        parsed_url = urlparse(canonical_url)
        if (
            parsed_url.scheme != "https"
            or (parsed_url.hostname or "").casefold()
            not in {"bibleodyssey.org", "www.bibleodyssey.org"}
        ):
            raise ValueError(
                f"Bible Odyssey page {page_number} has an unexpected URL: {canonical_url}"
            )

        permission_record = normalize_text(str(page.get("permission_record", "")))
        if not permission_record:
            raise ValueError(
                f"Bible Odyssey page {page_number} has no written-permission record"
            )

        page_id = normalize_text(str(page.get("id", "")))
        if not page_id:
            digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
            page_id = f"bo_{digest}"
        if page_id in seen_ids or canonical_url in seen_urls:
            raise ValueError(
                f"Duplicate Bible Odyssey page identity: {page_id} / {canonical_url}"
            )
        seen_ids.add(page_id)
        seen_urls.add(canonical_url)

        book_tags = page.get("book_tags") or []
        passage_tags = page.get("passage_tags") or []
        if isinstance(book_tags, str):
            book_tags = [book_tags]
        if isinstance(passage_tags, str):
            passage_tags = [passage_tags]
        reference_page = dict(page)
        reference_page["book_tags"] = book_tags
        reference_page["passage_tags"] = passage_tags
        book, chapter, verse_start, verse_end = tow_reference_metadata(reference_page)
        article_text = f"Bible Odyssey article: {title}\n\n{body}"
        chunks.extend(make_chunks(
            "BIBLE_ODYSSEY",
            page_id,
            title,
            article_text,
            book=book,
            chapter=chapter,
            verse_start=verse_start,
            verse_end=verse_end,
            extra={
                "url": canonical_url,
                "canonical_url": canonical_url,
                "source_page_id": page_id,
                "author": page.get("author") or page.get("authors"),
                "published_at": page.get("published_at") or page.get("date_published"),
                "accessed_at": page.get("accessed_at"),
                "content_type": page.get("content_type") or "article",
                "book_tags": book_tags,
                "passage_tags": passage_tags,
                "themes": page.get("themes") or [],
                "bibliography_text": page.get("bibliography_text"),
                "original_html_sha256": page.get("html_sha256"),
                "original_text_sha256": (
                    page.get("text_sha256")
                    or hashlib.sha256(body.encode("utf-8")).hexdigest()
                ),
                "license": page.get("license") or SOURCE_META["BIBLE_ODYSSEY"]["license"],
                "license_url": page.get("license_url") or page.get("terms_url"),
                "permission_record": permission_record,
                "manual_review_required": True,
            },
        ))
    return chunks


def parse_utn_reference(reference: str) -> tuple[int | None, int | None, int | None]:
    match = re.fullmatch(r"(\d+):(\d+)(?:-(\d+))?", reference.strip())
    if not match:
        chapter_match = re.fullmatch(r"(\d+):intro", reference.strip(), flags=re.I)
        return (int(chapter_match.group(1)), None, None) if chapter_match else (None, None, None)
    start = int(match.group(2))
    return int(match.group(1)), start, int(match.group(3) or start)


def parse_utn() -> list[dict]:
    root = RAW_DIR / "unfoldingword_notes"
    files = sorted(
        p for p in root.rglob("*.tsv")
        if usable_file(p) and re.search(r"(?:^|_)\d?[A-Z]{2,3}\.tsv$", p.name)
    )
    if not files:
        raise FileNotFoundError(f"No Translation Notes .tsv files found under {root}")

    chunks: list[dict] = []
    for path in files:
        code_match = re.search(r"(?:^|_)(\d?[A-Z]{2,3})\.tsv$", path.name)
        if not code_match:
            continue
        code = code_match.group(1)
        book = BOOK_NAMES.get(code, code)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=2):
                reference = (row.get("Reference") or "").strip()
                note_id = (row.get("ID") or str(row_number)).strip()
                note = normalize_text((row.get("Note") or "").replace("\\n", "\n"))
                if not note:
                    continue
                quote = normalize_text(row.get("Quote") or "")
                support = normalize_text(row.get("SupportReference") or "")
                parts = []
                if quote:
                    parts.append(f"Quoted expression: {quote}")
                if support:
                    parts.append(f"Supporting topic: {support}")
                parts.append(note)
                chapter, verse_start, verse_end = parse_utn_reference(reference)
                title = f"{book} {reference} translation note"
                chunks.extend(make_chunks(
                    "UTN", f"{code}_{reference}_{note_id}", title,
                    "\n\n".join(parts), book=book, chapter=chapter,
                    verse_start=verse_start, verse_end=verse_end,
                    extra={"note_id": note_id, "reference": reference},
                ))
    return chunks


def markdown_title(text: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return normalize_text(match.group(1)) if match else fallback


def clean_markdown(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[\[([^]]+)\]\]", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"[*_`>]", "", text)
    return normalize_text(text)


def parse_utw() -> list[dict]:
    root = RAW_DIR / "unfoldingword_words"
    files = sorted(
        p for p in root.rglob("*.md")
        if usable_file(p) and "bible" in p.parts
        and p.name.lower() not in {"readme.md", "license.md"}
    )
    if not files:
        raise FileNotFoundError(f"No Translation Words Markdown files found under {root}")

    chunks: list[dict] = []
    for path in files:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        title = markdown_title(raw, path.stem.replace("-", " ").title())
        text = clean_markdown(raw)
        category = path.parent.name
        relative = path.relative_to(root).with_suffix("").as_posix()
        chunks.extend(make_chunks(
            "UTW", relative, title, text,
            extra={"term_category": category, "source_file": relative},
        ))
    return chunks


def parse_step() -> list[dict]:
    root = RAW_DIR / "stepbible"
    files = sorted(
        p for p in root.glob("*.txt")
        if usable_file(p) and (p.name.startswith("TBESH") or p.name.startswith("TBESG"))
    )
    if not files:
        raise FileNotFoundError(f"No TBESH/TBESG .txt files found under {root}")

    chunks: list[dict] = []
    expected_header = [
        "eStrong#", "dStrong", "uStrong", "Original", "Transliteration",
        "Morph", "Gloss", "Meaning",
    ]
    for path in files:
        language = "Hebrew" if path.name.startswith("TBESH") else "Greek"
        dataset = "TBESH" if language == "Hebrew" else "TBESG"
        header: list[str] | None = None
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, columns in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if not columns:
                    continue
                if columns[0] == "eStrong#":
                    header = columns
                    continue
                if header is None or len(columns) < 7:
                    continue
                columns += [""] * (len(header) - len(columns))
                row = dict(zip(header or expected_header, columns))
                strong = normalize_text(row.get("eStrong#", ""))
                if not re.match(r"^[GH]\d", strong):
                    continue
                gloss = normalize_text(plain_html(row.get("Gloss", "")))
                # The long Meaning field is intentionally not indexed here.
                # TBESH itself states that its Abridged-BDB-derived definitions
                # require separate permission from Online Bible. The STEP-created
                # gloss and structured lexical fields are retained.
                fields = [
                    f"Extended Strong number: {strong}",
                    f"Disambiguated Strong number: {normalize_text(row.get('dStrong', ''))}",
                    f"Unified Strong number: {normalize_text(row.get('uStrong', ''))}",
                    f"{language} form: {normalize_text(row.get('Hebrew', row.get('Greek', row.get('Original', ''))))}",
                    f"Transliteration: {normalize_text(row.get('Transliteration', ''))}",
                    f"Morphology: {normalize_text(row.get('Morph', ''))}",
                    f"Gloss: {gloss}",
                ]
                text = "\n".join(field for field in fields if not field.endswith(": "))
                title = f"{strong}: {gloss}" if gloss else strong
                chunks.extend(make_chunks(
                    "STEP", f"{dataset}_{row_number}_{strong}", title, text,
                    extra={"dataset": dataset, "strong_number": strong, "language": language},
                ))
    return chunks


def parse_oshb() -> list[dict]:
    root = RAW_DIR / "oshb"
    files = sorted(p for p in root.rglob("*.xml") if usable_file(p))
    if not files:
        raise FileNotFoundError(
            f"No usable OSHB XML files found under {root}. Files inside __MACOSX are metadata only."
        )

    chunks: list[dict] = []
    for path in files:
        try:
            tree = ET.parse(path)
        except ET.ParseError as error:
            print(f"WARNING: skipping malformed XML {path}: {error}", file=sys.stderr)
            continue
        for verse in tree.getroot().iter():
            if local_name(verse.tag) != "verse":
                continue
            osis_id = verse.attrib.get("osisID")
            if not osis_id:
                continue
            reference_parts = osis_id.split(".")
            if len(reference_parts) < 3:
                continue
            osis_book, chapter_text, verse_text = reference_parts[:3]
            try:
                chapter = int(chapter_text)
                verse_number = int(re.match(r"\d+", verse_text).group())
            except (AttributeError, ValueError):
                continue
            code = OSIS_TO_USFM.get(osis_book, osis_book.upper())
            book = BOOK_NAMES.get(code, osis_book)
            word_rows = []
            surface_words = []
            for word in verse.iter():
                if local_name(word.tag) != "w":
                    continue
                surface = normalize_text("".join(word.itertext()))
                if not surface:
                    continue
                surface_words.append(surface)
                lemma = normalize_text(word.attrib.get("lemma", ""))
                morph = normalize_text(word.attrib.get("morph", ""))
                word_rows.append(f"{surface} | lemma={lemma or 'NA'} | morphology={morph or 'NA'}")
            if not word_rows:
                continue
            text = "Hebrew text: " + " ".join(surface_words)
            text += "\nWord-level lemma and morphology:\n" + "\n".join(word_rows)
            chunks.extend(make_chunks(
                "OSHB", f"{code}_{chapter:03d}_{verse_number:03d}",
                f"{book} {chapter}:{verse_number} Hebrew morphology",
                text, book=book, chapter=chapter,
                verse_start=verse_number, verse_end=verse_number,
                extra={"osis_id": osis_id},
            ))
    return chunks


def validate(chunks: Iterable[dict]) -> tuple[list[dict], dict]:
    chunks = list(chunks)
    required = {
        "chunk_id", "source_id", "source_name", "source_type", "title",
        "text", "version", "license", "url",
    }
    ids = Counter(chunk.get("chunk_id") for chunk in chunks)
    duplicates = sorted(chunk_id for chunk_id, count in ids.items() if count > 1)
    malformed = [
        index for index, chunk in enumerate(chunks, start=1)
        if required - chunk.keys() or not str(chunk.get("text", "")).strip()
    ]
    if duplicates:
        raise ValueError(f"Duplicate chunk IDs found; examples: {duplicates[:10]}")
    if malformed:
        raise ValueError(f"Malformed or empty chunks found at positions: {malformed[:10]}")

    ineligible = [
        chunk["chunk_id"] for chunk in chunks
        if chunk.get("index_eligible") is not True
    ]
    if ineligible:
        raise ValueError(
            "A licence- or policy-gated record reached the normalized corpus: "
            f"{ineligible[:10]}"
        )

    chunks.sort(key=lambda item: item["chunk_id"])
    counts = Counter(chunk["source_id"] for chunk in chunks)
    tow_chunks = [chunk for chunk in chunks if chunk["source_id"] == "TOW"]
    pdf_commentary_chunks = [
        chunk for chunk in chunks if chunk["source_id"] in PDF_COMMENTARY_SOURCE_IDS
    ]
    tow_pages = {
        str(chunk.get("source_page_id"))
        for chunk in tow_chunks
        if chunk.get("source_page_id")
    }
    tow_page_quote_counts: dict[str, int] = {}
    tow_page_table_counts: dict[str, int] = {}
    for chunk in tow_chunks:
        page_id = str(chunk.get("source_page_id", ""))
        if page_id and page_id not in tow_page_quote_counts:
            tow_page_quote_counts[page_id] = int(
                chunk.get("inline_quoted_spans_removed_from_page", 0)
            )
            tow_page_table_counts[page_id] = int(
                chunk.get("table_source_cells_removed_from_page", 0)
            )
    report = {
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "total_chunks": len(chunks),
        "chunks_by_source": dict(sorted(counts.items())),
        "sha256": None,
        "theology_of_work": {
            "status": "included from controlled Bible Commentary crawl",
            "source_pages": len(tow_pages),
            "chunks": len(tow_chunks),
            "license": "CC BY-NC 4.0",
            "inline_quoted_spans_removed": sum(tow_page_quote_counts.values()),
            "table_source_cells_removed": sum(tow_page_table_counts.values()),
            "manual_sample_review_still_recommended": True,
        },
        "historical_academic_commentaries": {
            "status": "included as scholarly interpretation, not Bible text",
            "chunks": len(pdf_commentary_chunks),
            "source_ids": sorted({chunk["source_id"] for chunk in pdf_commentary_chunks}),
            "raw_page_provenance_retained": True,
        },
        "source_gates": {
            "pdf_confessions": "held: index_eligible=false and license_status=review_required",
            "baptist_1689": "held: website-transcription terms review_required",
            "bible_odyssey": "held: written permission required",
        },
    }
    return chunks, report

def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print("Building normalized corpus...")

    # Parse WEB first because the OpenBible parser uses the WEB chunks
    # to attach public-domain verse text to the cross-references.
    web_chunks = parse_web()

    all_chunks: list[dict] = list(web_chunks)
    print(f"  WEB: {len(web_chunks):,} chunks")

    parsers = [
        ("OPENBIBLE_XREF", lambda: parse_openbible(web_chunks)),
        ("STEP", parse_step),
        ("OSHB", parse_oshb),
        ("UTN", parse_utn),
        ("UTW", parse_utw),
        ("TOW", parse_tow),
        ("PDF_COMMENTARIES", parse_pdf_sources),
        ("CONF_BAPTIST_1689", inspect_1689_held_source),
    ]

    bible_odyssey_file = RAW_DIR / "bible_odyssey" / "pages.jsonl"
    if not SOURCE_META["BIBLE_ODYSSEY"]["index_eligible"]:
        print(
            "  BIBLE_ODYSSEY: held out pending written permission for download, "
            "storage, extraction, embeddings, RAG use, and excerpt display."
        )
    elif (
        usable_file(bible_odyssey_file)
        and bible_odyssey_file.is_file()
        and bible_odyssey_file.stat().st_size > 0
    ):
        # The controlled scraper has produced at least one permitted article.
        parsers.insert(1, ("BIBLE_ODYSSEY", parse_bible_odyssey))
    else:
        print(
            "  BIBLE_ODYSSEY: not included because pages.jsonl is empty. "
            "Run scripts.scrape_bible_odyssey after written permission is recorded."
        )

    for source_id, parser in parsers:
        source_chunks = parser()
        all_chunks.extend(source_chunks)
        print(f"  {source_id}: {len(source_chunks):,} chunks")

    chunks, report = validate(all_chunks)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report["sha256"] = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {report['total_chunks']:,} chunks "
        f"to {OUTPUT_FILE}"
    )
    print(f"Build report: {REPORT_FILE}")
    print(f"SHA-256: {report['sha256']}")

if __name__ == "__main__":
    main()
