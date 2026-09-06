"""Build a normalized JSONL corpus for the Bible-study RAG system.

Run this file from the oTree project directory with:

    python -m scripts.build_corpus

Expected project layout:

    corpus/raw/oshb/
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "corpus" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
REPORT_FILE = OUTPUT_DIR / "build_report.json"

# Character based splitting is used here so no tokenizer dependency is needed.
# This remains comfortably below the embedding model's input limit.
MAX_CHARS = 3_500


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
}


SOURCE_META = {
    "WEB": {
        "source_name": "World English Bible, Protestant edition",
        "source_type": "bible_text",
        "version": "engwebp",
        "license": "Public domain",
        "url": "https://ebible.org/find/details.php?id=engwebp",
    },
    "STEP": {
        "source_name": "STEPBible TBESH/TBESG brief lexicons",
        "source_type": "lexical_data",
        "version": "downloaded repository snapshot",
        "license": "CC BY 4.0",
        "url": "https://github.com/STEPBible/STEPBible-Data/tree/master/Lexicons",
    },
    "OSHB": {
        "source_name": "Open Scriptures Hebrew Bible",
        "source_type": "hebrew_morphology",
        "version": "v2.2",
        "license": "WLC text public domain; lemma and morphology CC BY 4.0",
        "url": "https://github.com/openscriptures/morphhb/releases/tag/v.2.2",
    },
    "UTN": {
        "source_name": "unfoldingWord Translation Notes",
        "source_type": "translation_note",
        "version": "v90",
        "license": "CC BY-SA 4.0",
        "url": "https://git.door43.org/unfoldingWord/en_tn/releases/tag/v90",
    },
    "UTW": {
        "source_name": "unfoldingWord Translation Words",
        "source_type": "biblical_term",
        "version": "v90",
        "license": "CC BY-SA 4.0",
        "url": "https://git.door43.org/unfoldingWord/en_tw/releases/tag/v90",
    },
    "TOW": {
        "source_name": "Theology of Work Bible Commentary",
        "source_type": "commentary",
        "version": "Website snapshot 2026-08-29 (scraper v2.1.0)",
        "license": "CC BY-NC 4.0",
        "url": "https://www.theologyofwork.org/resources/the-theology-of-work-bible-commentary",
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
    """Infer the first explicit Bible reference on a TOW page."""
    candidates = [
        normalize_text(str(page.get("passage", ""))),
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

    for tag in page.get("book_tags", []):
        book = TOW_BOOK_TAG_NAMES.get(str(tag).lower())
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

    chunks.sort(key=lambda item: item["chunk_id"])
    counts = Counter(chunk["source_id"] for chunk in chunks)
    tow_chunks = [chunk for chunk in chunks if chunk["source_id"] == "TOW"]
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
    }
    return chunks, report


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print("Building normalized corpus...")
    parsers = [
        ("WEB", parse_web),
        ("STEP", parse_step),
        ("OSHB", parse_oshb),
        ("UTN", parse_utn),
        ("UTW", parse_utw),
        ("TOW", parse_tow),
    ]
    all_chunks: list[dict] = []
    for source_id, parser in parsers:
        source_chunks = parser()
        all_chunks.extend(source_chunks)
        print(f"  {source_id}: {len(source_chunks):,} chunks")

    chunks, report = validate(all_chunks)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    report["sha256"] = hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest()
    REPORT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report['total_chunks']:,} chunks to {OUTPUT_FILE}")
    print(f"Build report: {REPORT_FILE}")
    print(f"SHA-256: {report['sha256']}")


if __name__ == "__main__":
    main()
