"""Controlled Theology of Work Bible Commentary scraper for academic RAG.

Install as ``scripts/scrape_theology_of_work.py`` and run from the project root.

Modes
-----
Pilot (default):
    python -m scripts.scrape_theology_of_work --pilot

Discover the complete English Bible Commentary URL set without changing output:
    python -m scripts.scrape_theology_of_work --full --discover-only

Scrape a small resumable full-corpus batch:
    python -m scripts.scrape_theology_of_work --full --max-new-pages 20

Resume/finish the complete Bible Commentary crawl:
    python -m scripts.scrape_theology_of_work --full

Outputs
-------
corpus/raw/theology_of_work/pages.jsonl
    Accepted, cleaned page records (one JSON object per line).
corpus/raw/theology_of_work/rejected_pages.jsonl
    Rejected or failed page records (one JSON object per line).
corpus/raw/theology_of_work/crawl_manifest.json
    Discovery, progress, policy, and reproducibility metadata.

Full mode discovers URLs only from sitemap child files named
``sitemap_sections_2_*.xml`` and only considers ``/old-testament/`` and
``/new-testament/`` paths. Every candidate must identify itself as Bible
Commentary, carry an eligible producer designation, and expose a supported
Creative Commons licence. The script never crawls sitemap_resources.xml or
sitemap_thc.xml.

The scraper checkpoints atomically, resumes by input URL, and makes an automatic
backup when converting pilot output into a full-commentary dataset. Standalone
Bible quotation blocks are removed. Short quotations embedded in commentary
still require manual review before indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SCRIPT_VERSION = "2.1.0"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "raw" / "theology_of_work"
PAGES_PATH = OUTPUT_DIR / "pages.jsonl"
REJECTED_PATH = OUTPUT_DIR / "rejected_pages.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "crawl_manifest.json"
BACKUP_ROOT = OUTPUT_DIR / "backups"

ROBOTS_URL = "https://www.theologyofwork.org/robots.txt"
SITEMAP_INDEX_URL = "https://www.theologyofwork.org/sitemap.xml"
LICENSE_URL = "https://www.theologyofwork.org/about/cc-license"

ALLOWED_HOSTS = {
    "theologyofwork.org",
    "www.theologyofwork.org",
    "static.theologyofwork.org",
}
COMMENTARY_PATH_PREFIXES = ("/old-testament/", "/new-testament/")
COMMENTARY_SITEMAP_RE = re.compile(r"/sitemap_sections_2_\d+\.xml$")

ALLOWED_PRODUCER_MARKERS = (
    "produced by tow project",
    "produced by individual tow project member",
    "produced by the high calling",
)

TERMINAL_REJECTION_REASONS = {
    "blocked_by_robots",
    "host_not_allowed",
    "redirected_to_disallowed_host",
    "unexpected_content_type",
    "required_html_element_missing",
    "producer_not_eligible",
    "content_type_not_bible_commentary",
    "supported_license_not_detected",
    "clean_text_too_short",
    "duplicate_clean_text",
}

PILOT_PAGES: tuple[dict[str, str], ...] = (
    {
        "passage": "2 Kings 5:9-14",
        "role": "experimental",
        "url": (
            "https://www.theologyofwork.org/old-testament/"
            "samuel-kings-chronicles-and-work/"
            "from-failed-monarchies-to-exile-1-kings-11-2-kings-25-2-chronicles-10-36/"
            "the-prophet-elishas-attention-to-ordinary-work-2-kings-2-6/"
            "elishas-restoration-of-a-military-commanders-health-2-kings-51-14/"
        ),
    },
    {
        "passage": "Romans 12:1-5",
        "role": "experimental",
        "url": (
            "https://www.theologyofwork.org/new-testament/romans-and-work/"
            "the-community-of-grace-at-work-romans-12/"
        ),
    },
    {
        "passage": "1 Samuel 8:4-9",
        "role": "experimental",
        "url": (
            "https://www.theologyofwork.org/old-testament/"
            "samuel-kings-chronicles-and-work/"
            "from-tribal-confederation-to-monarchy-1-samuel/"
            "the-israelites-ask-for-a-king-1-samuel-84-22"
        ),
    },
    {
        "passage": "1 Corinthians 8:1-6",
        "role": "experimental",
        "url": (
            "https://www.theologyofwork.org/the-high-calling/"
            "whats-wrong-knowledge/"
        ),
    },
    {
        "passage": "Mark 4:35-41",
        "role": "practice",
        "url": (
            "https://www.theologyofwork.org/new-testament/mark/"
            "kingdom-and-discipleship/discipleship-in-process/"
        ),
    },
)

ARTICLE_SELECTORS = (
    "section#content .body .body-main",
    "section#content .body-main",
    "main article",
    "article",
)

NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "svg",
    "canvas",
    "video",
    "audio",
    "picture",
    "img",
    "figure",
    "figcaption",
    # Full Scripture quotation blocks often contain NRSV/NIV text.
    "blockquote",
    # On TOW pages these are inset stories/media, not core commentary tables.
    "table.embed_table",
    ".back-to-contents",
    ".navigation-links",
    ".footnoteinfo",
    ".allfootnoteinfo",
    ".hidden",
    "[id*='bedrock']",
    "[class*='bedrock']",
    "[id*='tow-ai']",
    "[class*='tow-ai']",
)

SPACE_RE = re.compile(r"\s+")
VERSE_REFERENCE_RE = re.compile(
    r"^(?:[1-3]\s*)?[A-Za-z]+(?:\s+[A-Za-z]+)?\s+\d+:\d+(?:\s*[-\u2013]\s*\d+)?$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    value = value.replace("\u00ad", "")
    return SPACE_RE.sub(" ", value).strip()


def normalize_url(value: str) -> str:
    value = normalize_text(value)
    parsed = urlsplit(value)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    hostname = (parsed.hostname or "").lower()
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, payload)


def write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object in {path}, line {line_number}")
            records.append(value)
    return records


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0:
        return {}
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {MANIFEST_PATH}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def build_session(contact_email: str) -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                f"TUM-Bible-RAG-Research/{SCRIPT_VERSION} "
                f"(non-commercial academic research; contact: {contact_email})"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml,text/xml",
            "Accept-Language": "en",
        }
    )
    return session


def load_robots(session: requests.Session) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(ROBOTS_URL)
    try:
        response = session.get(ROBOTS_URL, timeout=(10, 30))
        response.raise_for_status()
        parser.parse(response.text.splitlines())
        parser.modified()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not read robots.txt: {exc}") from exc
    return parser


def fetch_sitemap_locations(
    session: requests.Session,
    robots: RobotFileParser,
    url: str,
) -> tuple[list[str], str]:
    if not robots.can_fetch(session.headers["User-Agent"], url):
        raise RuntimeError(f"robots.txt does not permit sitemap: {url}")
    try:
        response = session.get(url, timeout=(10, 60))
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch sitemap {url}: {exc}") from exc
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid XML sitemap {url}: {exc}") from exc

    root_type = root.tag.rsplit("}", 1)[-1]
    locations = [
        normalize_text(element.text or "")
        for element in root.iter()
        if element.tag.endswith("loc") and normalize_text(element.text or "")
    ]
    return locations, root_type


def discover_full_commentary_pages(
    session: requests.Session,
    robots: RobotFileParser,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    children, root_type = fetch_sitemap_locations(session, robots, SITEMAP_INDEX_URL)
    if root_type != "sitemapindex":
        raise RuntimeError(f"Expected sitemap index at {SITEMAP_INDEX_URL}")

    commentary_sitemaps = [
        url
        for url in children
        if COMMENTARY_SITEMAP_RE.search(urlparse(url).path)
    ]
    if not commentary_sitemaps:
        raise RuntimeError("No sitemap_sections_2_*.xml entries were discovered")

    discovered: list[str] = []
    sitemap_counts: dict[str, int] = {}
    for index, sitemap_url in enumerate(commentary_sitemaps, start=1):
        locations, child_type = fetch_sitemap_locations(session, robots, sitemap_url)
        if child_type != "urlset":
            raise RuntimeError(f"Expected URL set at {sitemap_url}")
        sitemap_counts[sitemap_url] = len(locations)
        print(
            f"  Sitemap {index}/{len(commentary_sitemaps)}: "
            f"{len(locations):,} URLs"
        )
        for location in locations:
            parsed = urlparse(location)
            if parsed.hostname not in ALLOWED_HOSTS:
                continue
            if not parsed.path.startswith(COMMENTARY_PATH_PREFIXES):
                continue
            discovered.append(normalize_url(location))

    unique_urls = list(dict.fromkeys(discovered))
    pilot_metadata = {normalize_url(page["url"]): page for page in PILOT_PAGES}
    candidate_paths = {
        urlparse(url).path.rstrip("/")
        for url in unique_urls
    }
    pages: list[dict[str, Any]] = []
    for url in unique_urls:
        metadata = pilot_metadata.get(url, {})
        path = urlparse(url).path.rstrip("/")
        has_children = any(
            other_path.startswith(path + "/")
            for other_path in candidate_paths
            if other_path != path
        )
        pages.append(
            {
                "url": url,
                "passage": metadata.get("passage", ""),
                "role": metadata.get("role", "corpus"),
                "has_children": has_children,
            }
        )

    summary = {
        "sitemap_index": SITEMAP_INDEX_URL,
        "commentary_sitemaps": commentary_sitemaps,
        "sitemap_url_counts": sitemap_counts,
        "candidate_count": len(pages),
        "parent_page_count": sum(bool(page["has_children"]) for page in pages),
        "leaf_page_count": sum(not bool(page["has_children"]) for page in pages),
        "path_prefixes": list(COMMENTARY_PATH_PREFIXES),
    }
    return pages, summary


def select_first(soup: BeautifulSoup | Tag, selectors: tuple[str, ...]) -> Tag | None:
    for selector in selectors:
        match = soup.select_one(selector)
        if isinstance(match, Tag):
            return match
    return None


def get_canonical_url(soup: BeautifulSoup, final_url: str) -> str:
    canonical = soup.select_one("link[rel='canonical'][href]")
    candidate = canonical.get("href", "") if isinstance(canonical, Tag) else ""
    candidate = normalize_text(str(candidate)) or final_url
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_HOSTS:
        return normalize_url(final_url)
    return normalize_url(candidate)


def parse_type_and_producer(label: str) -> tuple[str, str]:
    parts = [normalize_text(part) for part in label.split("/", 1)]
    content_type = parts[0] if parts else ""
    producer = parts[1] if len(parts) == 2 else ""
    return content_type, producer


def detect_license(copyright_text: str) -> tuple[str, str]:
    lowered = copyright_text.lower()
    if "attribution-noncommercial 4.0" in lowered or "by-nc/4.0" in lowered:
        return "CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"
    if "attribution 4.0" in lowered or "by/4.0" in lowered:
        return "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"
    return "", ""


def detect_scripture_translation(copyright_text: str) -> list[str]:
    lowered = copyright_text.lower()
    translations: list[str] = []
    if "new revised standard version" in lowered or "nrsv" in lowered:
        translations.append("NRSV")
    if "new international version" in lowered or "niv" in lowered:
        translations.append("NIV")
    return translations


def extract_contributors(copyright_text: str) -> str:
    match = re.search(
        r"Contributors?:\s*(.+?)(?:\s+Adopted by|\s+Published by|\s+Theology of Work)",
        copyright_text,
        flags=re.IGNORECASE,
    )
    return normalize_text(match.group(1)) if match else ""


def extract_book_tags(soup: BeautifulSoup) -> list[str]:
    tags: list[str] = []
    for link in soup.select(".sidebar-tags a[href*='book=']"):
        href = str(link.get("href", ""))
        values = parse_qs(urlparse(href).query).get("book", [])
        for value in values:
            value = normalize_text(value)
            if value:
                tags.append(value)
    return list(dict.fromkeys(tags))


def extract_passage_reference(title: str) -> str:
    groups = re.findall(r"\(([^()]*)\)", title)
    candidates = [
        normalize_text(group)
        for group in groups
        if re.search(r"\b\d{1,3}(?::\d{1,3})?", group)
    ]
    return "; ".join(candidates)


def extract_clean_content(
    main: Tag,
    *,
    remove_aggregated_children: bool,
) -> tuple[str, list[dict[str, str]], dict[str, int]]:
    clean = deepcopy(main)
    removed_counts = {
        "standalone_scripture_blocks": len(clean.select("blockquote")),
        "embedded_tables": len(clean.select("table.embed_table")),
        "forms": len(clean.select("form")),
        "leading_scripture_paragraphs": 0,
        "aggregated_child_sections": 0,
    }

    if remove_aggregated_children:
        aggregated = clean.select(".short-wrap")
        removed_counts["aggregated_child_sections"] = len(aggregated)
        for node in aggregated:
            node.decompose()

    for selector in NOISE_SELECTORS:
        for node in clean.select(selector):
            node.decompose()

    # Some pages place a full Bible quotation in a paragraph followed by a
    # paragraph containing only its verse reference rather than a blockquote.
    for parent in [clean, *clean.find_all(["div", "section"])]:
        direct_paragraphs = parent.find_all("p", recursive=False)
        if len(direct_paragraphs) < 2:
            continue
        for index in range(1, len(direct_paragraphs)):
            current = direct_paragraphs[index]
            previous = direct_paragraphs[index - 1]
            if current.parent is None or previous.parent is None:
                continue
            reference = normalize_text(current.get_text(" ", strip=True))
            previous_text = normalize_text(previous.get_text(" ", strip=True))
            if VERSE_REFERENCE_RE.fullmatch(reference) and len(previous_text) >= 80:
                previous.decompose()
                current.decompose()
                removed_counts["leading_scripture_paragraphs"] += 2

    blocks: list[dict[str, str]] = []
    last_signature: tuple[str, str] | None = None

    for node in clean.find_all(["h2", "h3", "h4", "h5", "p", "table"]):
        if node.name == "p" and node.find_parent("table") is not None:
            continue
        if node.name == "table" and node.find_parent("table") is not None:
            continue

        if node.name == "table":
            for row in node.find_all("tr"):
                cells = [
                    normalize_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                cells = [cell for cell in cells if cell]
                if not cells:
                    continue
                text = " | ".join(cells)
                signature = ("table_row", text)
                if signature != last_signature:
                    blocks.append({"type": "table_row", "text": text})
                    last_signature = signature
            continue

        text = normalize_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if text.lower() in {"back to table of contents", "view full article"}:
            continue
        kind = "heading" if node.name.startswith("h") else "paragraph"
        signature = (kind, text)
        if signature == last_signature:
            continue
        blocks.append({"type": kind, "text": text})
        last_signature = signature

    rendered: list[str] = []
    for block in blocks:
        if block["type"] == "heading":
            rendered.append(f"## {block['text']}")
        else:
            rendered.append(block["text"])
    text = "\n\n".join(rendered).strip()
    return text, blocks, removed_counts


def rejection_record(
    page: dict[str, Any],
    reason: str,
    *,
    status_code: int | None = None,
    final_url: str = "",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "scraper_version": SCRIPT_VERSION,
        "input_url": normalize_url(page["url"]),
        "passage": page.get("passage", ""),
        "role": page.get("role", "corpus"),
        "status": "rejected",
        "reason": reason,
        "terminal": reason in TERMINAL_REJECTION_REASONS,
        "detail": detail,
        "http_status": status_code,
        "final_url": normalize_url(final_url) if final_url else "",
        "checked_at": utc_now(),
    }


def scrape_page(
    session: requests.Session,
    page: dict[str, Any],
    robots: RobotFileParser,
    *,
    require_bible_commentary: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    input_url = normalize_url(page["url"])
    if not robots.can_fetch(session.headers["User-Agent"], input_url):
        return None, rejection_record(page, "blocked_by_robots")

    if urlparse(input_url).hostname not in ALLOWED_HOSTS:
        return None, rejection_record(page, "host_not_allowed")

    try:
        response = session.get(input_url, timeout=(10, 60), allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        final = exc.response.url if exc.response is not None else ""
        return None, rejection_record(
            page,
            "request_failed",
            status_code=status,
            final_url=final,
            detail=f"{type(exc).__name__}: {exc}",
        )

    final_host = urlparse(response.url).hostname
    if final_host not in ALLOWED_HOSTS:
        return None, rejection_record(
            page,
            "redirected_to_disallowed_host",
            status_code=response.status_code,
            final_url=response.url,
        )

    content_type_header = response.headers.get("Content-Type", "")
    if "text/html" not in content_type_header.lower():
        return None, rejection_record(
            page,
            "unexpected_content_type",
            status_code=response.status_code,
            final_url=response.url,
            detail=content_type_header,
        )

    soup = BeautifulSoup(response.content, "lxml")
    title_node = soup.select_one("section#content h1") or soup.select_one("h1")
    label_node = soup.select_one("section#content .body-lead-title span")
    copyright_node = soup.select_one(".sidebar-copyright")
    main = select_first(soup, ARTICLE_SELECTORS)

    if not all(isinstance(node, Tag) for node in (title_node, copyright_node, main)):
        return None, rejection_record(
            page,
            "required_html_element_missing",
            status_code=response.status_code,
            final_url=response.url,
        )

    title = normalize_text(title_node.get_text(" ", strip=True))
    copyright_text = normalize_text(copyright_node.get_text(" ", strip=True))
    label_inferred = False
    if isinstance(label_node, Tag):
        producer_label = normalize_text(label_node.get_text(" ", strip=True))
    elif require_bible_commentary and bool(page.get("has_children")):
        producer_label = "Bible Commentary / Produced by TOW Project"
        label_inferred = True
    else:
        return None, rejection_record(
            page,
            "required_html_element_missing",
            status_code=response.status_code,
            final_url=response.url,
            detail="body-lead content-type/producer label missing",
        )
    content_type, producer = parse_type_and_producer(producer_label)

    if require_bible_commentary and content_type.lower() != "bible commentary":
        return None, rejection_record(
            page,
            "content_type_not_bible_commentary",
            status_code=response.status_code,
            final_url=response.url,
            detail=producer_label,
        )

    producer_lower = producer_label.lower()
    if not any(marker in producer_lower for marker in ALLOWED_PRODUCER_MARKERS):
        return None, rejection_record(
            page,
            "producer_not_eligible",
            status_code=response.status_code,
            final_url=response.url,
            detail=producer_label,
        )

    license_name, page_license_url = detect_license(copyright_text)
    if not license_name:
        return None, rejection_record(
            page,
            "supported_license_not_detected",
            status_code=response.status_code,
            final_url=response.url,
        )

    clean_text, blocks, removed_counts = extract_clean_content(
        main,
        remove_aggregated_children=(
            require_bible_commentary and bool(page.get("has_children"))
        ),
    )
    if len(clean_text) < 200:
        return None, rejection_record(
            page,
            "clean_text_too_short",
            status_code=response.status_code,
            final_url=response.url,
            detail=f"{len(clean_text)} characters",
        )

    scripture_translations = detect_scripture_translation(copyright_text)
    canonical_url = get_canonical_url(soup, response.url)
    passage = page.get("passage", "") or extract_passage_reference(title)
    scope = "bible_commentary_full" if require_bible_commentary else "five_page_pilot"

    record: dict[str, Any] = {
        "scraper_version": SCRIPT_VERSION,
        "id": "tow_" + sha256_text(canonical_url)[:16],
        "source": "Theology of Work",
        "source_type": "commentary",
        "scope": scope,
        "title": title,
        "passage": passage,
        "study_role": page.get("role", "corpus"),
        "hierarchy_role": (
            "parent_summary" if bool(page.get("has_children")) else "leaf"
        ),
        "book_tags": extract_book_tags(soup),
        "content_type": content_type,
        "producer": producer,
        "producer_label": producer_label,
        "producer_label_inferred": label_inferred,
        "contributors": extract_contributors(copyright_text),
        "input_url": input_url,
        "final_url": normalize_url(response.url),
        "canonical_url": canonical_url,
        "accessed_at": utc_now(),
        "http_status": response.status_code,
        "license": license_name,
        "license_url": page_license_url,
        "tow_license_page": LICENSE_URL,
        "copyright_notice": copyright_text,
        "scripture_translations_noted": scripture_translations,
        "manual_review_required": bool(scripture_translations),
        "review_note": (
            "Standalone quotation blocks were removed. Audit sampled pages and "
            "remaining inline/third-party quotations before final indexing."
        ),
        "removed_elements": removed_counts,
        "html_sha256": hashlib.sha256(response.content).hexdigest(),
        "text_sha256": sha256_text(clean_text),
        "character_count": len(clean_text),
        "block_count": len(blocks),
        "blocks": blocks,
        "text": clean_text,
    }
    return record, None


def record_key(record: dict[str, Any]) -> str:
    return normalize_url(str(record.get("input_url", "")))


def backup_existing_outputs() -> str:
    existing = [
        path
        for path in (PAGES_PATH, REJECTED_PATH, MANIFEST_PATH)
        if path.exists() and path.stat().st_size > 0
    ]
    if not existing:
        return ""
    destination = BACKUP_ROOT / timestamp_slug()
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, destination / path.name)
    return str(destination.relative_to(PROJECT_ROOT))


def ordered_records(
    candidates: list[dict[str, Any]],
    records_by_url: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        records_by_url[normalize_url(page["url"])]
        for page in candidates
        if normalize_url(page["url"]) in records_by_url
    ]


def pending_urls(
    candidates: list[dict[str, Any]],
    accepted: dict[str, dict[str, Any]],
    rejected: dict[str, dict[str, Any]],
    *,
    retry_all_rejected: bool,
) -> list[str]:
    pending: list[str] = []
    for page in candidates:
        key = normalize_url(page["url"])
        if key in accepted:
            continue
        rejection = rejected.get(key)
        if (
            rejection is not None
            and bool(rejection.get("terminal"))
            and not retry_all_rejected
        ):
            continue
        pending.append(key)
    return pending


def save_state(
    *,
    mode: str,
    status: str,
    candidates: list[dict[str, Any]],
    accepted: dict[str, dict[str, Any]],
    rejected: dict[str, dict[str, Any]],
    discovery: dict[str, Any],
    delay: float,
    checkpoint_every: int,
    run_started_at: str,
    processed_this_run: int,
    backup_path: str,
    retry_all_rejected: bool,
) -> dict[str, Any]:
    pending = pending_urls(
        candidates,
        accepted,
        rejected,
        retry_all_rejected=retry_all_rejected,
    )
    accepted_rows = ordered_records(candidates, accepted)
    rejected_rows = ordered_records(candidates, rejected)
    write_jsonl(PAGES_PATH, accepted_rows)
    write_jsonl(REJECTED_PATH, rejected_rows)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "source": "Theology of Work",
        "mode": mode,
        "status": status,
        "run_started_at": run_started_at,
        "last_checkpoint_at": utc_now(),
        "robots_url": ROBOTS_URL,
        "robots_result": "all requested URLs permitted",
        "license_policy_url": LICENSE_URL,
        "request_delay_seconds": delay,
        "checkpoint_every_pages": checkpoint_every,
        "candidate_count": len(candidates),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rejected_rows),
        "terminal_rejected_count": sum(
            bool(record.get("terminal")) for record in rejected_rows
        ),
        "transient_rejected_count": sum(
            not bool(record.get("terminal")) for record in rejected_rows
        ),
        "pending_count": len(pending),
        "processed_this_run": processed_this_run,
        "backup_created": backup_path,
        "discovery": discovery,
        "output_files": {
            "pages": str(PAGES_PATH.relative_to(PROJECT_ROOT)),
            "rejected_pages": str(REJECTED_PATH.relative_to(PROJECT_ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        },
        "content_policy": {
            "eligible_producer_markers": list(ALLOWED_PRODUCER_MARKERS),
            "full_mode_requires_content_type": "Bible Commentary",
            "standalone_blockquotes_removed": True,
            "embedded_tables_removed": True,
            "ordinary_content_tables_retained": True,
            "raw_html_stored": False,
            "manual_inline_quotation_review_required": True,
        },
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def prepare_existing_state(
    mode: str,
    candidates: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
]:
    old_manifest = load_manifest()
    old_mode = str(old_manifest.get("mode", ""))
    old_script_version = str(old_manifest.get("script_version", ""))
    backup_path = ""

    if mode == "pilot" and old_mode == "full_bible_commentary":
        raise RuntimeError(
            "Existing outputs contain a full Bible Commentary crawl. "
            "Run with --full; pilot mode will not overwrite it."
        )
    if mode == "full_bible_commentary" and (
        old_mode != "full_bible_commentary"
        or old_script_version != SCRIPT_VERSION
    ):
        backup_path = backup_existing_outputs()

    candidate_keys = {normalize_url(page["url"]) for page in candidates}
    accepted: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(PAGES_PATH):
        key = record_key(record)
        if key not in candidate_keys:
            continue
        if mode == "full_bible_commentary" and str(
            record.get("content_type", "")
        ).lower() != "bible commentary":
            continue
        if mode == "full_bible_commentary":
            if record.get("scraper_version") != SCRIPT_VERSION:
                continue
            record["scope"] = "bible_commentary_full"
        accepted[key] = record

    rejected: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(REJECTED_PATH):
        key = record_key(record)
        if (
            key in candidate_keys
            and key not in accepted
            and (
                mode != "full_bible_commentary"
                or record.get("scraper_version") == SCRIPT_VERSION
            )
        ):
            rejected[key] = record
    return accepted, rejected, backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Scrape the five study/practice pages (default).",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Discover and scrape the complete English Bible Commentary section.",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="In --full mode, report sitemap candidates without changing outputs.",
    )
    parser.add_argument(
        "--max-new-pages",
        type=int,
        default=None,
        help="Stop after this many new page attempts; the next run resumes.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds between article requests; minimum 3.0.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save progress every N attempted pages (default: env or 10).",
    )
    parser.add_argument(
        "--retry-all-rejected",
        action="store_true",
        help="Retry terminal rejections as well as transient request failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    contact_email = normalize_text(os.getenv("TOW_CONTACT_EMAIL", ""))
    if not contact_email or "@" not in contact_email:
        print(
            "ERROR: Add TOW_CONTACT_EMAIL=your_university_email to .env.",
            file=sys.stderr,
        )
        return 2

    try:
        env_delay = float(os.getenv("TOW_REQUEST_DELAY_SECONDS", "3.0"))
        delay = args.delay if args.delay is not None else env_delay
        delay = max(3.0, delay)
    except ValueError:
        print("ERROR: TOW_REQUEST_DELAY_SECONDS must be a number.", file=sys.stderr)
        return 2

    try:
        env_checkpoint = int(os.getenv("TOW_CHECKPOINT_EVERY", "10"))
        checkpoint_every = (
            args.checkpoint_every
            if args.checkpoint_every is not None
            else env_checkpoint
        )
        checkpoint_every = max(1, checkpoint_every)
    except ValueError:
        print("ERROR: TOW_CHECKPOINT_EVERY must be an integer.", file=sys.stderr)
        return 2

    if args.max_new_pages is not None and args.max_new_pages < 1:
        print("ERROR: --max-new-pages must be at least 1.", file=sys.stderr)
        return 2

    mode = "full_bible_commentary" if args.full else "pilot"
    session = build_session(contact_email)
    try:
        robots = load_robots(session)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    discovery: dict[str, Any]
    if mode == "full_bible_commentary":
        print("Discovering English Bible Commentary URLs...")
        try:
            candidates, discovery = discover_full_commentary_pages(session, robots)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Discovered candidates: {len(candidates):,}")
        print(
            "Hierarchy: "
            f"{discovery['parent_page_count']:,} parent pages; "
            f"{discovery['leaf_page_count']:,} leaf pages"
        )
        print("Included paths: /old-testament/ and /new-testament/")
        print("Excluded sitemaps: sitemap_resources.xml and sitemap_thc.xml")
        if args.discover_only:
            print("DISCOVERY COMPLETE; no corpus files were changed.")
            return 0
    else:
        if args.discover_only:
            print("ERROR: --discover-only requires --full.", file=sys.stderr)
            return 2
        candidates = [dict(page) for page in PILOT_PAGES]
        discovery = {
            "candidate_count": len(candidates),
            "source": "explicit five-page allowlist",
        }

    try:
        accepted, rejected, backup_path = prepare_existing_state(mode, candidates)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pending_before = pending_urls(
        candidates,
        accepted,
        rejected,
        retry_all_rejected=args.retry_all_rejected,
    )
    run_started_at = utc_now()
    print(f"Mode: {mode}")
    print(f"Candidates: {len(candidates):,}")
    print(f"Already accepted: {len(accepted):,}")
    print(f"Pending before this run: {len(pending_before):,}")
    print(f"Article delay: {delay:.1f} seconds")
    print(f"Checkpoint interval: {checkpoint_every} pages")
    if backup_path:
        print(f"Pilot outputs backed up before full-corpus conversion: {backup_path}")

    processed_this_run = 0
    page_requests_this_run = 0
    last_request_started = 0.0

    save_state(
        mode=mode,
        status="in_progress",
        candidates=candidates,
        accepted=accepted,
        rejected=rejected,
        discovery=discovery,
        delay=delay,
        checkpoint_every=checkpoint_every,
        run_started_at=run_started_at,
        processed_this_run=processed_this_run,
        backup_path=backup_path,
        retry_all_rejected=args.retry_all_rejected,
    )

    try:
        for page_index, page in enumerate(candidates, start=1):
            key = normalize_url(page["url"])
            if key in accepted:
                continue
            old_rejection = rejected.get(key)
            if (
                old_rejection is not None
                and bool(old_rejection.get("terminal"))
                and not args.retry_all_rejected
            ):
                continue
            if (
                args.max_new_pages is not None
                and processed_this_run >= args.max_new_pages
            ):
                break

            if page_requests_this_run > 0:
                elapsed = time.monotonic() - last_request_started
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            last_request_started = time.monotonic()
            page_requests_this_run += 1

            print(
                f"[{page_index}/{len(candidates)}] "
                f"{page.get('passage') or page['url']} ... ",
                end="",
                flush=True,
            )
            record, rejection = scrape_page(
                session,
                page,
                robots,
                require_bible_commentary=(mode == "full_bible_commentary"),
            )
            processed_this_run += 1

            if rejection is not None:
                rejected[key] = rejection
                print(f"REJECTED ({rejection['reason']})")
            else:
                assert record is not None
                duplicate = next(
                    (
                        existing
                        for existing_key, existing in accepted.items()
                        if existing_key != key
                        and existing.get("text_sha256") == record["text_sha256"]
                    ),
                    None,
                )
                if duplicate is not None:
                    rejection = rejection_record(
                        page,
                        "duplicate_clean_text",
                        status_code=record["http_status"],
                        final_url=record["final_url"],
                        detail=f"Duplicates {duplicate.get('input_url', '')}",
                    )
                    rejected[key] = rejection
                    print("REJECTED (duplicate_clean_text)")
                else:
                    accepted[key] = record
                    rejected.pop(key, None)
                    print(
                        f"ACCEPTED ({record['character_count']:,} chars; "
                        f"{record['license']})"
                    )

            if processed_this_run % checkpoint_every == 0:
                manifest = save_state(
                    mode=mode,
                    status="in_progress",
                    candidates=candidates,
                    accepted=accepted,
                    rejected=rejected,
                    discovery=discovery,
                    delay=delay,
                    checkpoint_every=checkpoint_every,
                    run_started_at=run_started_at,
                    processed_this_run=processed_this_run,
                    backup_path=backup_path,
                    retry_all_rejected=args.retry_all_rejected,
                )
                print(
                    "  CHECKPOINT: "
                    f"accepted={manifest['accepted_count']:,}, "
                    f"rejected={manifest['rejected_count']:,}, "
                    f"pending={manifest['pending_count']:,}"
                )
    except KeyboardInterrupt:
        manifest = save_state(
            mode=mode,
            status="interrupted",
            candidates=candidates,
            accepted=accepted,
            rejected=rejected,
            discovery=discovery,
            delay=delay,
            checkpoint_every=checkpoint_every,
            run_started_at=run_started_at,
            processed_this_run=processed_this_run,
            backup_path=backup_path,
            retry_all_rejected=args.retry_all_rejected,
        )
        print()
        print("INTERRUPTED: progress was saved. Rerun the same command to resume.")
        print(f"Pending: {manifest['pending_count']:,}")
        return 130

    remaining = pending_urls(
        candidates,
        accepted,
        rejected,
        retry_all_rejected=args.retry_all_rejected,
    )
    if remaining:
        final_status = "partial"
    elif rejected:
        final_status = "complete_with_rejections"
    else:
        final_status = "complete"

    manifest = save_state(
        mode=mode,
        status=final_status,
        candidates=candidates,
        accepted=accepted,
        rejected=rejected,
        discovery=discovery,
        delay=delay,
        checkpoint_every=checkpoint_every,
        run_started_at=run_started_at,
        processed_this_run=processed_this_run,
        backup_path=backup_path,
        retry_all_rejected=args.retry_all_rejected,
    )

    print()
    print(f"Status: {manifest['status']}")
    print(f"Accepted: {manifest['accepted_count']:,}")
    print(f"Rejected: {manifest['rejected_count']:,}")
    print(f"Pending: {manifest['pending_count']:,}")
    print(f"Pages: {PAGES_PATH}")
    print(f"Rejected pages: {REJECTED_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")

    if manifest["pending_count"]:
        print("PARTIAL CRAWL SAVED: rerun with --full to continue.")
        return 0
    print("CRAWL COMPLETE: review rejected pages and audit accepted samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
