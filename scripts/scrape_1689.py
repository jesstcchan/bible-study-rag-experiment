"""Controlled scraper for the 1689 Baptist Confession website.

This scraper uses an explicit page list.  It does not discover or crawl the
whole website.  It saves page-level provenance so the historical confession,
the website transcription, and any modern editorial material can be reviewed
separately before indexing.

Run from the project root:

    python -m scripts.scrape_1689 --limit 1
    python -m scripts.scrape_1689

Before running, set a contact email in .env or the environment:

    SCRAPER_CONTACT_EMAIL=your.email@example.org

The output is written to:

    corpus/raw/confessions/1689/pages.jsonl
    corpus/raw/confessions/1689/crawl_manifest.json
    corpus/raw/confessions/1689/raw_html/

The website transcription's copyright/licence status is deliberately recorded
as unverified here.  Do not add the output to the final RAG index until the
exact website terms and edition/transcription status have been documented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "corpus" / "raw" / "confessions" / "1689"
PAGES_FILE = OUTPUT_DIR / "pages.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "crawl_manifest.json"
SMOKE_PAGES_FILE = OUTPUT_DIR / "pages.smoke.jsonl"
SMOKE_MANIFEST_FILE = OUTPUT_DIR / "crawl_manifest.smoke.json"
RAW_HTML_DIR = OUTPUT_DIR / "raw_html"

BASE_URL = "https://www.the1689confession.com"
INTRO_URL = f"{BASE_URL}/1689/introduction"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
ALLOWED_HOSTS = {"www.the1689confession.com", "the1689confession.com"}
MAX_HTML_BYTES = 8_000_000
MIN_DELAY_SECONDS = 1.0
SCRAPER_VERSION = "1.0.0"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", value).strip()


def canonical_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"URL is outside the approved 1689 host: {value}")
    path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/")
    return f"https://www.the1689confession.com{path or '/'}"


def page_urls(include_signatories: bool) -> list[str]:
    urls = [INTRO_URL]
    urls.extend(f"{BASE_URL}/1689/chapter-{number}" for number in range(1, 33))
    if include_signatories:
        urls.append(f"{BASE_URL}/1689/signatories")
    return urls


def load_robots(user_agent: str) -> RobotFileParser:
    robots = RobotFileParser()
    robots.set_url(ROBOTS_URL)
    try:
        robots.read()
    except Exception as error:  # pragma: no cover - depends on network state
        raise RuntimeError(f"Could not read {ROBOTS_URL}: {error}") from error
    if not robots.can_fetch(user_agent, INTRO_URL):
        raise RuntimeError("robots.txt does not permit the requested 1689 pages")
    return robots


def remove_noise(root: BeautifulSoup) -> None:
    selectors = (
        "script", "style", "noscript", "nav", "header", "footer", "aside",
        "form", "iframe", "svg", "canvas", "button", "input", "select",
        "textarea", ".cookie-banner", ".cookie-consent", ".share-buttons",
        ".social-share", ".breadcrumb", ".pagination", ".related-content",
        ".sidebar", ".comments", ".comment", ".author", ".site-footer",
        ".site-header",
    )
    for element in reversed(root.select(", ".join(selectors))):
        if isinstance(element, Tag) and element.parent is not None:
            element.decompose()

    # Remove common layout blocks when their class or id clearly indicates
    # navigation, advertising, sharing, or unrelated site chrome.
    noise_words = (
        "cookie", "consent", "share", "social", "breadcrumb", "pagination",
        "related", "sidebar", "advert", "newsletter", "search", "menu",
        "navigation",
    )
    for element in list(root.find_all(True)):
        if not isinstance(element, Tag) or element.parent is None:
            continue

        identity = " ".join(
            [
                str(element.get("id", "")),
                " ".join(str(value) for value in element.get("class", [])),
            ]
        ).casefold()

        if any(word in identity for word in noise_words):
            element.decompose()

def cleaned_text(candidate: Tag | BeautifulSoup) -> str:
    fragment = BeautifulSoup(str(candidate), "html.parser")
    remove_noise(fragment)

    blocks: list[str] = []
    seen: set[str] = set()
    for element in fragment.select("h1, h2, h3, h4, h5, p, li, blockquote"):
        text = normalize_text(element.get_text(" ", strip=True))
        if not text or text in seen:
            continue
        seen.add(text)
        blocks.append(text)

    if blocks:
        return "\n\n".join(blocks)
    return normalize_text(fragment.get_text(" ", strip=True))


def extract_content(html_bytes: bytes, requested_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html_bytes, "html.parser")
    title_element = soup.select_one("h1") or soup.select_one("title")
    title = normalize_text(title_element.get_text(" ", strip=True)) if title_element else ""

    candidates: list[Tag | BeautifulSoup] = []
    candidates.extend(soup.select("article, main, [role='main'], .entry-content, .post-content"))
    if soup.body is not None:
        candidates.append(soup.body)
    if not candidates:
        candidates.append(soup)

    scored: list[tuple[int, Tag | BeautifulSoup, str]] = []
    for candidate in candidates:
        text = cleaned_text(candidate)
        score = len(text)
        if "judicial and impartial reader" in text.casefold():
            score += 100_000
        if "chapter" in text.casefold():
            score += 1_000
        scored.append((score, candidate, text))

    _, _, body_text = max(scored, key=lambda item: item[0])
    if len(body_text) < 200:
        raise ValueError(f"Extracted text is unexpectedly short: {requested_url}")

    canonical = soup.select_one("link[rel='canonical']")
    canonical_value = (
        canonical_url(str(canonical.get("href")))
        if isinstance(canonical, Tag) and canonical.get("href")
        else canonical_url(requested_url)
    )
    return {
        "title": title,
        "text": body_text,
        "canonical_url": canonical_value,
    }


def fetch_page(
    session: requests.Session,
    robots: RobotFileParser,
    url: str,
    user_agent: str,
    timeout: float,
) -> tuple[bytes, str]:
    requested = canonical_url(url)
    if not robots.can_fetch(user_agent, requested):
        raise RuntimeError(f"robots.txt does not permit: {requested}")

    response = session.get(
        requested,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en",
        },
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    final_url = canonical_url(response.url)
    if not robots.can_fetch(user_agent, final_url):
        raise RuntimeError(f"robots.txt does not permit redirect target: {final_url}")
    content_type = response.headers.get("Content-Type", "").casefold()
    if "text/html" not in content_type:
        raise ValueError(f"Unexpected Content-Type for {final_url}: {content_type}")
    if len(response.content) > MAX_HTML_BYTES:
        raise ValueError(f"HTML response exceeds {MAX_HTML_BYTES:,} bytes")
    return response.content, final_url


def page_metadata(url: str) -> dict[str, object]:
    path = urlparse(url).path.rstrip("/")
    match = re.search(r"/chapter-(\d+)$", path)
    if match:
        chapter = int(match.group(1))
        return {
            "source_page_id": f"1689_chapter_{chapter:02d}",
            "section_type": "chapter",
            "chapter_number": chapter,
        }
    if path.endswith("/introduction"):
        return {
            "source_page_id": "1689_introduction",
            "section_type": "introduction",
            "chapter_number": None,
        }
    return {
        "source_page_id": "1689_signatories",
        "section_type": "signatories",
        "chapter_number": None,
    }


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Fetch only the first N pages")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--include-signatories", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.delay < MIN_DELAY_SECONDS:
        raise SystemExit(f"--delay must be at least {MIN_DELAY_SECONDS:.1f} seconds")

    load_dotenv(PROJECT_ROOT / ".env")
    contact_email = os.getenv("SCRAPER_CONTACT_EMAIL", "").strip()
    if not contact_email or "@" not in contact_email:
        raise SystemExit(
            "Add SCRAPER_CONTACT_EMAIL=your.email@example.org to .env before scraping."
        )
    user_agent = f"bible-study-rag-experiment/{SCRAPER_VERSION} (+mailto:{contact_email})"

    urls = page_urls(args.include_signatories)
    if args.limit is not None:
        urls = urls[: args.limit]

    pages_path = SMOKE_PAGES_FILE if args.limit is not None else PAGES_FILE
    manifest_path = (
        SMOKE_MANIFEST_FILE if args.limit is not None else MANIFEST_FILE
    )

    robots = load_robots(user_agent)
    session = requests.Session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    seen_canonical: set[str] = set()
    started_at = utc_now()

    for index, requested_url in enumerate(urls, start=1):
        if index > 1:
            time.sleep(args.delay)
        entry: dict[str, object] = {
            "requested_url": requested_url,
            "status": "failed",
            "accessed_at": utc_now(),
        }
        try:
            html_bytes, final_url = fetch_page(
                session, robots, requested_url, user_agent, args.timeout
            )
            html_hash = sha256_bytes(html_bytes)
            raw_path = RAW_HTML_DIR / f"{html_hash[:20]}.html"
            if not raw_path.exists():
                raw_path.write_bytes(html_bytes)

            extracted = extract_content(html_bytes, final_url)
            canonical = str(extracted["canonical_url"])
            if canonical in seen_canonical:
                raise ValueError(f"Duplicate canonical URL: {canonical}")
            seen_canonical.add(canonical)

            metadata = page_metadata(canonical)
            record = {
                "id": metadata["source_page_id"],
                "source_id": "CONF_BAPTIST_1689",
                "source_name": "The 1689 Baptist Confession of Faith",
                "source_type": "confessional_document",
                "tradition_scope": "Baptist",
                "claim_status": "denominational_confession",
                "global_context": True,
                "title": extracted["title"],
                "text": extracted["text"],
                "url": canonical,
                "canonical_url": canonical,
                "license": "Unverified website transcription; review before indexing",
                "license_status": "review_required",
                "accessed_at": entry["accessed_at"],
                "scraper_version": SCRAPER_VERSION,
                "original_html_sha256": html_hash,
                "raw_html_file": raw_path.relative_to(PROJECT_ROOT).as_posix(),
                **metadata,
            }
            records.append(record)
            entry.update(
                {
                    "status": "success",
                    "final_url": final_url,
                    "canonical_url": canonical,
                    "source_page_id": metadata["source_page_id"],
                    "original_html_sha256": html_hash,
                }
            )
            print(f"[{index}/{len(urls)}] OK: {metadata['source_page_id']}")
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{len(urls)}] FAILED: {requested_url}: {error}", file=sys.stderr)
        manifest_entries.append(entry)

    failures = [entry for entry in manifest_entries if entry["status"] != "success"]
    manifest = {
        "scraper_version": SCRAPER_VERSION,
        "source_id": "CONF_BAPTIST_1689",
        "base_url": BASE_URL,
        "mode": "limited" if args.limit is not None else "complete",
        "requested_pages": len(urls),
        "successful_pages": len(records),
        "failed_pages": len(failures),
        "started_at": started_at,
        "completed_at": utc_now(),
        "robots_url": ROBOTS_URL,
        "license_status": "review_required",
        "entries": manifest_entries,
    }

    if failures:
        write_json(manifest_path, manifest)
        raise SystemExit(
            f"Scrape failed for {len(failures)} page(s); {pages_path.name} was not "
            "replaced. Review the manifest and rerun."
        )

    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    temporary = pages_path.with_name(f".{pages_path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(pages_path)
    write_json(manifest_path, manifest)
    print(f"Wrote {len(records)} pages to {pages_path}")
    print(f"Manifest: {manifest_path}")
    print("LICENSE REVIEW REQUIRED before indexing this source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
