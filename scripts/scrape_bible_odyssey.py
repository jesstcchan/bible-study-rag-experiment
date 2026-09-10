"""Controlled, whitelist-only Bible Odyssey scraper.

Run this module only after written permission covers downloading, local
storage, text extraction, embedding, use in the RAG system, and display of
source excerpts. The scraper does not discover links and never attempts to
bypass an HTTP access restriction.

Input:
    corpus/raw/bible_odyssey/whitelist.csv

Required CSV columns:
    url,enabled

Optional CSV columns:
    books,passages,notes

Separate multiple curator-verified tags with semicolons. A limited smoke run
writes pages.smoke.jsonl and raw_manifest.smoke.json; it never replaces the
complete pages.jsonl corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables still work without python-dotenv.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "corpus" / "raw" / "bible_odyssey"
WHITELIST_FILE = RAW_DIR / "whitelist.csv"
PAGES_FILE = RAW_DIR / "pages.jsonl"
MANIFEST_FILE = RAW_DIR / "raw_manifest.json"
SMOKE_PAGES_FILE = RAW_DIR / "pages.smoke.jsonl"
SMOKE_MANIFEST_FILE = RAW_DIR / "raw_manifest.smoke.json"
RAW_HTML_DIR = RAW_DIR / "raw_html"

SCRAPER_VERSION = "2.0.0"
ALLOWED_HOSTS = {"bibleodyssey.org", "www.bibleodyssey.org"}
ALLOWED_PATH_PREFIXES = ("/articles/", "/dictionary/")
MIN_DELAY_SECONDS = 1.0
MIN_ARTICLE_CHARS = 300
MAX_HTML_BYTES = 8_000_000
TRUE_VALUES = {"1", "true", "yes", "y"}

TERMS_URL = "https://www.bibleodyssey.org/terms-of-use/"
RIGHTS_BASIS = "Written permission; see permission_record"
CONTENT_CLASSES = {
    "entry-content",
    "wp-block-post-content",
    "post-content",
    "article-content",
}
SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "aside",
    "form",
    "figure",
    "blockquote",
}
SKIP_CLASSES = {
    "sidebar",
    "related-content",
    "contributors",
    "sharedaddy",
    "share-buttons",
    "breadcrumb",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
BLOCK_TAGS = {"h2", "h3", "h4", "h5", "p", "li"}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_output_layout() -> None:
    """Repair the zero-byte raw_html placeholder found in the uploaded ZIP."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_HTML_DIR.exists() and not RAW_HTML_DIR.is_dir():
        if RAW_HTML_DIR.stat().st_size != 0:
            raise RuntimeError(
                f"Expected a directory at {RAW_HTML_DIR}, but found a nonempty file. "
                "Move it aside manually before running the scraper."
            )
        backup = RAW_HTML_DIR.with_name("raw_html.invalid-empty-file")
        if backup.exists():
            raise RuntimeError(
                f"Cannot repair {RAW_HTML_DIR}: backup already exists at {backup}"
            )
        RAW_HTML_DIR.replace(backup)
        print(f"Moved invalid empty placeholder to {backup.name}")
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)


def validate_article_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs are permitted: {url}")
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"URL is outside the approved Bible Odyssey host: {url}")
    if not any(parsed.path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        raise ValueError(
            "Only Bible Odyssey /articles/ and /dictionary/ pages are permitted: "
            f"{url}"
        )
    if parsed.username or parsed.password:
        raise ValueError(f"Credentials are not permitted in a URL: {url}")
    return url


def load_whitelist(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Whitelist not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {str(name).strip().lower() for name in (reader.fieldnames or [])}
        if not {"url", "enabled"}.issubset(fieldnames):
            raise ValueError("whitelist.csv must contain 'url' and 'enabled' columns")

        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for line_number, raw_row in enumerate(reader, start=2):
            row = {
                str(key).strip().lower(): str(value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            if row.get("enabled", "").lower() not in TRUE_VALUES:
                continue
            url = validate_article_url(row.get("url", ""))
            if url in seen:
                raise ValueError(
                    f"Duplicate enabled URL in whitelist on line {line_number}: {url}"
                )
            seen.add(url)
            rows.append(
                {
                    "url": url,
                    "books": row.get("books", ""),
                    "passages": row.get("passages", ""),
                    "notes": row.get("notes", ""),
                    "line_number": str(line_number),
                }
            )

    if not rows:
        raise ValueError(
            f"No enabled URLs found in {path}. Set enabled=true only for pages "
            "covered by your written permission and manually verified tags."
        )
    return rows


def split_verified_tags(value: str) -> list[str]:
    tags: list[str] = []
    for raw_tag in value.split(";"):
        tag = normalize_inline_text(raw_tag)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def required_environment() -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    user_agent = os.getenv("BIBLE_ODYSSEY_USER_AGENT", "").strip()
    permission_record = os.getenv("BIBLE_ODYSSEY_PERMISSION_RECORD", "").strip()
    if not user_agent:
        raise RuntimeError(
            "BIBLE_ODYSSEY_USER_AGENT is missing. Add it to the project .env "
            "file with your real TUM contact email."
        )
    if not permission_record:
        raise RuntimeError(
            "BIBLE_ODYSSEY_PERMISSION_RECORD is missing. Do not scrape until "
            "written permission and its date/scope are recorded in .env."
        )
    return user_agent, permission_record


@dataclass(frozen=True)
class FetchResult:
    status: int
    final_url: str
    content_type: str
    body: bytes


def fetch_article(
    url: str,
    user_agent: str,
    timeout_seconds: float,
    retries: int = 3,
) -> FetchResult:
    """Fetch one whitelisted page without bypassing access restrictions."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.status)
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "").lower()
                body = response.read(MAX_HTML_BYTES + 1)
        except HTTPError as error:
            last_error = error
            status = int(error.code)
            if status in {401, 403, 404}:
                raise RuntimeError(
                    f"HTTP {status}; stop and contact Bible Odyssey. "
                    "Do not try to bypass the restriction."
                ) from error
            if status == 429 or 500 <= status <= 599:
                if attempt < retries:
                    retry_after = str(error.headers.get("Retry-After", "")).strip()
                    wait_seconds = (
                        float(retry_after)
                        if retry_after.isdigit()
                        else float(attempt * 3)
                    )
                    time.sleep(min(wait_seconds, 60.0))
                    continue
            raise RuntimeError(f"HTTP {status}") from error
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(float(attempt * 2))
                continue
            break

        validate_article_url(final_url)
        if status != 200:
            raise RuntimeError(f"HTTP {status}")
        if "text/html" not in content_type:
            raise ValueError(f"Unexpected Content-Type: {content_type or 'missing'}")
        if len(body) > MAX_HTML_BYTES:
            raise ValueError(
                f"HTML response exceeds the {MAX_HTML_BYTES:,}-byte safety limit"
            )
        return FetchResult(status, final_url, content_type, body)

    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


class BibleOdysseyHTMLParser(HTMLParser):
    """Extract semantic article fields without relying on fragile CSS paths."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.fallback_depth: int | None = None
        self.content_depth: int | None = None
        self.skip_depth: int | None = None
        self.title_depth: int | None = None
        self.title_parts: list[str] = []
        self.title = ""
        self.active_block_depth: int | None = None
        self.active_block_type = ""
        self.active_block_explicit = False
        self.active_block_parts: list[str] = []
        self.explicit_blocks: list[dict[str, str]] = []
        self.fallback_blocks: list[dict[str, str]] = []
        self.author_depth: int | None = None
        self.author_parts: list[str] = []
        self.authors: list[str] = []
        self.theme_depth: int | None = None
        self.theme_parts: list[str] = []
        self.themes: list[str] = []
        self.bibliography_depth: int | None = None
        self.bibliography_parts: list[str] = []
        self.bibliography_text = ""
        self.json_ld_depth: int | None = None
        self.json_ld_parts: list[str] = []
        self.json_ld_documents: list[str] = []
        self.canonical_url = ""
        self.published_at = ""
        self.summary = ""

    @staticmethod
    def _attributes(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {name.casefold(): str(value or "") for name, value in attrs}

    @staticmethod
    def _classes(attributes: dict[str, str]) -> set[str]:
        return set(attributes.get("class", "").casefold().split())

    @staticmethod
    def _append_unique(values: list[str], value: str) -> None:
        if value and value not in values:
            values.append(value)

    def _finish_block(self) -> None:
        text = normalize_inline_text("".join(self.active_block_parts))
        if text and text != self.title:
            block = {"type": self.active_block_type, "text": text}
            destination = (
                self.explicit_blocks
                if self.active_block_explicit
                else self.fallback_blocks
            )
            if not destination or destination[-1]["text"] != text:
                destination.append(block)
        self.active_block_depth = None
        self.active_block_type = ""
        self.active_block_explicit = False
        self.active_block_parts = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        attributes = self._attributes(attrs)
        classes = self._classes(attributes)
        depth = len(self.stack)

        if tag == "link" and "canonical" in attributes.get("rel", "").split():
            self.canonical_url = attributes.get("href", "").strip()
        if tag == "meta":
            name = (
                attributes.get("name", "") or attributes.get("property", "")
            ).casefold()
            if name in {"description", "og:description"} and not self.summary:
                self.summary = normalize_inline_text(attributes.get("content", ""))
        if tag == "time" and attributes.get("datetime") and not self.published_at:
            self.published_at = normalize_inline_text(attributes["datetime"])
        if tag == "script" and attributes.get("type", "").casefold() == "application/ld+json":
            self.json_ld_depth = depth
            self.json_ld_parts = []

        if tag in {"main", "article"} and self.fallback_depth is None:
            self.fallback_depth = depth
        if classes.intersection(CONTENT_CLASSES) and self.content_depth is None:
            self.content_depth = depth
        in_scope = self.fallback_depth is not None or self.content_depth is not None

        if in_scope and tag == "h1" and self.title_depth is None and not self.title:
            self.title_depth = depth
            self.title_parts = []

        href = attributes.get("href", "")
        if in_scope and tag == "a" and "/author/" in href and self.author_depth is None:
            self.author_depth = depth
            self.author_parts = []
        if in_scope and tag == "a" and "/theme/" in href and self.theme_depth is None:
            self.theme_depth = depth
            self.theme_parts = []

        is_bibliography = (
            attributes.get("id", "").casefold() == "bibliography"
            or "bibliography" in classes
        )
        if in_scope and is_bibliography and self.bibliography_depth is None:
            self.bibliography_depth = depth
            self.bibliography_parts = []

        should_skip = (
            tag in SKIP_TAGS
            or bool(classes.intersection(SKIP_CLASSES))
            or is_bibliography
            or attributes.get("aria-label", "").casefold() == "breadcrumb"
        )
        if in_scope and should_skip and self.skip_depth is None:
            self.skip_depth = depth

        if (
            in_scope
            and self.skip_depth is None
            and tag in BLOCK_TAGS
            and self.active_block_depth is None
        ):
            self.active_block_depth = depth
            self.active_block_type = (
                "heading" if tag in {"h2", "h3", "h4", "h5"} else "paragraph"
            )
            self.active_block_explicit = self.content_depth is not None
            self.active_block_parts = []

        if tag == "br":
            if self.active_block_depth is not None and self.skip_depth is None:
                self.active_block_parts.append("\n")
            if self.bibliography_depth is not None:
                self.bibliography_parts.append("\n")

        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.json_ld_depth is not None:
            self.json_ld_parts.append(data)
        if self.title_depth is not None:
            self.title_parts.append(data)
        if self.active_block_depth is not None and self.skip_depth is None:
            self.active_block_parts.append(data)
        if self.author_depth is not None:
            self.author_parts.append(data)
        if self.theme_depth is not None:
            self.theme_parts.append(data)
        if self.bibliography_depth is not None:
            self.bibliography_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in VOID_TAGS or not self.stack:
            return
        depth = len(self.stack) - 1

        if self.title_depth == depth:
            self.title = normalize_inline_text("".join(self.title_parts))
            self.title_depth = None
            self.title_parts = []
        if self.active_block_depth == depth:
            self._finish_block()
        if self.author_depth == depth:
            self._append_unique(
                self.authors,
                normalize_inline_text("".join(self.author_parts)),
            )
            self.author_depth = None
            self.author_parts = []
        if self.theme_depth == depth:
            self._append_unique(
                self.themes,
                normalize_inline_text("".join(self.theme_parts)),
            )
            self.theme_depth = None
            self.theme_parts = []
        if self.bibliography_depth == depth:
            self.bibliography_text = normalize_inline_text(
                "".join(self.bibliography_parts)
            )
            self.bibliography_depth = None
            self.bibliography_parts = []
        if self.json_ld_depth == depth:
            self.json_ld_documents.append("".join(self.json_ld_parts))
            self.json_ld_depth = None
            self.json_ld_parts = []
        if self.skip_depth == depth:
            self.skip_depth = None
        if self.content_depth == depth:
            self.content_depth = None
        if self.fallback_depth == depth:
            self.fallback_depth = None
        self.stack.pop()

    @property
    def blocks(self) -> list[dict[str, str]]:
        return self.explicit_blocks or self.fallback_blocks


def json_ld_objects(documents: list[str]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for document in documents:
        try:
            value = json.loads(document)
        except json.JSONDecodeError:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                objects.extend(node for node in graph if isinstance(node, dict))
            else:
                objects.append(item)
    return objects


def author_names(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    names: list[str] = []
    for item in values:
        name = (
            normalize_inline_text(str(item.get("name", "")))
            if isinstance(item, dict)
            else normalize_inline_text(str(item or ""))
        )
        if name and name not in names:
            names.append(name)
    return names


def extract_article(
    html_bytes: bytes,
    requested_url: str,
    final_url: str,
    accessed_at: str,
    permission_record: str,
    raw_html_file: str,
) -> dict[str, Any]:
    validate_article_url(final_url)
    parser = BibleOdysseyHTMLParser()
    parser.feed(html_bytes.decode("utf-8", errors="replace"))
    parser.close()

    canonical_url = validate_article_url(
        urljoin(final_url, parser.canonical_url or final_url)
    )
    structured = json_ld_objects(parser.json_ld_documents)
    article_schema = next(
        (
            item
            for item in structured
            if str(item.get("@type", "")).casefold()
            in {"article", "newsarticle", "scholarlyarticle"}
        ),
        {},
    )
    title = normalize_inline_text(
        str(article_schema.get("headline", "")) or parser.title
    )
    if not title:
        raise ValueError("Article title was not found")
    blocks = [block for block in parser.blocks if block["text"] != title]
    article_text = "\n\n".join(block["text"] for block in blocks).strip()
    if len(article_text) < MIN_ARTICLE_CHARS:
        raise ValueError(
            f"Extracted article is unexpectedly short ({len(article_text)} characters)"
        )

    authors = author_names(article_schema.get("author")) or parser.authors
    themes = parser.themes
    published_at = normalize_inline_text(
        str(article_schema.get("datePublished", "")) or parser.published_at
    )
    summary = normalize_inline_text(
        str(article_schema.get("description", "")) or parser.summary
    )

    slug = urlparse(canonical_url).path.rstrip("/").split("/")[-1]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        slug = sha256_text(canonical_url)[:20]

    return {
        "id": slug,
        "title": title,
        "canonical_url": canonical_url,
        "requested_url": requested_url,
        "final_url": final_url,
        "summary": summary,
        "article_text": article_text,
        "blocks": blocks,
        "authors": authors,
        "themes": themes,
        "published_at": published_at or None,
        "accessed_at": accessed_at,
        "content_type": "Bible Odyssey article or dictionary entry",
        "http_status": 200,
        "license": RIGHTS_BASIS,
        "license_url": TERMS_URL,
        "terms_url": TERMS_URL,
        "permission_record": permission_record,
        "scraper_version": SCRAPER_VERSION,
        "html_sha256": sha256_bytes(html_bytes),
        "text_sha256": sha256_text(article_text),
        "raw_html_file": raw_html_file,
        "bibliography_text": parser.bibliography_text or None,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N enabled rows and write separate smoke files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (minimum: 1; default: 2).",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if args.delay < MIN_DELAY_SECONDS:
        raise ValueError(f"--delay must be at least {MIN_DELAY_SECONDS:.0f} second")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    user_agent, permission_record = required_environment()
    whitelist_rows = load_whitelist(WHITELIST_FILE)
    smoke_mode = args.limit is not None
    if smoke_mode:
        whitelist_rows = whitelist_rows[: args.limit]
    pages_file = SMOKE_PAGES_FILE if smoke_mode else PAGES_FILE
    manifest_file = SMOKE_MANIFEST_FILE if smoke_mode else MANIFEST_FILE

    prepare_output_layout()
    started_at = utc_now()
    records: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    canonical_urls: set[str] = set()
    failures: list[str] = []

    for index, row in enumerate(whitelist_rows, start=1):
        if index > 1:
            time.sleep(args.delay)

        requested_url = row["url"]
        accessed_at = utc_now()
        manifest_entry: dict[str, Any] = {
            "requested_url": requested_url,
            "whitelist_line": int(row["line_number"]),
            "book_tags": split_verified_tags(row["books"]),
            "passage_tags": split_verified_tags(row["passages"]),
            "notes": row["notes"],
            "accessed_at": accessed_at,
            "status": "failed",
        }

        try:
            response = fetch_article(
                requested_url,
                user_agent=user_agent,
                timeout_seconds=args.timeout,
            )
            html_sha256 = sha256_bytes(response.body)
            raw_path = RAW_HTML_DIR / f"{html_sha256[:20]}.html"
            if not raw_path.exists():
                atomic_write_bytes(raw_path, response.body)
            raw_relative = raw_path.relative_to(PROJECT_ROOT).as_posix()

            record = extract_article(
                response.body,
                requested_url=requested_url,
                final_url=response.final_url,
                accessed_at=accessed_at,
                permission_record=permission_record,
                raw_html_file=raw_relative,
            )
            record["book_tags"] = split_verified_tags(row["books"])
            record["passage_tags"] = split_verified_tags(row["passages"])
            canonical_url = str(record["canonical_url"])
            if canonical_url in canonical_urls:
                raise ValueError(f"Duplicate canonical URL: {canonical_url}")
            canonical_urls.add(canonical_url)
            records.append(record)

            manifest_entry.update(
                {
                    "status": "success",
                    "http_status": response.status,
                    "final_url": response.final_url,
                    "canonical_url": canonical_url,
                    "page_id": record["id"],
                    "html_sha256": record["html_sha256"],
                    "text_sha256": record["text_sha256"],
                    "raw_html_file": raw_relative,
                }
            )
            print(f"[{index}/{len(whitelist_rows)}] OK: {record['id']}")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            manifest_entry["error"] = message
            failures.append(f"{requested_url} -> {message}")
            print(
                f"[{index}/{len(whitelist_rows)}] FAILED: {requested_url}: {message}",
                file=sys.stderr,
            )
        manifest_entries.append(manifest_entry)

    manifest: dict[str, Any] = {
        "scraper_version": SCRAPER_VERSION,
        "mode": "smoke" if smoke_mode else "complete",
        "started_at": started_at,
        "completed_at": utc_now(),
        "whitelist_file": WHITELIST_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "permission_record": permission_record,
        "requested_pages": len(whitelist_rows),
        "successful_pages": len(records),
        "failed_pages": len(failures),
        "pages_file_written": not failures and bool(records),
        "entries": manifest_entries,
    }
    atomic_write_text(
        manifest_file,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    if failures:
        raise RuntimeError(
            f"Crawl failed; {pages_file.name} was not replaced.\n- "
            + "\n- ".join(failures)
        )
    if not records:
        raise RuntimeError("No Bible Odyssey pages were extracted")

    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(pages_file, payload)
    print(f"Wrote {len(records)} permitted pages to {pages_file}")
    print(f"Manifest: {manifest_file}")
    if smoke_mode:
        print("Smoke output is separate; the complete pages.jsonl was not changed.")


if __name__ == "__main__":
    main()
