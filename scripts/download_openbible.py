"""Download the official OpenBible.info cross-reference dataset.

This source should be downloaded from the dataset link, not scraped one verse
page at a time. Existing files are preserved unless --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables still work without python-dotenv.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "corpus" / "raw" / "openbible"
DATA_FILE = RAW_DIR / "cross_references.txt"
ZIP_FILE = RAW_DIR / "cross-references.zip"
MANIFEST_FILE = RAW_DIR / "download_manifest.json"
DOWNLOAD_URL = "https://a.openbible.info/data/cross-references.zip"
SOURCE_PAGE = "https://www.openbible.info/labs/cross-references/"
MAX_DOWNLOAD_BYTES = 20_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(value)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cross_references.txt intentionally.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0:
        raise SystemExit("ERROR: --timeout must be greater than zero")
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0 and not args.force:
        print(f"OpenBible data already exists: {DATA_FILE}")
        print("Nothing changed. Use --force only for an intentional new snapshot.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    user_agent = (
        os.getenv("OPENBIBLE_USER_AGENT", "").strip()
        or os.getenv("BIBLE_ODYSSEY_USER_AGENT", "").strip()
    )
    if not user_agent:
        raise SystemExit(
            "ERROR: Add OPENBIBLE_USER_AGENT to .env with your real research "
            "project and TUM contact email."
        )

    request = Request(
        DOWNLOAD_URL,
        headers={"User-Agent": user_agent, "Accept": "application/zip"},
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            if int(response.status) != 200:
                raise RuntimeError(f"HTTP {response.status}")
            archive_bytes = response.read(MAX_DOWNLOAD_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise SystemExit(f"ERROR: OpenBible download failed: {error}") from error

    if len(archive_bytes) > MAX_DOWNLOAD_BYTES:
        raise SystemExit("ERROR: downloaded ZIP exceeds the safety limit")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            candidates = []
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe ZIP member: {member.filename}")
                if path.name in {"cross_references.txt", "cross-references.txt"}:
                    candidates.append(member)
            if len(candidates) != 1:
                raise ValueError(
                    "expected exactly one cross-reference text file; "
                    f"found {len(candidates)}"
                )
            text_bytes = archive.read(candidates[0])
    except (zipfile.BadZipFile, ValueError) as error:
        raise SystemExit(f"ERROR: invalid OpenBible ZIP: {error}") from error

    if not text_bytes.startswith(b"From Verse"):
        raise SystemExit("ERROR: unexpected OpenBible text-file header")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(ZIP_FILE, archive_bytes)
    atomic_write(DATA_FILE, text_bytes)
    manifest = {
        "downloaded_at": utc_now(),
        "download_url": DOWNLOAD_URL,
        "source_page": SOURCE_PAGE,
        "license": "CC BY 4.0",
        "zip_sha256": sha256_bytes(archive_bytes),
        "data_sha256": sha256_bytes(text_bytes),
        "data_file": DATA_FILE.relative_to(PROJECT_ROOT).as_posix(),
    }
    atomic_write(
        MANIFEST_FILE,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"Wrote {DATA_FILE}")
    print(f"SHA-256: {manifest['data_sha256']}")


if __name__ == "__main__":
    main()
