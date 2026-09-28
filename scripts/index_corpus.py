"""Create a resumable Gemini or Ollama embedding index from a JSONL corpus.

Examples, run from the oTree project directory:

    python -m scripts.index_corpus --smoke-test
    python -m scripts.index_corpus
    python -m scripts.index_corpus \
        --corpus-file corpus/processed/development/five_passages.jsonl \
        --index-dir corpus/processed/development/index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
try:
    from dotenv import load_dotenv
except ImportError:  # Environment variables still work without python-dotenv.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

from llm_rag.embeddings import embed_texts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "corpus" / "processed" / "index"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OLLAMA_API_BASE_URL = "http://localhost:11434"


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not record.get("chunk_id") or not str(record.get("text", "")).strip():
                raise ValueError(f"Invalid corpus record on line {line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"Corpus contains no records: {path}")
    return records


def document_text(record: dict) -> str:
    """Build the exact text embedded for every corpus chunk.

    All corpus records are indexed. Passage and book tags are included so
    book-level commentary and Bible Odyssey pages remain discoverable even
    when they are not tagged to one exact verse.
    """
    fields = [
        f"Title: {record.get('title', '')}",
        f"Source: {record.get('source_name', '')}",
        f"Source ID: {record.get('source_id', '')}",
        f"Source type: {record.get('source_type', '')}",
    ]
    if record.get("book"):
        reference = str(record["book"])
        if record.get("chapter") is not None:
            reference += f" {record['chapter']}"
            if record.get("verse_start") is not None:
                reference += f":{record['verse_start']}"
                if record.get("verse_end") not in (None, record.get("verse_start")):
                    reference += f"-{record['verse_end']}"
        fields.append(f"Biblical scope: {reference}")

    for key, label in (
        ("book_tags", "Book tags"),
        ("passage_tags", "Passage tags"),
        ("themes", "Themes"),
        ("author", "Author"),
    ):
        value = record.get(key)
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if str(item).strip())
        if value not in (None, "", []):
            fields.append(f"{label}: {value}")

    url = record.get("canonical_url") or record.get("url")
    if url:
        fields.append(f"URL: {url}")
    if record.get("license"):
        fields.append(f"Rights/licence: {record['license']}")
    fields.append(f"Content:\n{record['text']}")
    return "\n".join(fields)


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def get_vectors(
    texts: list[str],
    provider: str,
    model: str,
    dimension: int,
    max_retries: int,
    timeout_seconds: float,
    api_key: str = "",
    ollama_base_url: str = OLLAMA_API_BASE_URL,
) -> np.ndarray:
    matrix = embed_texts(
        texts,
        provider=provider,
        model=model,
        dimension=dimension,
        task="document",
        gemini_api_key=api_key,
        gemini_base_url=GEMINI_API_BASE_URL,
        ollama_base_url=ollama_base_url,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def smoke_test(
    records: list[dict],
    provider: str,
    model: str,
    dimension: int,
    max_retries: int,
    timeout_seconds: float,
    api_key: str = "",
    ollama_base_url: str = OLLAMA_API_BASE_URL,
) -> None:
    sample = records[:2]
    vectors = get_vectors(
        [document_text(record) for record in sample],
        provider,
        model,
        dimension,
        max_retries,
        timeout_seconds,
        api_key,
        ollama_base_url,
    )
    print("SMOKE TEST PASSED")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Documents embedded: {len(sample)}")
    print(f"Embedding shape: {vectors.shape}")
    print("No index files were changed.")


def new_progress(
    corpus_hash: str,
    total: int,
    provider: str,
    model: str,
    dimension: int,
) -> dict:
    return {
        "status": "in_progress",
        "corpus_sha256": corpus_hash,
        "total_chunks": total,
        "completed_chunks": 0,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "task_type": "RETRIEVAL_DOCUMENT",
        "normalized": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_progress(
    progress: dict,
    corpus_hash: str,
    total: int,
    provider: str,
    model: str,
    dimension: int,
) -> None:
    expected = {
        "corpus_sha256": corpus_hash,
        "total_chunks": total,
        "embedding_provider": provider,
        "embedding_model": model,
        "embedding_dimension": dimension,
    }
    mismatches = {}
    for key, value in expected.items():
        existing = (
            progress.get(key, "gemini")
            if key == "embedding_provider"
            else progress.get(key)
        )
        if existing != value:
            mismatches[key] = (existing, value)
    if mismatches:
        details = ", ".join(
            f"{key}: existing={old!r}, current={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise RuntimeError(
            "Existing checkpoint does not match the corpus/settings. "
            f"{details}. Use --restart only to intentionally replace this index."
        )


def build_index(
    records: list[dict],
    corpus_file: Path,
    output_dir: Path,
    provider: str,
    model: str,
    dimension: int,
    batch_size: int,
    max_retries: int,
    timeout_seconds: float,
    restart: bool,
    api_key: str = "",
    ollama_base_url: str = OLLAMA_API_BASE_URL,
) -> None:
    embeddings_file = output_dir / "embeddings.npy"
    metadata_file = output_dir / "chunk_metadata.jsonl"
    config_file = output_dir / "index_config.json"
    progress_file = output_dir / "embedding_progress.json"
    index_files = (embeddings_file, metadata_file, config_file, progress_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = sha256_file(corpus_file)
    total = len(records)

    if restart:
        for path in index_files:
            if path.exists():
                path.unlink()

    if progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        validate_progress(progress, corpus_hash, total, provider, model, dimension)
        completed = int(progress.get("completed_chunks", 0))
        if progress.get("status") == "complete" and completed == total:
            print(f"Index is already complete: {output_dir}")
            return
        if not embeddings_file.exists() or not metadata_file.exists():
            raise RuntimeError(
                "A progress file exists, but an index file is missing. "
                "Use --restart only to rebuild intentionally."
            )
        vectors = np.lib.format.open_memmap(embeddings_file, mode="r+")
        if vectors.shape != (total, dimension):
            raise RuntimeError(
                f"Existing embeddings have shape {vectors.shape}; "
                f"expected {(total, dimension)}"
            )
        print(f"Resuming from chunk {completed:,} of {total:,}")
    else:
        existing = [path for path in index_files if path.exists()]
        if existing:
            raise RuntimeError(
                "Index files already exist without a usable checkpoint: "
                + ", ".join(str(path) for path in existing)
                + ". Use --restart only to replace them intentionally."
            )
        progress = new_progress(corpus_hash, total, provider, model, dimension)
        completed = 0
        vectors = np.lib.format.open_memmap(
            embeddings_file,
            mode="w+",
            dtype=np.float32,
            shape=(total, dimension),
        )
        shutil.copyfile(corpus_file, metadata_file)
        atomic_write_json(progress_file, progress)

    started = time.monotonic()
    try:
        for start in range(completed, total, batch_size):
            end = min(start + batch_size, total)
            batch_vectors = get_vectors(
                [document_text(record) for record in records[start:end]],
                provider,
                model,
                dimension,
                max_retries,
                timeout_seconds,
                api_key,
                ollama_base_url,
            )
            vectors[start:end] = batch_vectors
            vectors.flush()
            progress["completed_chunks"] = end
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_json(progress_file, progress)

            elapsed = max(time.monotonic() - started, 0.001)
            rate = (end - completed) / elapsed
            remaining = total - end
            eta_minutes = remaining / rate / 60 if rate else math.inf
            print(
                f"Embedded {end:,}/{total:,} ({end / total:.1%}); "
                f"estimated remaining {eta_minutes:.1f} min"
            )
    except KeyboardInterrupt:
        print("\nIndexing interrupted safely. Rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)

    progress["status"] = "complete"
    progress["completed_chunks"] = total
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(progress_file, progress)
    config = {
        **progress,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_file": display_path(corpus_file),
        "embeddings_file": display_path(embeddings_file),
        "metadata_file": display_path(metadata_file),
        "numpy_dtype": "float32",
        "similarity": "cosine via normalized dot product",
        "chunks_by_source": dict(
            sorted(Counter(str(record.get("source_id", "")) for record in records).items())
        ),
        "document_template": (
            "Title + source/provenance + exact biblical scope + book/passage tags "
            "+ themes/author + URL/licence + content"
        ),
    }
    atomic_write_json(config_file, config)
    print("INDEXING COMPLETE")
    print(f"Embeddings: {embeddings_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Configuration: {config_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=DEFAULT_CORPUS_FILE,
        help="JSONL corpus to embed (default: full normalized corpus).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory in which index files are stored.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Embed only two chunks without creating or changing an index.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Intentionally replace the four index files in --index-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    corpus_file = project_path(args.corpus_file)
    output_dir = project_path(args.index_dir)
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().casefold()
    if provider not in {"gemini", "ollama"}:
        raise SystemExit(
            "ERROR: EMBEDDING_PROVIDER must be either 'gemini' or 'ollama'"
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if provider == "gemini" and not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY is missing from .env")
    default_model = (
        "nomic-embed-text" if provider == "ollama" else "gemini-embedding-001"
    )
    model = os.getenv("EMBEDDING_MODEL", default_model).strip()
    dimension = int(os.getenv("EMBEDDING_DIM", "768"))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))
    timeout_seconds = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120"))
    ollama_base_url = os.getenv(
        "OLLAMA_API_BASE_URL",
        OLLAMA_API_BASE_URL,
    ).strip().rstrip("/")

    if provider == "gemini" and dimension not in {128, 768, 1536, 3072}:
        raise SystemExit(
            "ERROR: EMBEDDING_DIM should be 128, 768, 1536, or 3072. "
            "Use 768 for this project."
        )
    if dimension < 1:
        raise SystemExit("ERROR: EMBEDDING_DIM must be at least 1")
    if batch_size < 1:
        raise SystemExit("ERROR: EMBEDDING_BATCH_SIZE must be at least 1")
    if timeout_seconds <= 0:
        raise SystemExit("ERROR: GEMINI_TIMEOUT_SECONDS must be greater than zero")
    if not corpus_file.exists():
        raise SystemExit(f"ERROR: corpus file not found: {corpus_file}")

    records = read_jsonl(corpus_file)
    print(f"Corpus file: {corpus_file}")
    print(f"Corpus chunks: {len(records):,}")
    print("Corpus chunks by source:")
    for source_id, count in sorted(
        Counter(str(record.get("source_id", "")) for record in records).items()
    ):
        print(f"  {source_id}: {count:,}")
    print(f"Index directory: {output_dir}")
    print(f"Embedding provider: {provider}")
    print(f"Embedding model: {model}")
    print(f"Embedding dimension: {dimension}")
    print(f"Batch size: {batch_size}")

    if args.smoke_test:
        smoke_test(
            records,
            provider,
            model,
            dimension,
            max_retries,
            timeout_seconds,
            api_key,
            ollama_base_url,
        )
        return
    build_index(
        records,
        corpus_file,
        output_dir,
        provider,
        model,
        dimension,
        batch_size,
        max_retries,
        timeout_seconds,
        args.restart,
        api_key,
        ollama_base_url,
    )


if __name__ == "__main__":
    main()
