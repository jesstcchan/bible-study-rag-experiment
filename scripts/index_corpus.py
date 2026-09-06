"""Create a resumable Gemini embedding index from a JSONL corpus.

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
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_FILE = PROJECT_ROOT / "corpus" / "processed" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "corpus" / "processed" / "index"


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
    fields = [
        f"Title: {record.get('title', '')}",
        f"Source: {record.get('source_name', '')}",
    ]
    if record.get("book") and record.get("chapter"):
        reference = f"{record['book']} {record['chapter']}"
        if record.get("verse_start") is not None:
            reference += f":{record['verse_start']}"
            if record.get("verse_end") not in (None, record.get("verse_start")):
                reference += f"-{record['verse_end']}"
        fields.append(f"Biblical reference: {reference}")
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
    client: genai.Client,
    texts: list[str],
    model: str,
    dimension: int,
    max_retries: int,
) -> np.ndarray:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=dimension,
                ),
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Gemini returned {len(embeddings)} vectors for {len(texts)} documents"
                )
            matrix = np.asarray([item.values for item in embeddings], dtype=np.float32)
            if matrix.shape != (len(texts), dimension):
                raise RuntimeError(
                    f"Unexpected embedding shape {matrix.shape}; "
                    f"expected {(len(texts), dimension)}"
                )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise RuntimeError("Gemini returned a zero-length embedding")
            return matrix / norms
        except ValueError:
            # Configuration errors cannot be repaired by retrying.
            raise
        except Exception as error:
            if attempt >= max_retries:
                raise
            delay = min(60.0, (2**attempt) * 2.0 + random.random())
            print(
                f"API attempt {attempt + 1} failed: {type(error).__name__}: {error}\n"
                f"Retrying in {delay:.1f} seconds...",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state")


def smoke_test(
    client: genai.Client,
    records: list[dict],
    model: str,
    dimension: int,
    max_retries: int,
) -> None:
    sample = records[:2]
    vectors = get_vectors(
        client,
        [document_text(record) for record in sample],
        model,
        dimension,
        max_retries,
    )
    print("SMOKE TEST PASSED")
    print(f"Model: {model}")
    print(f"Documents embedded: {len(sample)}")
    print(f"Embedding shape: {vectors.shape}")
    print("No index files were changed.")


def new_progress(corpus_hash: str, total: int, model: str, dimension: int) -> dict:
    return {
        "status": "in_progress",
        "corpus_sha256": corpus_hash,
        "total_chunks": total,
        "completed_chunks": 0,
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
    model: str,
    dimension: int,
) -> None:
    expected = {
        "corpus_sha256": corpus_hash,
        "total_chunks": total,
        "embedding_model": model,
        "embedding_dimension": dimension,
    }
    mismatches = {
        key: (progress.get(key), value)
        for key, value in expected.items()
        if progress.get(key) != value
    }
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
    client: genai.Client,
    records: list[dict],
    corpus_file: Path,
    output_dir: Path,
    model: str,
    dimension: int,
    batch_size: int,
    max_retries: int,
    restart: bool,
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
        validate_progress(progress, corpus_hash, total, model, dimension)
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
        progress = new_progress(corpus_hash, total, model, dimension)
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
                client,
                [document_text(record) for record in records[start:end]],
                model,
                dimension,
                max_retries,
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
        "document_template": "Title + source + biblical reference + content",
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
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY is missing from .env")
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
    dimension = int(os.getenv("EMBEDDING_DIM", "768"))
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "6"))

    if dimension not in {128, 768, 1536, 3072}:
        raise SystemExit(
            "ERROR: EMBEDDING_DIM should be 128, 768, 1536, or 3072. "
            "Use 768 for this project."
        )
    if batch_size < 1:
        raise SystemExit("ERROR: EMBEDDING_BATCH_SIZE must be at least 1")
    if not corpus_file.exists():
        raise SystemExit(f"ERROR: corpus file not found: {corpus_file}")

    records = read_jsonl(corpus_file)
    client = genai.Client(api_key=api_key)
    print(f"Corpus file: {corpus_file}")
    print(f"Corpus chunks: {len(records):,}")
    print(f"Index directory: {output_dir}")
    print(f"Embedding model: {model}")
    print(f"Embedding dimension: {dimension}")
    print(f"Batch size: {batch_size}")

    if args.smoke_test:
        smoke_test(client, records, model, dimension, max_retries)
        return
    build_index(
        client,
        records,
        corpus_file,
        output_dir,
        model,
        dimension,
        batch_size,
        max_retries,
        args.restart,
    )


if __name__ == "__main__":
    main()
