"""Query and validate a completed local Gemini embedding index.

Examples, run from the oTree project directory:

    python -m scripts.test_retrieval \
        --index-dir corpus/processed/development/index \
        --test-suite

    python -m scripts.test_retrieval \
        --index-dir corpus/processed/development/index \
        --query "Why was Naaman initially resistant to Elisha's instruction?"
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "corpus" / "processed" / "index"

TEST_CASES = (
    ("2 Kings 5:9–14", "Why was Naaman resistant to Elisha's instruction?", "2 Kings", 5, 9, 14),
    ("Romans 12:1–5", "What does Romans 12:1–5 teach about worship and one body?", "Romans", 12, 1, 5),
    ("1 Samuel 8:4–9", "Why did Israel ask Samuel for a king?", "1 Samuel", 8, 4, 9),
    ("1 Corinthians 8:1–6", "How do knowledge, love, and idols relate in 1 Corinthians 8:1–6?", "1 Corinthians", 8, 1, 6),
    ("Mark 4:35–41", "What does Jesus calming the storm reveal in Mark 4:35–41?", "Mark", 4, 35, 41),
)


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid metadata JSON on line {line_number}: {error}") from error
    return records


def load_index(index_dir: Path) -> tuple[np.ndarray, list[dict], dict]:
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "chunk_metadata.jsonl"
    config_path = index_dir / "index_config.json"
    missing = [
        str(path)
        for path in (embeddings_path, metadata_path, config_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing completed index files: " + ", ".join(missing))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise RuntimeError(f"Index status is not complete: {config.get('status')!r}")
    if config.get("task_type") != "RETRIEVAL_DOCUMENT":
        raise RuntimeError("Index was not created with RETRIEVAL_DOCUMENT embeddings")
    if config.get("normalized") is not True:
        raise RuntimeError("Index vectors are not marked as normalized")

    vectors = np.load(embeddings_path, mmap_mode="r")
    records = read_jsonl(metadata_path)
    expected_shape = (
        int(config["total_chunks"]),
        int(config["embedding_dimension"]),
    )
    if vectors.shape != expected_shape:
        raise RuntimeError(f"Embedding shape {vectors.shape} does not match {expected_shape}")
    if len(records) != vectors.shape[0]:
        raise RuntimeError(
            f"Metadata has {len(records)} rows but embeddings have {vectors.shape[0]} rows"
        )
    return vectors, records, config


def embed_query(
    client: genai.Client,
    query: str,
    model: str,
    dimension: int,
) -> np.ndarray:
    response = client.models.embed_content(
        model=model,
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=dimension,
        ),
    )
    embeddings = response.embeddings or []
    if len(embeddings) != 1:
        raise RuntimeError(f"Gemini returned {len(embeddings)} query vectors; expected 1")
    vector = np.asarray(embeddings[0].values, dtype=np.float32)
    if vector.shape != (dimension,):
        raise RuntimeError(f"Query vector shape {vector.shape}; expected {(dimension,)}")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise RuntimeError("Gemini returned a zero-length query embedding")
    return vector / norm


def retrieve(
    vectors: np.ndarray,
    records: list[dict],
    query_vector: np.ndarray,
    top_k: int,
) -> list[tuple[float, dict]]:
    top_k = min(max(1, top_k), len(records))
    scores = np.asarray(vectors @ query_vector)
    indices = np.argpartition(scores, -top_k)[-top_k:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return [(float(scores[index]), records[int(index)]) for index in indices]


def reference(record: dict) -> str:
    book = record.get("book")
    chapter = record.get("chapter")
    if not book:
        return "no explicit reference"
    if chapter is None:
        return str(book)
    value = f"{book} {chapter}"
    start = record.get("verse_start")
    end = record.get("verse_end")
    if start is not None:
        value += f":{start}"
        if end not in (None, start):
            value += f"-{end}"
    return value


def print_results(query: str, results: list[tuple[float, dict]], show_text: bool) -> None:
    print("\n" + "=" * 78)
    print(f"QUERY: {query}")
    for rank, (score, record) in enumerate(results, start=1):
        print("-" * 78)
        print(
            f"#{rank} score={score:.4f} | {record.get('source_id')} | "
            f"{reference(record)}"
        )
        print(f"Title: {record.get('title', '')}")
        print(f"Chunk ID: {record.get('chunk_id', '')}")
        print(f"URL: {record.get('url', '')}")
        if show_text:
            snippet = " ".join(str(record.get("text", "")).split())
            print("Text: " + textwrap.shorten(snippet, width=500, placeholder=" …"))


def overlaps_expected(
    record: dict,
    book: str,
    chapter: int,
    start: int,
    end: int,
) -> bool:
    if str(record.get("book", "")).casefold() != book.casefold():
        return False
    try:
        if int(record.get("chapter")) != chapter:
            return False
    except (TypeError, ValueError):
        return False
    record_start = record.get("verse_start")
    if record_start is None:
        return True
    try:
        record_start = int(record_start)
        record_end = int(record.get("verse_end") or record_start)
    except (TypeError, ValueError):
        return False
    return record_start <= end and record_end >= start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", help="Run one retrieval query.")
    group.add_argument("--test-suite", action="store_true", help="Run five passage checks.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY is missing from .env")

    index_dir = project_path(args.index_dir)
    vectors, records, config = load_index(index_dir)
    model = str(config["embedding_model"])
    dimension = int(config["embedding_dimension"])
    top_k = args.top_k or int(os.getenv("RAG_TOP_K", "5"))
    if top_k < 1:
        raise SystemExit("ERROR: --top-k/RAG_TOP_K must be at least 1")

    print(f"Index: {index_dir}")
    print(f"Documents: {len(records):,}")
    print(f"Model: {model}")
    print(f"Dimension: {dimension}")
    client = genai.Client(api_key=api_key)

    if args.query:
        query_vector = embed_query(client, args.query, model, dimension)
        results = retrieve(vectors, records, query_vector, top_k)
        print_results(args.query, results, args.show_text)
        return

    passed = 0
    for label, query, book, chapter, start, end in TEST_CASES:
        query_vector = embed_query(client, query, model, dimension)
        results = retrieve(vectors, records, query_vector, top_k)
        print_results(query, results, args.show_text)
        success = any(
            overlaps_expected(record, book, chapter, start, end)
            for _score, record in results
        )
        print(f"CHECK {label}: {'PASS' if success else 'REVIEW'}")
        passed += int(success)

    print("\n" + "=" * 78)
    print(f"PASSAGE CHECKS: {passed}/{len(TEST_CASES)}")
    if passed != len(TEST_CASES):
        raise SystemExit(
            "One or more passages were not represented in the top results. "
            "Review the displayed results before changing retrieval settings."
        )
    print("RETRIEVAL TEST SUITE PASSED")


if __name__ == "__main__":
    main()
