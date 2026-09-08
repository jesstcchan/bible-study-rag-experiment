"""Passage-aware retrieval and controlled baseline/RAG answer generation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MAX_CONTEXT_CHARACTERS_PER_SOURCE,
    MAX_CONTEXT_CHARACTERS_TOTAL,
    RAG_TOP_K,
)
from .llm import embed_query, generate_answer


BASELINE_CONDITIONS = {"baseline", "llm", "no_retrieval"}
RAG_CONDITIONS = {"rag", "retrieval", "rag_enhanced"}


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    source_id: str
    source_name: str
    title: str
    url: str
    license: str
    biblical_reference: str
    text: str

    def log_record(self) -> dict[str, Any]:
        """Return JSON-serializable retrieval evidence for later study logging."""
        return asdict(self)


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    condition: str
    sources: tuple[RetrievedChunk, ...]
    latency_ms: int

    def source_log_json(self) -> str:
        return json.dumps(
            [source.log_record() for source in self.sources],
            ensure_ascii=False,
        )


def _read_jsonl(path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid index metadata JSON on line {line_number}: {error}"
                ) from error
    return records


@lru_cache(maxsize=1)
def _load_index():
    embeddings_path = INDEX_DIR / "embeddings.npy"
    metadata_path = INDEX_DIR / "chunk_metadata.jsonl"
    config_path = INDEX_DIR / "index_config.json"

    missing = [
        str(path)
        for path in (embeddings_path, metadata_path, config_path)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("Missing RAG index file(s): " + ", ".join(missing))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise RuntimeError("The configured RAG index is not marked complete.")

    index_model = str(config.get("embedding_model", ""))
    index_dimension = int(config.get("embedding_dimension", 0))
    if index_model != EMBEDDING_MODEL:
        raise RuntimeError(
            "Embedding model mismatch: "
            f"index={index_model!r}, environment={EMBEDDING_MODEL!r}."
        )
    if index_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"index={index_dimension}, environment={EMBEDDING_DIMENSION}."
        )

    vectors = np.load(embeddings_path, mmap_mode="r")
    metadata = _read_jsonl(metadata_path)
    if vectors.shape != (len(metadata), index_dimension):
        raise RuntimeError(
            f"Index shape {vectors.shape} does not match "
            f"{len(metadata)} metadata records and dimension {index_dimension}."
        )

    return vectors, metadata, config


def _normalize_book(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _parse_passage_reference(reference: str) -> tuple[str, int, int, int]:
    normalized = reference.replace("–", "-").replace("—", "-").strip()
    match = re.fullmatch(
        r"(?P<book>(?:[1-3]\s+)?[A-Za-z]+(?:\s+[A-Za-z]+)*)\s+"
        r"(?P<chapter>\d+):(?P<start>\d+)(?:-(?P<end>\d+))?",
        normalized,
    )
    if not match:
        raise ValueError(f"Unsupported passage reference: {reference!r}")
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    return match.group("book"), int(match.group("chapter")), start, end


def _integer_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_matches_passage(
    record: dict[str, Any],
    reference: str,
) -> bool:
    book, chapter, passage_start, passage_end = _parse_passage_reference(reference)
    record_book = record.get("book")
    record_chapter = _integer_or_none(record.get("chapter"))

    if record_book and record_chapter is not None:
        if _normalize_book(record_book) != _normalize_book(book):
            return False
        if record_chapter != chapter:
            return False

        record_start = _integer_or_none(record.get("verse_start"))
        record_end = _integer_or_none(record.get("verse_end"))
        if record_start is None:
            # Keep chapter introductions and summaries for the correct chapter.
            return True
        if record_end is None:
            record_end = record_start
        return record_start <= passage_end and record_end >= passage_start

    # Some commentary records may encode the reference only in their title/text.
    searchable = f"{record.get('title', '')}\n{record.get('text', '')}".casefold()
    return f"{book} {chapter}".casefold() in searchable


def _biblical_reference(record: dict[str, Any]) -> str:
    if not record.get("book") or record.get("chapter") in (None, ""):
        return ""
    reference = f"{record['book']} {record['chapter']}"
    start = record.get("verse_start")
    end = record.get("verse_end")
    if start not in (None, ""):
        reference += f":{start}"
        if end not in (None, "", start):
            reference += f"-{end}"
    return reference


def retrieve_chunks(
    question: str,
    passage_reference: str,
    top_k: int = RAG_TOP_K,
) -> tuple[RetrievedChunk, ...]:
    """Retrieve the highest-scoring chunks associated with this passage."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    vectors, metadata, config = _load_index()
    candidate_indices = [
        index
        for index, record in enumerate(metadata)
        if _record_matches_passage(record, passage_reference)
    ]
    if not candidate_indices:
        raise RuntimeError(
            f"No indexed chunks matched passage {passage_reference!r}."
        )

    query_text = (
        f"Biblical passage: {passage_reference}\n"
        f"Bible-study question: {question}"
    )
    raw_query = np.asarray(
        embed_query(
            query_text,
            model=str(config["embedding_model"]),
            dimension=int(config["embedding_dimension"]),
        ),
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(raw_query))
    if norm == 0:
        raise RuntimeError("Gemini returned a zero-length query embedding.")
    query_vector = raw_query / norm

    candidate_array = np.asarray(candidate_indices, dtype=np.int64)
    scores = np.asarray(vectors[candidate_array] @ query_vector)
    number_to_return = min(top_k, len(candidate_indices))
    local_order = np.argsort(scores)[::-1][:number_to_return]

    results: list[RetrievedChunk] = []
    for rank, local_index in enumerate(local_order, start=1):
        metadata_index = candidate_indices[int(local_index)]
        record = metadata[metadata_index]
        results.append(
            RetrievedChunk(
                rank=rank,
                score=float(scores[int(local_index)]),
                chunk_id=str(record.get("chunk_id", "")),
                source_id=str(record.get("source_id", "")),
                source_name=str(record.get("source_name", "")),
                title=str(record.get("title", "")),
                url=str(record.get("url", "")),
                license=str(record.get("license", "")),
                biblical_reference=_biblical_reference(record),
                text=str(record.get("text", "")),
            )
        )
    return tuple(results)


def format_source_context(sources: tuple[RetrievedChunk, ...]) -> str:
    """Format retrieved evidence using stable citation labels."""
    blocks: list[str] = []
    used_characters = 0

    for source in sources:
        available = MAX_CONTEXT_CHARACTERS_TOTAL - used_characters
        if available <= 0:
            break
        text = source.text[: min(MAX_CONTEXT_CHARACTERS_PER_SOURCE, available)]
        block = (
            f"[S{source.rank}]\n"
            f"Title: {source.title}\n"
            f"Source: {source.source_name}\n"
            f"Biblical reference: {source.biblical_reference}\n"
            f"URL: {source.url}\n"
            f"Content:\n{text}"
        )
        blocks.append(block)
        used_characters += len(text)

    return "\n\n".join(blocks) if blocks else "NONE"


def answer_question(
    condition: str,
    question: str,
    passage_reference: str,
    passage_text: str,
    top_k: int = RAG_TOP_K,
    conversation_history: list[dict[str, str]] | None = None,
) -> AnswerResult:
    """Run either condition while holding the generator settings constant."""
    normalized_condition = condition.strip().casefold().replace("-", "_")
    started = time.monotonic()

    if normalized_condition in BASELINE_CONDITIONS:
        canonical_condition = "baseline"
        sources: tuple[RetrievedChunk, ...] = ()
        source_context = "NONE"
    elif normalized_condition in RAG_CONDITIONS:
        canonical_condition = "rag"
        sources = retrieve_chunks(question, passage_reference, top_k=top_k)
        source_context = format_source_context(sources)
    else:
        raise ValueError(
            f"Unknown system condition {condition!r}; expected 'baseline' or 'rag'."
        )

    answer = generate_answer(
        question=question,
        passage_reference=passage_reference,
        passage_text=passage_text,
        source_context=source_context,
        conversation_history=conversation_history,
    )
    latency_ms = round((time.monotonic() - started) * 1000)
    return AnswerResult(
        answer=answer,
        condition=canonical_condition,
        sources=sources,
        latency_ms=latency_ms,
    )