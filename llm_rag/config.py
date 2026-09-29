"""Shared, fixed configuration for the Bible-study LLM and RAG conditions."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _required_environment_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing from {PROJECT_ROOT / '.env'}")
    return value


def _project_path(environment_name: str, default: str) -> Path:
    value = Path(os.getenv(environment_name, default).strip())
    return value if value.is_absolute() else PROJECT_ROOT / value


GEMINI_API_KEY = _required_environment_value("GEMINI_API_KEY")
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().casefold()
if EMBEDDING_PROVIDER not in {"gemini", "ollama"}:
    raise RuntimeError(
        "EMBEDDING_PROVIDER must be either 'gemini' or 'ollama'."
    )

_default_embedding_model = (
    "nomic-embed-text"
    if EMBEDDING_PROVIDER == "ollama"
    else "gemini-embedding-001"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    _default_embedding_model,
).strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIM", "768"))
OLLAMA_API_BASE_URL = os.getenv(
    "OLLAMA_API_BASE_URL",
    "http://localhost:11434",
).strip().rstrip("/")

INDEX_DIR = _project_path(
    "RAG_INDEX_DIR",
    "corpus/processed/index",
)

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "10"))
MAX_CHUNKS_PER_SOURCE = int(
    os.getenv("MAX_CHUNKS_PER_SOURCE", "1")
)

# Identical generation settings are used in the baseline and RAG conditions.
# Gemini 3.5 Flash-Lite no longer accepts legacy sampling parameters such as
# temperature, so freeze the supported thinking level instead.
GENERATION_THINKING_LEVEL = os.getenv(
    "GEMINI_THINKING_LEVEL",
    "minimal",
).strip().casefold()
if GENERATION_THINKING_LEVEL not in {"minimal", "low", "medium", "high"}:
    raise RuntimeError(
        "GEMINI_THINKING_LEVEL must be minimal, low, medium, or high."
    )
GENERATION_MAX_OUTPUT_TOKENS = int(
    os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "700")
)

API_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
API_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
EMBEDDING_TIMEOUT_SECONDS = float(
    os.getenv("EMBEDDING_TIMEOUT_SECONDS", "120")
)
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))

MAX_CONTEXT_CHARACTERS_PER_SOURCE = int(
    os.getenv("RAG_MAX_CHARS_PER_SOURCE", "4000")
)
MAX_CONTEXT_CHARACTERS_TOTAL = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "15000")
)
