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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIM", "768"))

INDEX_DIR = _project_path(
    "RAG_INDEX_DIR",
    "corpus/processed/development/index",
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Identical generation settings are used in the baseline and RAG conditions.
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
GENERATION_MAX_OUTPUT_TOKENS = int(
    os.getenv("GENERATION_MAX_OUTPUT_TOKENS", "700")
)

API_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45"))
API_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))

MAX_CONTEXT_CHARACTERS_PER_SOURCE = int(
    os.getenv("RAG_MAX_CHARS_PER_SOURCE", "4000")
)
MAX_CONTEXT_CHARACTERS_TOTAL = int(
    os.getenv("RAG_MAX_CONTEXT_CHARS", "15000")
)
