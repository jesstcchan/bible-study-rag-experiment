"""Gemini REST calls shared by the baseline and RAG conditions.

This module deliberately uses ``requests`` rather than ``google-genai`` so it
can run inside the oTree environment without changing oTree's websockets pin.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any

import requests

from .config import (
    API_MAX_RETRIES,
    API_TIMEOUT_SECONDS,
    CHAT_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY,
    GENERATION_MAX_OUTPUT_TOKENS,
    GENERATION_TEMPERATURE,
)
from .prompts import SYSTEM_PROMPT


class GeminiAPIError(RuntimeError):
    """A sanitized Gemini API error that never includes the API key."""


def _remove_internal_process_notes(text: str) -> str:
    """Remove model commentary about hidden retrieval/prompt mechanics."""
    cleaned = text.replace("\ufeff", "").strip()
    cleaned = re.sub(
        r"\n*\s*\*?\(?\s*Note:\s*[^\n]*SOURCE_CONTEXT[\s\S]*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    blocks = re.split(r"\n\s*\n", cleaned)
    retained_blocks = []

    for block in blocks:
        compact = re.sub(r"\s+", " ", block).casefold()
        is_internal_note = (
            compact.lstrip("* (").startswith("note:")
            and "source_context" in compact
            and ("citation" in compact or "instruction" in compact)
        )
        if not is_internal_note:
            retained_blocks.append(block)

    return "\n\n".join(retained_blocks).strip()


def _error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:500] or "No error message was returned."

    error = payload.get("error", {})
    return str(error.get("message") or payload)[:500]


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(2.0**attempt + random.random(), 30.0)


def _post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{GEMINI_API_BASE_URL}/models/{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    for attempt in range(API_MAX_RETRIES + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            if attempt >= API_MAX_RETRIES:
                raise GeminiAPIError(
                    f"Gemini request failed after {attempt + 1} attempts: "
                    f"{type(error).__name__}"
                ) from error
        else:
            if response.ok:
                try:
                    return response.json()
                except ValueError as error:
                    raise GeminiAPIError(
                        "Gemini returned a non-JSON success response."
                    ) from error

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= API_MAX_RETRIES:
                raise GeminiAPIError(
                    f"Gemini request failed with HTTP {response.status_code}: "
                    f"{_error_message(response)}"
                )

        time.sleep(_retry_delay(response, attempt))

    raise GeminiAPIError("Gemini request reached an unexpected retry state.")


def embed_query(
    text: str,
    model: str = EMBEDDING_MODEL,
    dimension: int = EMBEDDING_DIMENSION,
) -> list[float]:
    """Embed one live search query compatibly with the document index."""
    payload = {
        "model": f"models/{model}",
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": dimension,
        "content": {
            "parts": [{"text": text}],
        },
    }
    data = _post_json(f"{model}:embedContent", payload)
    values = data.get("embedding", {}).get("values")
    if not isinstance(values, list) or len(values) != dimension:
        actual = len(values) if isinstance(values, list) else "missing"
        raise GeminiAPIError(
            f"Unexpected query embedding dimension: {actual}; expected {dimension}."
        )
    return [float(value) for value in values]


def _generated_text(data: dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        detail = f" Block reason: {block_reason}." if block_reason else ""
        raise GeminiAPIError(f"Gemini returned no answer candidate.{detail}")

    parts = candidates[0].get("content", {}).get("parts") or []
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        finish_reason = candidates[0].get("finishReason", "unknown")
        raise GeminiAPIError(
            f"Gemini returned an empty answer. Finish reason: {finish_reason}."
        )
    cleaned_text = _remove_internal_process_notes(text)
    if not cleaned_text:
        raise GeminiAPIError("Gemini returned only internal process commentary.")
    return cleaned_text


def generate_answer(
    question: str,
    passage_reference: str,
    passage_text: str,
    source_context: str = "NONE",
) -> str:
    """Generate one answer with fixed settings for either study condition."""
    user_input = f"""PASSAGE REFERENCE:
{passage_reference}

PASSAGE TEXT:
{passage_text}

SOURCE CONTEXT:
{source_context}

USER QUESTION:
{question}
"""

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_input}],
            }
        ],
        "generationConfig": {
            "temperature": GENERATION_TEMPERATURE,
            "maxOutputTokens": GENERATION_MAX_OUTPUT_TOKENS,
        },
    }
    data = _post_json(f"{CHAT_MODEL}:generateContent", payload)
    return _generated_text(data)
