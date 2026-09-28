"""Provider-aware embedding calls shared by indexing and live retrieval."""

from __future__ import annotations

import random
import time
from typing import Literal

import numpy as np
import requests


EmbeddingProvider = Literal["gemini", "ollama"]
EmbeddingTask = Literal["document", "query"]


class EmbeddingAPIError(RuntimeError):
    """A sanitized embedding error that never includes an API key."""


def _response_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:500] or "No error message was returned."

    if not isinstance(payload, dict):
        return str(payload)[:500]
    error = payload.get("error", {})
    if isinstance(error, dict):
        return str(error.get("message") or payload)[:500]
    return str(error or payload)[:500]


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
    return min(60.0, (2**attempt) * 2.0 + random.random())


def _validate_matrix(
    values: object,
    text_count: int,
    dimension: int,
    provider: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    expected_shape = (text_count, dimension)
    if matrix.shape != expected_shape:
        raise EmbeddingAPIError(
            f"{provider} returned embedding shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise EmbeddingAPIError(f"{provider} returned non-finite embedding values.")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise EmbeddingAPIError(f"{provider} returned a zero-length embedding.")
    return matrix


def _nomic_text(text: str, model: str, task: EmbeddingTask) -> str:
    """Apply the retrieval prefixes expected by Nomic embedding models."""
    model_family = model.split(":", 1)[0].casefold()
    if model_family != "nomic-embed-text":
        return text
    prefix = "search_document" if task == "document" else "search_query"
    return f"{prefix}: {text}"


def _gemini_embeddings(
    texts: list[str],
    model: str,
    dimension: int,
    task: EmbeddingTask,
    api_key: str,
    base_url: str,
    max_retries: int,
    timeout_seconds: float,
) -> np.ndarray:
    if not api_key:
        raise EmbeddingAPIError("GEMINI_API_KEY is required for Gemini embeddings.")

    model_name = f"models/{model}"
    url = f"{base_url.rstrip('/')}/{model_name}:batchEmbedContents"
    task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
    payload = {
        "requests": [
            {
                "model": model_name,
                "content": {"parts": [{"text": text}]},
                "embedContentConfig": {
                    "taskType": task_type,
                    "outputDimensionality": dimension,
                },
            }
            for text in texts
        ]
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    for attempt in range(max_retries + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if not response.ok:
                retryable = response.status_code == 429 or response.status_code >= 500
                message = _response_message(response)
                if not retryable:
                    raise EmbeddingAPIError(
                        f"Gemini embedding request failed with HTTP "
                        f"{response.status_code}: {message}"
                    )
                raise RuntimeError(
                    f"Gemini returned retryable HTTP {response.status_code}: {message}"
                )

            data = response.json()
            embeddings = data.get("embeddings") or []
            values = [item.get("values", []) for item in embeddings]
            return _validate_matrix(values, len(texts), dimension, "Gemini")
        except EmbeddingAPIError:
            raise
        except (requests.RequestException, RuntimeError, ValueError) as error:
            if attempt >= max_retries:
                raise EmbeddingAPIError(
                    f"Gemini embedding request failed after {attempt + 1} attempts: "
                    f"{type(error).__name__}: {error}"
                ) from error
            time.sleep(_retry_delay(response, attempt))

    raise EmbeddingAPIError("Gemini embedding request reached an unexpected state.")


def _ollama_embeddings(
    texts: list[str],
    model: str,
    dimension: int,
    task: EmbeddingTask,
    base_url: str,
    max_retries: int,
    timeout_seconds: float,
) -> np.ndarray:
    url = f"{base_url.rstrip('/')}/api/embed"
    payload = {
        "model": model,
        "input": [_nomic_text(text, model, task) for text in texts],
        "truncate": True,
        "keep_alive": "30m",
    }

    for attempt in range(max_retries + 1):
        response: requests.Response | None = None
        try:
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            if not response.ok:
                retryable = response.status_code == 429 or response.status_code >= 500
                message = _response_message(response)
                if not retryable:
                    raise EmbeddingAPIError(
                        f"Ollama embedding request failed with HTTP "
                        f"{response.status_code}: {message}"
                    )
                raise RuntimeError(
                    f"Ollama returned retryable HTTP {response.status_code}: {message}"
                )

            data = response.json()
            return _validate_matrix(
                data.get("embeddings") or [],
                len(texts),
                dimension,
                "Ollama",
            )
        except EmbeddingAPIError:
            raise
        except (requests.RequestException, RuntimeError, ValueError) as error:
            if attempt >= max_retries:
                detail = (
                    " Start Ollama and pull the model first: "
                    f"ollama pull {model}"
                )
                raise EmbeddingAPIError(
                    f"Ollama embedding request failed after {attempt + 1} attempts: "
                    f"{type(error).__name__}: {error}.{detail}"
                ) from error
            time.sleep(_retry_delay(response, attempt))

    raise EmbeddingAPIError("Ollama embedding request reached an unexpected state.")


def embed_texts(
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimension: int,
    task: EmbeddingTask,
    gemini_api_key: str = "",
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ollama_base_url: str = "http://localhost:11434",
    max_retries: int = 4,
    timeout_seconds: float = 120.0,
) -> np.ndarray:
    """Embed document or query text with a validated provider response."""
    if not texts:
        raise ValueError("At least one text is required for embedding.")
    if dimension < 1:
        raise ValueError("Embedding dimension must be at least 1.")

    normalized_provider = provider.strip().casefold()
    if normalized_provider == "gemini":
        return _gemini_embeddings(
            texts,
            model,
            dimension,
            task,
            gemini_api_key,
            gemini_base_url,
            max_retries,
            timeout_seconds,
        )
    if normalized_provider == "ollama":
        return _ollama_embeddings(
            texts,
            model,
            dimension,
            task,
            ollama_base_url,
            max_retries,
            timeout_seconds,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER {provider!r}; use 'gemini' or 'ollama'."
    )
