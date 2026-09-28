from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from llm_rag.embeddings import EmbeddingAPIError, embed_texts


class OllamaEmbeddingTests(unittest.TestCase):
    @patch("llm_rag.embeddings.requests.post")
    def test_nomic_document_prefix_and_shape(self, mock_post: Mock) -> None:
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        }
        mock_post.return_value = response

        matrix = embed_texts(
            ["first", "second"],
            provider="ollama",
            model="nomic-embed-text",
            dimension=3,
            task="document",
            max_retries=0,
        )

        self.assertEqual(matrix.shape, (2, 3))
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            payload["input"],
            ["search_document: first", "search_document: second"],
        )

    @patch("llm_rag.embeddings.requests.post")
    def test_nomic_query_prefix(self, mock_post: Mock) -> None:
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {"embeddings": [[0.0, 1.0, 0.0]]}
        mock_post.return_value = response

        embed_texts(
            ["What does the passage mean?"],
            provider="ollama",
            model="nomic-embed-text:latest",
            dimension=3,
            task="query",
            max_retries=0,
        )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(
            payload["input"],
            ["search_query: What does the passage mean?"],
        )

    @patch("llm_rag.embeddings.requests.post")
    def test_wrong_dimension_is_rejected(self, mock_post: Mock) -> None:
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {"embeddings": [[1.0, 2.0]]}
        mock_post.return_value = response

        with self.assertRaises(EmbeddingAPIError):
            embed_texts(
                ["text"],
                provider="ollama",
                model="nomic-embed-text",
                dimension=3,
                task="document",
                max_retries=0,
            )

    def test_provider_must_be_supported(self) -> None:
        with self.assertRaises(ValueError):
            embed_texts(
                ["text"],
                provider="unknown",
                model="model",
                dimension=3,
                task="document",
            )


if __name__ == "__main__":
    unittest.main()
