from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch


os.environ.setdefault("GEMINI_API_KEY", "test-api-key")

from llm_rag.llm import generate_answer


class GeminiGenerationTests(unittest.TestCase):
    @patch("llm_rag.llm.requests.post")
    def test_supported_gemini_35_generation_payload(self, mock_post: Mock) -> None:
        response = Mock()
        response.ok = True
        response.status_code = 200
        response.headers = {}
        response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "A test answer."}]}}
            ]
        }
        mock_post.return_value = response

        answer = generate_answer(
            question="What does this passage mean?",
            passage_reference="1 Samuel 8:4–9",
            passage_text="A short passage.",
        )

        self.assertEqual(answer, "A test answer.")
        payload = mock_post.call_args.kwargs["json"]
        generation_config = payload["generationConfig"]
        self.assertNotIn("temperature", generation_config)
        self.assertEqual(
            generation_config["thinkingConfig"]["thinkingLevel"],
            "minimal",
        )
        self.assertEqual(payload["contents"][-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
