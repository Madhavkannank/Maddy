"""
Unit tests for ModelProvider interface and OllamaProvider backend using mocked HTTP calls.
"""
import io
import json
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from config.settings import Settings
from models.base import ModelMessage
from models.ollama_provider import ModelProviderError, OllamaProvider

class TestOllamaProvider(unittest.TestCase):

    def setUp(self):
        self.settings = Settings(
            model_host="http://localhost:11434",
            model_name="qwen3.5-4b",
            model_timeout=10.0,
        )
        self.provider = OllamaProvider(settings=self.settings)

    @patch("urllib.request.urlopen")
    def test_generate_success(self, mock_urlopen):
        mock_response_data = {
            "model": "qwen3.5-4b",
            "message": {"role": "assistant", "content": "Hello! I am M.A.D.D.Y."},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 12,
            "eval_count": 8,
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        response = self.provider.generate(
            prompt="Hello",
            system_prompt="You are M.A.D.D.Y.",
        )

        self.assertEqual(response.content, "Hello! I am M.A.D.D.Y.")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 8)
        self.assertIsNotNone(response.latency_ms)

    @patch("urllib.request.urlopen")
    def test_stream_success(self, mock_urlopen):
        chunk1 = json.dumps({"message": {"content": "Hello "}}).encode("utf-8") + b"\n"
        chunk2 = json.dumps({"message": {"content": "world!"}}).encode("utf-8") + b"\n"
        
        mock_response = MagicMock()
        mock_response.__iter__.return_value = iter([chunk1, chunk2])
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        chunks = list(self.provider.stream(prompt="Say hello"))
        self.assertEqual(chunks, ["Hello ", "world!"])

    @patch("urllib.request.urlopen")
    def test_structured_success(self, mock_urlopen):
        expected_json = {"action": "search", "query": "weather"}
        mock_response_data = {
            "message": {"role": "assistant", "content": json.dumps(expected_json)},
            "done": True,
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "query": {"type": "string"},
            },
        }

        result = self.provider.structured(prompt="Search weather", schema=schema)
        self.assertEqual(result, expected_json)

    @patch("urllib.request.urlopen")
    def test_unreachable_host_raises_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        with self.assertRaises(ModelProviderError):
            self.provider.generate(prompt="Test")

if __name__ == "__main__":
    unittest.main()
