import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from raglab.config import ModelConfig
from raglab.models import HTTPBackend, MockBackend
from raglab.prompts import answer_messages
from raglab.schema import Document


class BackendTests(unittest.TestCase):
    def test_http_payload_and_response(self):
        response = io.BytesIO(
            json.dumps({"choices": [{"message": {"content": '{"answer":"X"}'}}]}).encode()
        )
        messages = answer_messages("Q", (Document("d", "T", "Evidence"),))
        with patch("raglab.models.urllib.request.urlopen", return_value=response) as opened:
            backend = HTTPBackend(ModelConfig(backend="http", model="served-qwen"))
            self.assertEqual(backend.complete(messages), '{"answer":"X"}')
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["model"], "served-qwen")
        self.assertEqual(body["temperature"], 0)

    def test_http_failure_never_falls_back_to_mock(self):
        error = urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, None)
        with patch("raglab.models.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
                HTTPBackend(ModelConfig(backend="http", model="qwen")).complete([])

    def test_mock_refuses_real_questions_without_demo_fact(self):
        with self.assertRaisesRegex(ValueError, "synthetic fixtures"):
            MockBackend("generator").complete(answer_messages("Real Q", (Document("d", "T", "E"),)))


if __name__ == "__main__":
    unittest.main()
