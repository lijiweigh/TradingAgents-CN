"""OpenAI 厂家连接测试的参数兼容和错误反馈回归测试（不调用外部 API）。"""

import unittest
from unittest.mock import Mock, patch

import requests

from app.services.config_service import ConfigService


class OpenAIProviderConnectionTests(unittest.TestCase):
    def setUp(self):
        self.service = ConfigService()

    @patch("requests.post")
    def test_gpt61_sol_uses_supported_reasoning_parameters(self, post):
        def upstream(url, *, json, headers, timeout):
            self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
            self.assertEqual(json["model"], "gpt-6.1-sol")
            self.assertNotIn("max_tokens", json)
            self.assertNotIn("temperature", json)
            self.assertEqual(json["reasoning_effort"], "low")
            self.assertGreaterEqual(json["max_completion_tokens"], 1024)
            self.assertGreaterEqual(timeout, 30)
            return Mock(status_code=200, json=lambda: {
                "choices": [{"message": {"content": "OK"}}]
            })

        post.side_effect = upstream
        result = self.service._test_openai_api("sk-test", "OpenAI", " gpt-6.1-sol ")
        self.assertTrue(result["success"])

    @patch("requests.post")
    def test_default_model_and_existing_gpt4_parameters(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "OK"}}]
        })
        result = self.service._test_openai_api("sk-test", "OpenAI")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["max_tokens"], 50)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertNotIn("reasoning_effort", payload)
        self.assertTrue(result["success"])

    @patch("requests.post")
    def test_upstream_error_is_visible_without_api_key(self, post):
        post.return_value = Mock(status_code=400, json=lambda: {"error": {
            "message": "Unsupported parameter: max_tokens; key=sk-private-key",
            "code": "unsupported_parameter",
        }})
        result = self.service._test_openai_api("sk-private-key", "OpenAI", "gpt-6.1-sol")
        self.assertFalse(result["success"])
        self.assertIn("HTTP 400", result["message"])
        self.assertIn("Unsupported parameter: max_tokens", result["message"])
        self.assertNotIn("sk-private-key", result["message"])

    @patch("requests.post")
    def test_non_json_upstream_error_preserves_http_status(self, post):
        post.return_value = Mock(status_code=502)
        post.return_value.json.side_effect = ValueError("not JSON")
        result = self.service._test_openai_api("sk-test", "OpenAI", "gpt-6.1-sol")
        self.assertFalse(result["success"])
        self.assertIn("HTTP 502", result["message"])

    @patch("requests.post")
    def test_empty_response_is_not_reported_as_success(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}]
        })
        result = self.service._test_openai_api("sk-test", "OpenAI", "gpt-6.1-sol")
        self.assertFalse(result["success"])

    @patch("requests.post")
    def test_network_exception_does_not_expose_key(self, post):
        post.side_effect = requests.Timeout("request timed out sk-private-key")
        result = self.service._test_openai_api("sk-private-key", "OpenAI", "gpt-6.1-sol")
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["message"])
        self.assertNotIn("sk-private-key", result["message"])


if __name__ == "__main__":
    unittest.main()
