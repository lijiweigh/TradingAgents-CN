"""测试配置页、聚合渠道和 LangChain 共用的 GPT-6 参数规则。"""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from app.models.config import LLMConfig
from app.services.config_service import ConfigService
from core.utils.openai_compat import normalize_openai_chat_payload, normalize_openai_responses_payload
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from tradingagents.llm_adapters.openai_model_adapter import ChatOpenAI


def successful_response():
    return Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "OK"}}]})


class PayloadCompatibilityTests(unittest.TestCase):
    def test_gpt6_variants_preserve_budget_and_remove_sampling(self):
        for model in ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna", "openai/gpt-6.1-sol"):
            with self.subTest(model=model):
                original = {"model": model, "max_tokens": 8000, "temperature": .2, "top_p": .9, "logprobs": True, "top_logprobs": 2}
                result = normalize_openai_chat_payload(original)
                self.assertEqual(result["max_completion_tokens"], 8000)
                self.assertEqual(result["reasoning_effort"], "low")
                for key in ("max_tokens", "temperature", "top_p", "logprobs", "top_logprobs"):
                    self.assertNotIn(key, result)
                self.assertIn("max_tokens", original)

    def test_explicit_effort_and_budget_win(self):
        result = normalize_openai_chat_payload({"model": "gpt-6.1-sol", "reasoning_effort": "high", "max_tokens": 8000, "max_completion_tokens": 4000})
        self.assertEqual(result["reasoning_effort"], "high")
        self.assertEqual(result["max_completion_tokens"], 4000)

    def test_none_is_supported_only_on_compatible_models(self):
        for model, effort in (("gpt-6.1-sol", "low"), ("gpt-6-astra", "low"), ("gpt-6-luna", "none"), ("gpt-6.1-terra", "none")):
            result = normalize_openai_chat_payload({"model": model, "reasoning_effort": "none", "temperature": .2})
            self.assertEqual(result["reasoning_effort"], effort)
            self.assertEqual("temperature" in result, effort == "none")

    def test_other_providers_are_unchanged(self):
        for model in ("gpt-4o-mini", "deepseek-chat", "qwen-plus", "kimi-k2.5", "vendor/gpt-6.1-sol"):
            original = {"model": model, "max_tokens": 50, "temperature": .2}
            self.assertEqual(normalize_openai_chat_payload(original, for_test=True), original)

    def test_test_minimum_does_not_change_runtime_budget(self):
        original = {"model": "gpt-6.1-sol", "max_tokens": 50}
        self.assertEqual(normalize_openai_chat_payload(original)["max_completion_tokens"], 50)
        self.assertEqual(normalize_openai_chat_payload(original, for_test=True)["max_completion_tokens"], 1024)

    def test_extra_body_cannot_reintroduce_invalid_parameters(self):
        result = normalize_openai_chat_payload({"model": "gpt-6.1-sol", "max_tokens": 8000, "extra_body": {"max_tokens": 2000, "temperature": .9, "custom": True}})
        self.assertEqual(result["max_completion_tokens"], 2000)
        self.assertNotIn("temperature", result)
        self.assertEqual(result["extra_body"], {"custom": True})

    def test_responses_keeps_effort_and_token_limit(self):
        result = normalize_openai_responses_payload({"model": "gpt-6.1-sol", "max_output_tokens": 2000, "reasoning": {"effort": "high", "summary": "auto"}, "temperature": .2, "include": ["message.output_text.logprobs", "reasoning.encrypted_content"]})
        self.assertEqual(result["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual(result["max_output_tokens"], 2000)
        self.assertNotIn("temperature", result)
        self.assertEqual(result["include"], ["reasoning.encrypted_content"])


class ModelConfigTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = ConfigService()
        db = Mock()
        db.llm_providers.find_one = AsyncMock(return_value={"api_key": "sk-test-valid-api-key", "default_base_url": "https://api.openai.com/v1"})
        self.service._get_db = AsyncMock(return_value=db)

    @patch("requests.post")
    async def test_config_page_uses_same_rules_for_astra_and_sol(self, post):
        post.return_value = successful_response()
        for model in ("gpt-6-astra", "gpt-6.1-sol"):
            result = await self.service.test_llm_config(LLMConfig(provider="openai", model_name=model, max_tokens=8000, temperature=.2))
            self.assertTrue(result["success"])
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], model)
            self.assertEqual(payload["max_completion_tokens"], 8000)
            self.assertNotIn("temperature", payload)
            self.assertNotIn("max_tokens", payload)

    @patch("requests.post")
    async def test_model_not_found_is_not_misreported_as_wrong_url(self, post):
        post.return_value = Mock(status_code=404, json=lambda: {"error": {"message": "model not found sk-test-valid-api-key"}})
        result = await self.service.test_llm_config(LLMConfig(provider="openai", model_name="gpt-6.1-sol"))
        self.assertIn("model not found", result["message"])
        self.assertNotIn("sk-test-valid-api-key", result["message"])
        self.assertNotIn("API端点不存在", result["message"])

    @patch("requests.post")
    async def test_regular_model_keeps_existing_parameters(self, post):
        post.return_value = successful_response()
        result = await self.service.test_llm_config(LLMConfig(provider="openai", model_name="gpt-4o-mini", max_tokens=300, temperature=.4))
        self.assertTrue(result["success"])
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 300)
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], .4)


class CompatibleProviderTests(unittest.TestCase):
    @patch("requests.post")
    def test_custom_provider_and_openrouter(self, post):
        post.return_value = successful_response()
        service = ConfigService()
        result = service._test_openai_compatible_api("sk-test", "Custom", "https://example.com/v1", "custom", "gpt-6.1-sol")
        self.assertTrue(result["success"])
        self.assertIn("max_completion_tokens", post.call_args.kwargs["json"])
        result = service._test_openrouter_api("sk-test", "OpenRouter", "openai/gpt-6-astra")
        self.assertTrue(result["success"])
        self.assertNotIn("temperature", post.call_args.kwargs["json"])


class LangChainCompatibilityTests(unittest.TestCase):
    def test_runtime_overrides_and_tools_use_responses(self):
        model = ChatOpenAI(model="gpt-6.1-sol", api_key="sk-test", temperature=.2, max_tokens=8000)
        tools = [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}]
        messages = [HumanMessage(content="echo OK"), AIMessage(content="", tool_calls=[{"id": "call_1", "name": "echo", "args": {"text": "OK"}}]), ToolMessage(content="OK", tool_call_id="call_1")]
        payload = model._get_request_payload(messages, tools=tools, temperature=.9, max_tokens=2000)
        self.assertTrue(model._use_responses_api(payload))
        self.assertNotIn("messages", payload)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["max_output_tokens"], 2000)
        self.assertEqual(payload["reasoning"]["effort"], "low")
        self.assertTrue(any(item.get("type") == "function_call_output" for item in payload["input"]))
        self.assertEqual(payload["tools"][0]["name"], "echo")

    def test_regular_langchain_model_remains_on_chat_completions(self):
        model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-test", temperature=.4, max_tokens=300)
        payload = model._get_request_payload([HumanMessage(content="OK")])
        self.assertIn("messages", payload)
        self.assertEqual(payload["temperature"], .4)
        self.assertFalse(model._use_responses_api(payload))


class LangChainTextOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = ChatOpenAI(model="gpt-6.1-sol", api_key="sk-test")

    def result(self):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=[
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
            {"type": "text", "text": "报告正文", "id": "msg_1"},
        ]))])

    async def test_sync_async_reports_are_strings_and_history_is_preserved(self):
        with patch.object(BaseChatOpenAI, "_generate", return_value=self.result()):
            sync = self.model._generate([]).generations[0].message
        with patch.object(BaseChatOpenAI, "_agenerate", new=AsyncMock(return_value=self.result())):
            async_message = (await self.model._agenerate([])).generations[0].message
        for message in (sync, async_message):
            self.assertEqual(message.content, "报告正文")
            payload = self.model._get_request_payload([HumanMessage(content="报告"), message])
            self.assertEqual(payload["input"][1]["type"], "reasoning")
            self.assertEqual(payload["input"][1]["encrypted_content"], "opaque")
            self.assertEqual(payload["input"][2]["content"][0]["text"], "报告正文")

    async def test_sync_async_streams_keep_text_and_reasoning_on_merge(self):
        def chunks():
            for block in [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque", "index": 0},
                {"type": "text", "text": "报告", "index": 1},
                {"type": "text", "text": "正文", "index": 1},
            ]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=[block]))
        async def async_chunks(*args, **kwargs):
            for chunk in chunks():
                yield chunk
        with patch.object(BaseChatOpenAI, "_stream", side_effect=lambda *a, **k: chunks()):
            sync = list(self.model._stream([]))
        with patch.object(BaseChatOpenAI, "_astream", new=async_chunks):
            asynchronous = [chunk async for chunk in self.model._astream([])]
        for stream in (sync, asynchronous):
            self.assertTrue(all(isinstance(chunk.message.content, str) for chunk in stream))
            merged = stream[0].message + stream[1].message + stream[2].message
            self.assertEqual(merged.content, "报告正文")
            payload = self.model._get_request_payload([merged])
            self.assertEqual(payload["input"][0]["encrypted_content"], "opaque")
            self.assertEqual(payload["input"][1]["content"][0]["text"], "报告正文")

    def test_tool_calls_and_output_order_survive_text_conversion(self):
        original = AIMessage(content=[
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "echo", "arguments": '{"text":"OK"}'},
        ], tool_calls=[{"id": "call_1", "name": "echo", "args": {"text": "OK"}}])
        message = self.model._text_message(original)
        self.assertEqual(message.content, "")
        self.assertEqual(message.tool_calls, original.tool_calls)
        payload = self.model._get_request_payload([message, ToolMessage(content="OK", tool_call_id="call_1")])
        self.assertEqual([item["type"] for item in payload["input"]], ["reasoning", "function_call", "function_call_output"])
        self.assertEqual(original.content[0]["encrypted_content"], "opaque")


if __name__ == "__main__":
    unittest.main()
