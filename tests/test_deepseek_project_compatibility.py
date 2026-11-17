"""DeepSeek 配置测试、模型工厂及 SDK 的离线回归。"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.models.config import LLMConfig
from app.services.config_service import ConfigService
from core.llm.models import LLMConfig as SDKConfig, LLMProvider, Message, MessageRole
from core.llm.providers.openai_compat import OpenAICompatAdapter
from tradingagents.graph.trading_graph import TradingAgentsGraph, create_llm_by_provider
from tradingagents.llm_adapters.deepseek_adapter import ChatDeepSeek
from tradingagents.llm_adapters.openai_compatible_base import create_openai_compatible_llm


class DeepSeekConfigTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = ConfigService()
        self.provider = {"api_key": "sk-offline-test-key", "name": "deepseek",
                         "default_base_url": "https://deepseek.example/v1", "test_model": "deepseek-flash"}
        db = Mock()
        db.llm_providers.find_one = AsyncMock(return_value=self.provider)
        self.service._get_db = AsyncMock(return_value=db)
        self.response = Mock(status_code=200, json=lambda: {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})

    async def test_config_page_passes_effort_budget_and_endpoint(self):
        for model, effort in [("deepseek-flash", "low"), ("deepseek-v4-pro", "none")]:
            with patch("requests.post", return_value=self.response) as post:
                result = await self.service.test_llm_config(LLMConfig(
                    provider="deepseek", model_name=model, max_tokens=4096, reasoning_effort=effort))
            self.assertTrue(result["success"])
            self.assertEqual(post.call_args.args[0], "https://deepseek.example/v1/chat/completions")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["reasoning_effort"], effort)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertNotIn("thinking", payload)

    async def test_provider_button_uses_saved_effort(self):
        config = LLMConfig(provider="deepseek", model_name="deepseek-flash", reasoning_effort="max")
        self.service.get_system_config = AsyncMock(return_value=SimpleNamespace(llm_configs=[config]))
        with patch("requests.post", return_value=self.response) as post:
            result = await self.service._test_provider_connection("deepseek", "sk-offline-test-key", "DeepSeek")
        self.assertTrue(result["success"])
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "max")

    def test_blank_model_defaults_to_latest_flash(self):
        with patch("requests.post", return_value=self.response) as post:
            self.service._test_deepseek_api("sk-offline-test-key", "DeepSeek")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "deepseek-flash")

    def test_default_catalog_includes_current_models(self):
        catalog = self.service._get_default_model_catalog()
        provider = next(item for item in catalog if item["provider"] == "deepseek")
        self.assertEqual({m["name"] for m in provider["models"]}, {"deepseek-flash", "deepseek-v4-pro"})


class DeepSeekRuntimeEntryTests(unittest.TestCase):
    def test_model_provider_fallback_recognizes_latest_names(self):
        from app.services.simple_analysis_service import _get_default_provider_by_model
        for model in ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"):
            self.assertEqual(_get_default_provider_by_model(model), "deepseek")

    def test_graph_factory_preserves_requested_effort(self):
        for name in ("deepseek-flash", "deepseek-v4-pro"):
            for effort in ("low", "none", "max"):
                model = create_llm_by_provider("deepseek", name, "https://api.deepseek.com", .2, 8000, 30,
                                               api_key="sk-offline-test-key", reasoning_effort=effort)
                payload = model._get_request_payload("OK")
                self.assertEqual(payload["reasoning_effort"], effort)
                self.assertEqual(payload["max_tokens"], 8000)
                self.assertNotIn("thinking", payload.get("extra_body", {}))

    def test_single_provider_graph_preserves_both_efforts(self):
        class ModelsCreated(Exception):
            pass
        config = {"project_dir": "/tmp/deepseek-offline", "llm_provider": "deepseek",
                  "quick_think_llm": "deepseek-flash", "deep_think_llm": "deepseek-v4-pro",
                  "quick_model_config": {"reasoning_effort": "low"},
                  "deep_model_config": {"reasoning_effort": "max"}}
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-offline-test-key"}), \
             patch("tradingagents.graph.trading_graph.set_config"), \
             patch("tradingagents.graph.trading_graph.os.makedirs"), \
             patch("tradingagents.llm_adapters.deepseek_adapter.ChatDeepSeek", side_effect=[Mock(), ModelsCreated]) as cls:
            with self.assertRaises(ModelsCreated):
                TradingAgentsGraph(config=config)
        self.assertEqual([c.kwargs["reasoning_effort"] for c in cls.call_args_list], ["max", "low"])

    def test_compatible_factory_reuses_reasoning_aware_adapter(self):
        model = create_openai_compatible_llm("deepseek", "deepseek-flash", api_key="sk-offline-test-key",
                                            base_url="https://deepseek.example/v1", reasoning_effort="low")
        self.assertIsInstance(model, ChatDeepSeek)
        self.assertEqual(model.openai_api_base, "https://deepseek.example/v1")
        self.assertEqual(model._get_request_payload("OK")["reasoning_effort"], "low")

    def test_sdk_config_and_call_override_effort(self):
        adapter = OpenAICompatAdapter(SDKConfig(provider=LLMProvider.DEEPSEEK, model="deepseek-flash",
                                                reasoning_effort="low", max_tokens=8000))
        messages = [Message(role=MessageRole.USER, content="OK")]
        self.assertEqual(adapter._build_request_kwargs(messages)["reasoning_effort"], "low")
        payload = adapter._build_request_kwargs(messages, reasoning_effort="none", max_tokens=2000)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["max_tokens"], 2000)

    def test_sdk_environment_default_uses_latest_flash(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(SDKConfig.from_env(LLMProvider.DEEPSEEK).model, "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
