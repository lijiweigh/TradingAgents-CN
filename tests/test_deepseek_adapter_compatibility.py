"""DeepSeek 最新模型的参数与推理工具回合回归，全部通过本地模拟 HTTP。"""

import json
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.llm_adapters.deepseek_adapter import ChatDeepSeek


TOOL = {"type": "function", "function": {
    "name": "echo", "description": "Echo text", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
    },
}}


class DeepSeekAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.sync_client = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.async_client = httpx.AsyncClient(transport=httpx.MockTransport(self.respond))
        self.model = ChatDeepSeek(
            api_key="sk-offline-test", max_tokens=8000, reasoning_effort="low",
            http_client=self.sync_client, http_async_client=self.async_client,
        )
        self.tracking = patch("tradingagents.llm_adapters.deepseek_adapter.TOKEN_TRACKING_ENABLED", False)
        self.tracking.start()

    async def asyncTearDown(self):
        self.tracking.stop()
        self.sync_client.close()
        await self.async_client.aclose()

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        self.assertEqual(request.url.path, "/chat/completions")
        self.assertNotIn("max_completion_tokens", body)
        self.assertIn("max_tokens", body)
        is_followup = any(message.get("role") == "tool" for message in body["messages"])
        if body.get("stream"):
            deltas = [
                {"role": "assistant", "content": "", "reasoning_content": "先检查"},
                {"reasoning_content": "工具参数"},
                {"tool_calls": [{"index": 0, "id": "call_echo", "type": "function", "function": {"name": "echo", "arguments": '{"text":'}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '"OK"}'}}]},
            ]
            events = [{"id": "chat-test", "object": "chat.completion.chunk", "created": 1,
                       "model": body["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                      for delta in deltas]
            events.append({"id": "chat-test", "object": "chat.completion.chunk", "created": 1,
                           "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})
            data = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})
        message = {"role": "assistant", "content": "OK" if is_followup else "", "reasoning_content": "先检查工具参数"}
        if not is_followup:
            message["tool_calls"] = [{"id": "call_echo", "type": "function", "function": {"name": "echo", "arguments": '{"text":"OK"}'}}]
        return httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": "stop" if is_followup else "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })

    def test_default_and_latest_model_recognition(self):
        self.assertEqual(self.model.model_name, "deepseek-flash")
        for name in ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4.1-flash"):
            self.assertTrue(ChatDeepSeek._supports_thinking_mode(name))

    def test_constructor_model_kwargs_and_none_preserved(self):
        for model in ("deepseek-flash", "deepseek-v4-pro"):
            for kwargs in ({"reasoning_effort": "none"}, {"model_kwargs": {"reasoning_effort": "low"}}):
                llm = ChatDeepSeek(model=model, api_key="sk-offline-test", **kwargs)
                payload = llm._get_request_payload("OK")
                self.assertEqual(payload["reasoning_effort"], "none" if "reasoning_effort" in kwargs else "low")
                self.assertNotIn("thinking", payload.get("extra_body") or {})

    def test_invoke_and_bind_overrides_use_deepseek_token_parameter(self):
        self.model.invoke("OK", max_tokens=1234, reasoning_effort="none")
        self.assertEqual(self.requests[-1]["max_tokens"], 1234)
        self.assertEqual(self.requests[-1]["reasoning_effort"], "none")
        self.assertNotIn("thinking", self.requests[-1])
        self.model.bind(max_completion_tokens=4321, reasoning_effort="max").invoke("OK")
        self.assertEqual(self.requests[-1]["max_tokens"], 4321)
        self.assertEqual(self.requests[-1]["reasoning_effort"], "max")

    def test_explicit_thinking_and_extra_body_override(self):
        self.model.invoke("OK", thinking={"type": "disabled"}, reasoning_effort="high")
        self.assertEqual(self.requests[-1]["thinking"], {"type": "disabled"})
        self.model.bind(thinking={"type": "enabled"}, reasoning_effort="none").invoke("OK")
        self.assertEqual(self.requests[-1]["thinking"], {"type": "enabled"})
        self.model.invoke("OK", extra_body={"max_completion_tokens": 999, "thinking": {"type": "disabled"}})
        self.assertEqual(self.requests[-1]["max_tokens"], 999)
        self.assertEqual(self.requests[-1]["thinking"], {"type": "disabled"})

    def assert_tool_history(self, message):
        self.assertEqual(message.additional_kwargs["reasoning_content"], "先检查工具参数")
        self.assertEqual(message.tool_calls[0]["args"], {"text": "OK"})
        return [HumanMessage(content="Echo OK"), message, ToolMessage(content="OK", tool_call_id="call_echo")]

    def assert_sent_reasoning(self):
        assistant = next(message for message in self.requests[-1]["messages"] if message["role"] == "assistant")
        self.assertEqual(assistant["reasoning_content"], "先检查工具参数")

    def test_sync_nonstream_tool_round_trip(self):
        first = self.model.bind_tools([TOOL]).invoke("Echo OK")
        final = self.model.bind_tools([TOOL]).invoke(self.assert_tool_history(first))
        self.assertEqual(final.content, "OK")
        self.assert_sent_reasoning()

    async def test_async_nonstream_tool_round_trip(self):
        first = await self.model.bind_tools([TOOL]).ainvoke("Echo OK")
        final = await self.model.bind_tools([TOOL]).ainvoke(self.assert_tool_history(first))
        self.assertEqual(final.content, "OK")
        self.assert_sent_reasoning()

    def test_sync_stream_tool_round_trip(self):
        chunks = list(self.model.bind_tools([TOOL]).stream("Echo OK"))
        merged = chunks[0]
        for chunk in chunks[1:]:
            merged += chunk
        final = self.model.bind_tools([TOOL]).invoke(self.assert_tool_history(merged))
        self.assertEqual(final.content, "OK")
        self.assert_sent_reasoning()
        self.assertEqual(merged.usage_metadata["total_tokens"], 30)

    async def test_async_stream_tool_round_trip(self):
        chunks = [chunk async for chunk in self.model.bind_tools([TOOL]).astream("Echo OK")]
        merged = chunks[0]
        for chunk in chunks[1:]:
            merged += chunk
        final = await self.model.bind_tools([TOOL]).ainvoke(self.assert_tool_history(merged))
        self.assertEqual(final.content, "OK")
        self.assert_sent_reasoning()

    def test_empty_reasoning_is_preserved(self):
        message = AIMessage(content="OK", additional_kwargs={"reasoning_content": ""})
        payload = self.model._get_request_payload([HumanMessage(content="OK"), message])
        self.assertEqual(payload["messages"][-1]["reasoning_content"], "")


if __name__ == "__main__":
    unittest.main()
