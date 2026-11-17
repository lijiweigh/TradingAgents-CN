"""GPT-6 SDK 适配器回归；全部使用内存响应，不访问网络。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from core.llm.models import LLMConfig, LLMProvider, Message
from core.llm.providers.openai_compat import OpenAICompatAdapter


TOOLS = [{"type": "function", "function": {
    "name": "echo", "description": "Return text", "parameters": {
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}}]


def tool_response(call_id="call-1"):
    return {"model": "gpt-6.1-sol", "status": "completed", "output": [
        {"id": "reason-1", "type": "reasoning", "summary": [], "encrypted_content": "encrypted-test"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking"}]},
        {"type": "function_call", "call_id": call_id, "name": "echo", "arguments": '{"text":"hello"}'},
    ], "usage": {"input_tokens": 20, "output_tokens": 12, "total_tokens": 32}}


def text_response():
    return {"model": "gpt-6.1-sol", "status": "completed", "output": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}
    ]}


def chat_response():
    return SimpleNamespace(model="test", choices=[SimpleNamespace(
        message=SimpleNamespace(content="hello", tool_calls=[]), finish_reason="stop"
    )], usage=None)


def adapter(model="gpt-6.1-sol", provider=LLMProvider.OPENAI):
    result = OpenAICompatAdapter(LLMConfig(provider=provider, model=model, api_key="sk-fake",
                                          max_tokens=4096, temperature=.7))
    result._sync_client = Mock()
    result._async_client = Mock()
    return result


class OpenAICompatGPT6Tests(unittest.TestCase):
    def test_sync_plain_request_uses_shared_parameter_rules(self):
        client = adapter()
        client._sync_client.chat.completions.create.return_value = chat_response()
        result = client.chat([Message(role="user", content="hello")], max_tokens=2048, top_p=.5)
        payload = client._sync_client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(result.content, "hello")
        client._sync_client.responses.create.assert_not_called()

    def test_other_provider_keeps_existing_chat_payload(self):
        client = adapter("deepseek-chat", LLMProvider.DEEPSEEK)
        client._sync_client.chat.completions.create.return_value = chat_response()
        client.chat([Message(role="user", content="hello")], tools=TOOLS)
        payload = client._sync_client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["temperature"], .7)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["tools"], TOOLS)
        client._sync_client.responses.create.assert_not_called()

    def test_sync_tool_roundtrip_preserves_reasoning_and_outputs_without_tools_on_followup(self):
        client = adapter()
        first_response = tool_response()
        client._sync_client.responses.create.side_effect = [first_response, text_response()]
        history = [Message(role="user", content="echo hello")]
        response = client.chat(history, tools=TOOLS)
        self.assertEqual(response.tool_calls[0].arguments, {"text": "hello"})
        self.assertEqual(response.usage, {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32})
        payload = client._sync_client.responses.create.call_args.kwargs
        self.assertEqual(payload["tools"][0]["name"], "echo")
        self.assertFalse(payload["tools"][0]["strict"])
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertFalse(payload["store"])
        self.assertIn("reasoning.encrypted_content", payload["include"])
        history += [response.to_message(), Message(role="tool", tool_call_id="call-1", content="hello")]
        final = client.chat(history)
        self.assertEqual(final.content, "hello")
        followup = client._sync_client.responses.create.call_args.kwargs
        self.assertEqual(followup["input"][1:-1], first_response["output"])
        self.assertEqual(followup["input"][-1], {"type": "function_call_output", "call_id": "call-1", "output": "hello"})
        client._sync_client.chat.completions.create.assert_not_called()

    def test_cached_items_are_bounded_and_do_not_leak_to_other_calls(self):
        client = adapter()
        response = client._parse_responses_response(tool_response("original"))
        for i in range(257):
            client._parse_responses_response(tool_response(f"other-{i}"))
        self.assertEqual(len(client._tool_response_items), 256)
        request = client._build_request_kwargs([response.to_message()])
        payload = client._responses_request(request)
        self.assertFalse(any(item.get("type") == "reasoning" for item in payload["input"]))
        self.assertEqual(payload["input"][-1]["call_id"], "original")

    def test_structured_format_and_named_tool_choice_conversion(self):
        client = adapter()
        payload = client._responses_request(client._build_request_kwargs(
            [Message(role="user", content="hello")], TOOLS,
            tool_choice={"type": "function", "function": {"name": "echo"}},
            response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
        ))
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "echo"})
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertNotIn("response_format", payload)

    def test_sync_stream_uses_responses_and_closes_stream(self):
        client = adapter()
        events = [{"type": "response.output_text.delta", "delta": "hello"},
                  {"type": "response.completed", "response": text_response()}]
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter(events))
        client._sync_client.responses.create.return_value = stream
        result = list(client.chat_stream_deltas([Message(role="user", content="hello")], tools=TOOLS))
        self.assertEqual(result, [{"type": "content", "text": "hello"}])
        self.assertTrue(client._sync_client.responses.create.call_args.kwargs["stream"])
        stream.close.assert_called_once()

    def test_sync_plain_stream_normalizes_chat_parameters(self):
        client = adapter()
        client._sync_client.chat.completions.create.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
        ])
        self.assertEqual(list(client.chat_stream_deltas([Message(role="user", content="hello")])),
                         [{"type": "content", "text": "hello"}])
        payload = client._sync_client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("temperature", payload)

    def test_failed_stream_does_not_silently_succeed(self):
        client = adapter()
        stream = Mock()
        stream.__iter__ = Mock(return_value=iter([{"type": "response.failed"}]))
        client._sync_client.responses.create.return_value = stream
        with self.assertRaises(ValueError):
            list(client.chat_stream_deltas([Message(role="user", content="hello")], tools=TOOLS))
        stream.close.assert_called_once()


class OpenAICompatGPT6AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_tool_request_and_followup_use_responses(self):
        client = adapter()
        client._async_client.responses.create = AsyncMock(side_effect=[tool_response(), text_response()])
        history = [Message(role="user", content="hello")]
        response = await client.achat(history, tools=TOOLS)
        result = await client.achat(history + [response.to_message(), Message(role="tool", tool_call_id="call-1", content="hello")])
        self.assertEqual(result.content, "hello")
        self.assertEqual(client._async_client.responses.create.await_count, 2)
        client._async_client.chat.completions.create.assert_not_called()

    async def test_async_plain_request_keeps_chat_api_with_supported_params(self):
        client = adapter()
        client._async_client.chat.completions.create = AsyncMock(return_value=chat_response())
        result = await client.achat([Message(role="user", content="hello")])
        payload = client._async_client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("temperature", payload)
        self.assertEqual(result.content, "hello")

    async def test_async_stream_uses_responses_and_closes_stream(self):
        client = adapter()
        stream = AsyncMock()
        stream.__aiter__.return_value = [{"type": "response.output_text.delta", "delta": "hello"},
                                         {"type": "response.completed", "response": text_response()}]
        client._async_client.responses.create = AsyncMock(return_value=stream)
        chunks = [chunk async for chunk in client.astream([Message(role="user", content="hello")], tools=TOOLS)]
        self.assertEqual(chunks, ["hello"])
        self.assertTrue(client._async_client.responses.create.call_args.kwargs["stream"])
        stream.close.assert_awaited_once()

    async def test_async_plain_stream_normalizes_chat_parameters(self):
        client = adapter()
        stream = AsyncMock()
        stream.__aiter__.return_value = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
        ]
        client._async_client.chat.completions.create = AsyncMock(return_value=stream)
        chunks = [chunk async for chunk in client.astream([Message(role="user", content="hello")])]
        self.assertEqual(chunks, ["hello"])
        payload = client._async_client.chat.completions.create.call_args.kwargs
        self.assertEqual(payload["max_completion_tokens"], 4096)
        self.assertNotIn("temperature", payload)


if __name__ == "__main__":
    unittest.main()
