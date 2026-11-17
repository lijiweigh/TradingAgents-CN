"""
OpenAI 兼容 API 适配器

支持所有兼容 OpenAI API 格式的提供商:
- OpenAI
- DeepSeek
- 通义千问 (DashScope)
- 智谱AI
- 硅基流动
- Ollama
- OpenRouter
"""

import json
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI
from core.utils.openai_compat import (
    is_gpt6_model, normalize_openai_chat_payload, normalize_openai_responses_payload,
)

from ..models import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    ToolCall,
)
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class OpenAICompatAdapter(BaseAdapter):
    """OpenAI 兼容 API 适配器"""
    
    # 支持的提供商列表
    SUPPORTED_PROVIDERS = [
        LLMProvider.OPENAI,
        LLMProvider.DEEPSEEK,
        LLMProvider.DASHSCOPE,
        LLMProvider.ZHIPU,
        LLMProvider.SILICONFLOW,
        LLMProvider.OLLAMA,
        LLMProvider.OPENROUTER,
    ]
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._sync_client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        # Message 的公开结构保持不变；按工具调用 ID 保留完整工具回合，维持推理项及输出项顺序。
        self._tool_response_items: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    
    def initialize(self) -> None:
        """初始化 OpenAI 客户端"""
        # 🔥 检查 API key 是否存在
        api_key = self.config.api_key
        
        # 🔥 本地模型（Ollama）不需要 API Key，使用占位符
        if not api_key and self.config.provider == LLMProvider.OLLAMA:
            api_key = "ollama"  # 本地模型使用占位符，实际不会验证
        
        if not api_key:
            # 根据 provider 确定应该使用的环境变量名
            provider_env_map = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                LLMProvider.DASHSCOPE: "DASHSCOPE_API_KEY",
                LLMProvider.ZHIPU: "ZHIPU_API_KEY",
                LLMProvider.SILICONFLOW: "SILICONFLOW_API_KEY",
                LLMProvider.OLLAMA: "OLLAMA_API_KEY",
                LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            }
            # 兼容 use_enum_values=True：config.provider 可能是字符串
            provider_val = self.config.provider
            provider_key = provider_val.value if hasattr(provider_val, "value") else provider_val
            expected_env_key = provider_env_map.get(provider_key, "API_KEY")
            raise ValueError(
                f"The api_key client option must be set either by passing api_key to the client "
                f"or by setting the {expected_env_key} environment variable. "
                f"Provider: {provider_key}, Model: {self.config.model}"
            )
        
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            # 关闭 SDK 自带重试：重试统一由应用层 retry_sync/retry_async 控制。
            # 否则 SDK(默认2次) x 应用层(3次) 双层叠加，单次调用最坏可达
            # 十几分钟级等待，且 timeout 语义变得不可预期。
            "max_retries": 0,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        self._sync_client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """转换消息格式为 OpenAI 格式"""
        result = []
        for msg in messages:
            converted = {"role": msg.role, "content": msg.content or ""}

            # 推理模型（如 Qwen thinking / DeepSeek reasoning）在工具回合中
            # 需要回传上一轮 assistant 的 reasoning_content，否则 provider 会拒绝请求。
            if getattr(msg, 'reasoning_content', None):
                converted["reasoning_content"] = msg.reasoning_content
            
            if msg.name:
                converted["name"] = msg.name
            
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id
            
            result.append(converted)
        
        return result
    
    def _build_request_kwargs(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """构建请求参数

        支持通过 kwargs 覆盖 temperature / max_tokens 等参数，
        优先级：kwargs > self.config。
        """
        request_kwargs = {
            "model": self.config.model,
            "messages": self._convert_messages(messages),
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }

        # max_tokens: kwargs 优先，其次 config
        max_tokens = kwargs.pop("max_tokens", None) or self.config.max_tokens
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens

        # response_format: 支持 json_object / json_schema 等（OpenAI 兼容协议）
        response_format = kwargs.pop("response_format", self.config.response_format if hasattr(self.config, "response_format") else None)
        if response_format:
            request_kwargs["response_format"] = response_format

        if self.config.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.config.reasoning_effort

        # 其余 kwargs 透传（如 top_p、seed 等）
        request_kwargs.update(kwargs)

        if tools:
            request_kwargs["tools"] = self.convert_tools_to_provider_format(tools)
            if self.config.tool_choice:
                request_kwargs["tool_choice"] = self.config.tool_choice

        return normalize_openai_chat_payload(request_kwargs)

    @staticmethod
    def _uses_responses(request: Dict[str, Any]) -> bool:
        return is_gpt6_model(request.get("model", "")) and bool(
            request.get("tools") or any(
                message.get("tool_calls") or message.get("role") == "tool"
                for message in request.get("messages", [])
            )
        )

    def _responses_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """将统一消息及函数工具转换为 Responses 格式，支持工具结果回传。"""
        result = dict(request)
        inputs = []
        for message in result.pop("messages"):
            if message.get("role") == "tool":
                inputs.append({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content") or "",
                })
                continue
            calls = message.get("tool_calls") or []
            if calls:
                # 同一回合的 reasoning/message/function_call 必须按上游原顺序回传。
                cached = next((self._tool_response_items[call["id"]]
                               for call in calls if call["id"] in self._tool_response_items), [])
                cached_ids = {item["call_id"] for item in cached if item.get("type") == "function_call"}
                if cached and cached_ids == {call["id"] for call in calls}:
                    inputs.extend(cached)
                    continue
            if message.get("content") or not calls:
                inputs.append({"role": message["role"], "content": message.get("content") or ""})
            for call in calls:
                inputs.append({
                    "type": "function_call", "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                })
        result["input"] = inputs
        if "max_completion_tokens" in result:
            result["max_output_tokens"] = result.pop("max_completion_tokens")
        if "reasoning_effort" in result:
            result["reasoning"] = {**result.get("reasoning", {}), "effort": result.pop("reasoning_effort")}
        if "tools" in result:
            result["tools"] = [
                {"type": "function", **tool["function"],
                 "strict": tool["function"].get("strict", False)}
                if tool.get("type") == "function" else tool
                for tool in result["tools"]
            ]
        choice = result.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") == "function":
            result["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        response_format = result.pop("response_format", None)
        if response_format:
            format_type = response_format.get("type")
            format_value = ({"type": "json_schema", **response_format["json_schema"]}
                            if format_type == "json_schema" else response_format)
            result["text"] = {**result.get("text", {}), "format": format_value}
        # 默认无服务端持久化；encrypted_content 供工具后续回合恢复推理上下文。
        result.setdefault("store", False)
        result["include"] = list(dict.fromkeys([*result.get("include", []), "reasoning.encrypted_content"]))
        return normalize_openai_responses_payload(result)

    @staticmethod
    def _response_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else value.model_dump(exclude_none=True)

    def _parse_responses_response(self, response: Any) -> LLMResponse:
        data = self._response_dict(response)
        if data.get("error") or data.get("status") in {"failed", "cancelled"}:
            raise ValueError("OpenAI Responses 请求失败，请检查模型配置或服务状态")
        output = data.get("output") or []
        content = []
        calls = []
        for item in output:
            if item.get("type") == "message":
                content.extend(part.get("text", "") for part in item.get("content", [])
                               if part.get("type") == "output_text")
            elif item.get("type") == "function_call":
                calls.append(ToolCall(id=item["call_id"], name=item["name"],
                                      arguments=json.loads(item.get("arguments") or "{}")))
                self._tool_response_items[item["call_id"]] = output
                self._tool_response_items.move_to_end(item["call_id"])
        while len(self._tool_response_items) > 256:
            self._tool_response_items.popitem(last=False)
        usage = data.get("usage")
        return LLMResponse(
            content="".join(content) or None,
            tool_calls=calls,
            finish_reason="length" if data.get("status") == "incomplete" else ("tool_calls" if calls else "stop"),
            model=data.get("model", self.config.model), provider=self.config.provider,
            usage={"prompt_tokens": usage.get("input_tokens", 0),
                   "completion_tokens": usage.get("output_tokens", 0),
                   "total_tokens": usage.get("total_tokens", 0)} if usage else None,
        )

    def _responses_stream_delta(self, event: Any) -> Optional[Dict[str, str]]:
        data = self._response_dict(event)
        event_type = data.get("type")
        if event_type == "response.output_text.delta":
            return {"type": "content", "text": data.get("delta", "")}
        if event_type == "response.reasoning_summary_text.delta":
            return {"type": "reasoning", "text": data.get("delta", "")}
        if event_type in {"response.completed", "response.incomplete"}:
            self._parse_responses_response(data["response"])
        elif event_type in {"error", "response.failed"}:
            raise ValueError("OpenAI Responses 流式请求失败，请检查模型配置或服务状态")
        return None
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """同步聊天"""
        if not self._sync_client:
            self.initialize()
        
        request_kwargs = self._build_request_kwargs(messages, tools, **kwargs)
        if self._uses_responses(request_kwargs):
            response = self._sync_client.responses.create(**self._responses_request(request_kwargs))
            return self._parse_responses_response(response)
        response = self._sync_client.chat.completions.create(**request_kwargs)
        
        return self._parse_response(response)
    
    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> LLMResponse:
        """异步聊天"""
        if not self._async_client:
            self.initialize()
        
        request_kwargs = self._build_request_kwargs(messages, tools, **kwargs)
        if self._uses_responses(request_kwargs):
            response = await self._async_client.responses.create(**self._responses_request(request_kwargs))
            return self._parse_responses_response(response)
        response = await self._async_client.chat.completions.create(**request_kwargs)
        
        return self._parse_response(response)
    
    async def astream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式输出"""
        if not self._async_client:
            self.initialize()
        
        request_kwargs = self._build_request_kwargs(messages, tools, **kwargs)
        request_kwargs["stream"] = True
        if self._uses_responses(request_kwargs):
            stream = await self._async_client.responses.create(**self._responses_request(request_kwargs))
            try:
                async for event in stream:
                    delta = self._responses_stream_delta(event)
                    if delta and delta["type"] == "content":
                        yield delta["text"]
            finally:
                await stream.close()
            return
        
        async for chunk in await self._async_client.chat.completions.create(**request_kwargs):
            # 安全访问 delta：推理模型流式时可能先输出 reasoning_content 再输出 content
            if chunk.choices:
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta and getattr(delta, 'content', None):
                    yield delta.content
    
    def chat_stream_deltas(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """同步流式调用，逐段产出增量字典。

        每个增量形如 {"type": "reasoning"|"content", "text": str}：
        - reasoning: 思考模型的 reasoning_content 增量（如 DeepSeek R1）
        - content: 正文增量

        仅适用于无工具调用的单轮场景（如 Skill 代码生成）。
        调用方负责拼接 content 得到完整回复。
        """
        if not self._sync_client:
            self.initialize()

        request_kwargs = self._build_request_kwargs(messages, tools, **kwargs)
        request_kwargs["stream"] = True

        if self._uses_responses(request_kwargs):
            stream = self._sync_client.responses.create(**self._responses_request(request_kwargs))
            try:
                for event in stream:
                    delta = self._responses_stream_delta(event)
                    if delta:
                        yield delta
            finally:
                stream.close()
            return

        stream = self._sync_client.chat.completions.create(**request_kwargs)
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            if not delta:
                continue
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
            if getattr(delta, "content", None):
                yield {"type": "content", "text": delta.content}

    def _parse_response(self, response) -> LLMResponse:
        """解析响应"""
        # 🔥 检查 choices 字段是否存在且不为空
        if not hasattr(response, 'choices') or response.choices is None:
            error_msg = f"LLM API 返回的响应中 choices 字段为 null。响应类型: {type(response).__name__}"
            if hasattr(response, 'model'):
                error_msg += f", 模型: {response.model}"
            if hasattr(response, 'id'):
                error_msg += f", 响应ID: {response.id}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not response.choices or len(response.choices) == 0:
            error_msg = f"LLM API 返回的响应中 choices 字段为空列表。响应类型: {type(response).__name__}"
            if hasattr(response, 'model'):
                error_msg += f", 模型: {response.model}"
            if hasattr(response, 'id'):
                error_msg += f", 响应ID: {response.id}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        choice = response.choices[0]
        
        # 🔥 检查 choice 是否有 message 属性
        if not hasattr(choice, 'message') or choice.message is None:
            error_msg = f"LLM API 返回的 choice 中 message 字段为 null。响应类型: {type(response).__name__}"
            if hasattr(response, 'model'):
                error_msg += f", 模型: {response.model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        message = choice.message

        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                # 🔥 修复：处理不同类型的 tool_call 对象
                if isinstance(tc, dict):
                    # 已经是字典，直接使用
                    tool_calls.append(ToolCall.from_openai_format(tc))
                elif isinstance(tc, str):
                    # 字符串类型，跳过（不应该出现，但做容错处理）
                    continue
                elif hasattr(tc, 'model_dump'):
                    # Pydantic 模型，调用 model_dump()
                    tool_calls.append(ToolCall.from_openai_format(tc.model_dump()))
                else:
                    # 其他类型，尝试转换为字典
                    try:
                        tool_calls.append(ToolCall.from_openai_format(dict(tc)))
                    except Exception:
                        # 转换失败，跳过
                        continue

        # 推理模型（DeepSeek R1、o1 等）可能返回 reasoning_content，content 为主答案
        # 当 content 为空时用 reasoning_content 作为 fallback，避免下游解析失败
        raw_content = message.content if hasattr(message, 'content') else None
        if raw_content is None or (isinstance(raw_content, str) and not raw_content.strip()):
            raw_content = getattr(message, 'reasoning_content', None)
        content = raw_content if raw_content else None
        reasoning_content = getattr(message, 'reasoning_content', None)

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason if hasattr(choice, 'finish_reason') else None,
            model=response.model if hasattr(response, 'model') else None,
            provider=self.config.provider,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if hasattr(response, 'usage') and response.usage else None
        )

