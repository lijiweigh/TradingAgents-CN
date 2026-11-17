"""共享的 LangChain OpenAI 模型兼容层。"""

from langchain_openai import ChatOpenAI as BaseChatOpenAI
from langchain_core.messages import AIMessage

from core.utils.openai_compat import (
    is_gpt6_model,
    normalize_openai_chat_payload,
    normalize_openai_responses_payload,
)


class ChatOpenAI(BaseChatOpenAI):
    """在最终出站请求归一化，覆盖构造参数和 invoke/bind 的覆盖参数。"""

    def _use_responses_api(self, payload: dict) -> bool:
        # GPT-6.1 Sol/Astra 的工具调用必须走 Responses；历史工具消息也需一致。
        if is_gpt6_model(payload.get("model") or self.model_name):
            return True
        return super()._use_responses_api(payload)

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        # 业务层使用字符串报告；重放消息时还原完整 Responses 内容块及顺序。
        messages = self._convert_input(input_).to_messages()
        input_ = [
            message.model_copy(update={
                "content": message.additional_kwargs["openai_response_content"],
                "additional_kwargs": {k: v for k, v in message.additional_kwargs.items()
                                      if k != "openai_response_content"},
            })
            if isinstance(message, AIMessage) and "openai_response_content" in message.additional_kwargs
            else message
            for message in messages
        ]
        # 避免 Responses 转换时构造参数的 token 别名覆盖 invoke 的新预算。
        if is_gpt6_model(kwargs.get("model") or self.model_name) and "max_tokens" in kwargs:
            kwargs.setdefault("max_completion_tokens", kwargs.pop("max_tokens"))
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self._use_responses_api(payload):
            return normalize_openai_responses_payload(payload)
        return normalize_openai_chat_payload(payload)

    @staticmethod
    def _text_message(message):
        if isinstance(message, AIMessage) and isinstance(message.content, list):
            return message.model_copy(update={
                "content": str(message.text),
                "additional_kwargs": {**message.additional_kwargs,
                                      "openai_response_content": message.content},
            })
        return message

    def _text_result(self, result, kwargs):
        if is_gpt6_model(kwargs.get("model") or self.model_name):
            for generation in result.generations:
                generation.message = self._text_message(generation.message)
        return result

    def _generate(self, *args, **kwargs):
        return self._text_result(super()._generate(*args, **kwargs), kwargs)

    async def _agenerate(self, *args, **kwargs):
        return self._text_result(await super()._agenerate(*args, **kwargs), kwargs)

    def _stream(self, *args, **kwargs):
        for chunk in super()._stream(*args, **kwargs):
            if is_gpt6_model(kwargs.get("model") or self.model_name):
                chunk.message = self._text_message(chunk.message)
            yield chunk

    async def _astream(self, *args, **kwargs):
        async for chunk in super()._astream(*args, **kwargs):
            if is_gpt6_model(kwargs.get("model") or self.model_name):
                chunk.message = self._text_message(chunk.message)
            yield chunk
