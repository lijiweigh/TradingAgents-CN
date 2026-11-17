"""OpenAI 模型参数兼容规则，供配置测试和运行时请求共用。"""

import re
from typing import Any


def is_gpt6_model(model: str) -> bool:
    """兼容 OpenRouter 的 openai/ 前缀，不匹配其他厂家的同名片段。"""
    return bool(re.match(r"^(?:openai/)?gpt-6(?:[.-])", str(model or "").strip(), re.I))


def _reasoning_effort(model: str, effort: str | None) -> str:
    model = str(model).lower().removeprefix("openai/")
    if not effort or effort == "minimal":
        return "low"
    if effort == "none" and (
        model.startswith("gpt-6-astra") or model.startswith("gpt-6.1-sol")
    ):
        return "low"
    return effort


def normalize_openai_chat_payload(payload: dict[str, Any], *, for_test: bool = False) -> dict[str, Any]:
    """返回副本；保持用户模型/预算及其他厂家参数，仅适配 GPT-6。"""
    result = dict(payload)
    model = result.get("model", "")
    if not is_gpt6_model(model):
        return result

    legacy_limit = result.pop("max_tokens", None)
    if result.get("max_completion_tokens") is None and legacy_limit is not None:
        result["max_completion_tokens"] = legacy_limit
    if for_test:
        result["max_completion_tokens"] = max(result.get("max_completion_tokens") or 0, 1024)

    effort = _reasoning_effort(model, result.get("reasoning_effort"))
    result["reasoning_effort"] = effort
    if effort != "none":
        for key in ("temperature", "top_p", "logprobs", "top_logprobs"):
            result.pop(key, None)
    # extra_body 会在 SDK 中覆盖顶层参数，也必须应用同一兼容规则。
    if isinstance(result.get("extra_body"), dict):
        extra = dict(result.pop("extra_body"))
        parameter_keys = {"max_tokens", "max_completion_tokens", "reasoning_effort",
                          "temperature", "top_p", "logprobs", "top_logprobs"}
        overrides = {key: extra.pop(key) for key in list(extra) if key in parameter_keys}
        if "max_tokens" in overrides and "max_completion_tokens" not in overrides:
            overrides["max_completion_tokens"] = overrides.pop("max_tokens")
        result = normalize_openai_chat_payload({**result, **overrides}, for_test=for_test)
        if extra:
            result["extra_body"] = extra
    return result


def normalize_openai_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """适配 LangChain 已转换为 Responses 格式的最终请求。"""
    if not is_gpt6_model(payload.get("model", "")):
        return dict(payload)
    result = dict(payload)
    reasoning = dict(result.pop("reasoning", None) or {})
    if reasoning.get("effort") is not None:
        result["reasoning_effort"] = reasoning["effort"]
    result = normalize_openai_chat_payload(result)
    reasoning["effort"] = result.pop("reasoning_effort")
    result["reasoning"] = reasoning
    completion_limit = result.pop("max_completion_tokens", None)
    if result.get("max_output_tokens") is None and completion_limit is not None:
        result["max_output_tokens"] = completion_limit
    if reasoning["effort"] != "none" and isinstance(result.get("include"), list):
        result["include"] = [item for item in result["include"] if item != "message.output_text.logprobs"]
    return result


def safe_openai_error(value: Any, api_key: str | None = None) -> str:
    message = str(value)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)[:500]


def openai_error_detail(response: Any, api_key: str | None = None) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return safe_openai_error(error.get("message") or error.get("code") or "", api_key)
    return ""
