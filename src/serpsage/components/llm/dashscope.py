from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from dashscope import Generation
from dashscope.api_entities.dashscope_response import Message

from serpsage.components.llm.base import LLMClientBase
from serpsage.models.llm import ChatResult, LLMUsage

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import OverviewModelSettings

class DashScopeClient(LLMClientBase):
    def __init__(self, *, rt: Runtime, model_cfg: OverviewModelSettings) -> None:
        super().__init__(rt=rt)
        self._model_cfg = model_cfg
        self._api_key = model_cfg.api_key

    @override
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        llm = self._model_cfg
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")

        timeout_ms = int(float(timeout_s or llm.timeout_s) * 1000)
        req_messages = _to_dashscope_messages(messages)

        try:
            response = await _async_generation_call(
                model=model,
                messages=req_messages,
                temperature=float(llm.temperature),
                timeout=timeout_ms,
                result_format="message",
                api_key=self._api_key,
            )
        except Exception as exc:
            if schema is not None and llm.schema_strict and _looks_like_schema_error(exc):
                response = await _async_generation_call(
                    model=model,
                    messages=req_messages,
                    temperature=float(llm.temperature),
                    timeout=timeout_ms,
                    result_format="message",
                    api_key=self._api_key,
                )
            else:
                raise

        usage = _to_usage(response)

        text = _get_text_content(response)
        if not isinstance(text, str):
            raise TypeError("LLM response content is not a string")

        data: object | None = None
        if schema is not None:
            data = _extract_json(text)
        return ChatResult(text=text, data=data, usage=usage)

async def _async_generation_call(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
    result_format: str,
    api_key: str | None,
) -> Any:
    """Call DashScope Generation.call in a thread to avoid blocking."""
    # Convert dict messages to Message objects
    message_objects = [
        Message(role=msg["role"], content=msg["content"]) for msg in messages
    ]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: Generation.call(
            model=model,
            messages=message_objects,
            temperature=temperature,
            timeout=timeout,
            result_format=result_format,
            api_key=cast("str", api_key),
        ),
    )

def _to_dashscope_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert messages to DashScope format."""
    result: list[dict[str, str]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"messages[{idx}] must be a dict")
        role = msg.get("role")
        content = msg.get("content")
        if role is None or content is None:
            raise ValueError(f"messages[{idx}] missing 'role' or 'content'")
        role_str = str(role).strip().lower()
        content_str = str(content)
        if role_str == "system":
            role_str = "system"
        elif role_str == "assistant":
            role_str = "assistant"
        else:
            role_str = "user"
        result.append({"role": role_str, "content": content_str})
    return result

def _get_text_content(response: Any) -> str:
    """Extract text content from DashScope response."""
    if response is None:
        return ""

    output = getattr(response, "output", None)
    if output is None:
        return ""

    choices = getattr(output, "choices", None)
    if not choices:
        return ""

    choice = choices[0] if isinstance(choices, list) and len(choices) > 0 else None
    if choice is None:
        return ""

    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    return str(content) if content else ""

def _to_usage(response: Any) -> LLMUsage:
    """Convert DashScope usage to LLMUsage."""
    if response is None:
        return LLMUsage()

    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()

    prompt_tokens = int(
        getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    )
    completion_tokens = int(
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )
    total_tokens = prompt_tokens + completion_tokens

    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

def _looks_like_schema_error(exc: Exception) -> bool:
    """Detect if exception is related to schema validation failure."""
    text = str(exc).lower()
    return "schema" in text or ("json" in text and "valid" in text)

def _extract_json(content: str) -> object:
    """Extract JSON from response content."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            return json.loads(content[start : end + 1])
        raise

__all__ = ["DashScopeClient"]
