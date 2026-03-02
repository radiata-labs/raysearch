from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload
from typing_extensions import override

from dashscope.aigc.generation import AioGeneration  # type: ignore[import-untyped]
from dashscope.api_entities.dashscope_response import (  # type: ignore[import-untyped]
    GenerationResponse,
    Message,
)
from pydantic import BaseModel

from serpsage.components.llm.base import LLMClientBase
from serpsage.models.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
    LLMUsage,
)

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import LLMModelSettings
TModel = TypeVar("TModel", bound=BaseModel)


class DashScopeClient(LLMClientBase):
    _SCHEMA_NAME = "SerpSageOverview"

    def __init__(self, *, rt: Runtime, model_cfg: LLMModelSettings) -> None:
        super().__init__(rt=rt)
        self._model_cfg = model_cfg
        self._api_key = model_cfg.api_key

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatTextResult: ...
    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatDictResult: ...
    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatModelResult[TModel]: ...
    @override
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        llm = self._model_cfg
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")
        response_schema, response_model = self.resolve_response_format(
            response_format,
            format_override=format_override,
        )
        request_messages = self._to_dashscope_messages(
            messages,
            response_model,
            llm.enable_structured,
        )
        timeout_ms = int(float(timeout_s or llm.timeout_s) * 1000)
        if response_schema is not None:
            request_messages = self._inject_structure_instruction(
                messages=request_messages,
                schema=response_schema,
                enable_structured=bool(llm.enable_structured),
            )
        response = await self._async_generation_call(
            model=model,
            messages=request_messages,
            temperature=float(llm.temperature),
            timeout=timeout_ms,
            result_format="message",
            **kwargs,
        )
        text = self._extract_text(response)
        usage = self._to_usage(response)
        if response_schema is None:
            return ChatTextResult(text=text, usage=usage)
        data = self._parse_json_object(text)
        if response_model is not None:
            return ChatModelResult(
                text=text,
                data=response_model.model_validate(data),
                usage=usage,
            )
        return ChatDictResult(text=text, data=data, usage=usage)

    async def _async_generation_call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        timeout: int,
        result_format: str,
        **kwargs: Any,
    ) -> GenerationResponse:
        if not self._api_key:
            raise RuntimeError("missing LLM api_key")
        message_objects = [
            Message(role=msg["role"], content=msg["content"]) for msg in messages
        ]
        response = await AioGeneration.call(
            model=model,
            messages=message_objects,
            temperature=temperature,
            timeout=timeout,
            result_format=result_format,
            api_key=self._api_key,
            **kwargs,
        )
        if inspect.isasyncgen(response):
            raise TypeError("DashScope streaming response is not supported")
        return cast("GenerationResponse", response)

    @classmethod
    def _inject_structure_instruction(
        cls,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, object],
        enable_structured: bool,
    ) -> list[dict[str, str]]:
        if enable_structured:
            content = (
                "Return JSON only and strictly validate against this JSON schema:\n"
                f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
            )
        else:
            content = "Return JSON object only."
        return [{"role": "system", "content": content}, *messages]

    def _to_dashscope_messages(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | None = None,
        enable_structured: bool = True,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if not enable_structured and response_model is not None:
            structure_prompt = self.get_format_instructions(response_model)
            if structure_prompt:
                out.append({"role": "system", "content": structure_prompt})
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            if role is None or content is None:
                raise ValueError(f"messages[{idx}] missing 'role' or 'content'")
            normalized_role = str(role).strip().lower()
            if normalized_role not in {"system", "assistant"}:
                normalized_role = "user"
            out.append({"role": normalized_role, "content": str(content)})
        if not out:
            out.append({"role": "user", "content": ""})
        return out

    @staticmethod
    def _extract_text(response: GenerationResponse | None) -> str:
        if response is None:
            return ""
        output = getattr(response, "output", None)
        if output is None:
            return ""
        choices = getattr(output, "choices", None)
        if not choices or not isinstance(choices, list):
            return ""
        first = choices[0] if choices else None
        message = getattr(first, "message", None) if first is not None else None
        content = getattr(message, "content", "") if message is not None else ""
        return str(content or "")

    @staticmethod
    def _to_usage(response: GenerationResponse | None) -> LLMUsage:
        if response is None:
            return LLMUsage()
        usage = getattr(response, "usage", None)
        if usage is None:
            return LLMUsage()
        prompt_tokens = int(
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", 0)
            or 0
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

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, object]:
        payload: object
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if 0 <= start < end:
                payload = json.loads(content[start : end + 1])
            else:
                raise
        if not isinstance(payload, dict):
            raise TypeError("structured LLM response must be a JSON object")
        return {str(k): v for k, v in payload.items()}


__all__ = ["DashScopeClient"]
