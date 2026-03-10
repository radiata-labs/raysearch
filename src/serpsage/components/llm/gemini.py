from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar, overload, runtime_checkable
from typing_extensions import override

from google import genai
from google.genai import types
from pydantic import BaseModel

from serpsage.components.base import ComponentMeta
from serpsage.components.llm.base import LLMClientBase, LLMModelConfig
from serpsage.models.components.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
    LLMUsage,
)

TModel = TypeVar("TModel", bound=BaseModel)


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self) -> object: ...


class GeminiModelConfig(LLMModelConfig):
    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        if not payload.get("api_key") and env.get("GEMINI_API_KEY"):
            payload["api_key"] = env["GEMINI_API_KEY"]
        if not payload.get("base_url") and env.get("GEMINI_BASE_URL"):
            payload["base_url"] = env["GEMINI_BASE_URL"]
        return payload


_GEMINI_META = ComponentMeta(
    family="llm",
    name="gemini",
    version="1.0.0",
    summary="Gemini route client.",
    provides=("llm.route",),
    config_model=GeminiModelConfig,
)


class GeminiClient(LLMClientBase[GeminiModelConfig]):
    meta = _GEMINI_META

    def __init__(
        self,
    ) -> None:
        timeout_ms = max(1, int(float(self.config.timeout_s) * 1000))
        attempts = max(1, int(self.config.max_retries) + 1)
        self.client = genai.Client(
            api_key=self.config.api_key,
            http_options=types.HttpOptions(
                base_url=self.config.base_url,
                headers=dict(self.config.headers or {}),
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=attempts),
            ),
        )

    @override
    def describe_model(self, name: str) -> GeminiModelConfig:
        if str(name) != str(self.config.name):
            super().describe_model(name)
            raise AssertionError("unreachable")
        return self.config

    @overload
    async def _create(
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
    async def _create(
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
    async def _create(
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
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        llm = self.config
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")
        response_schema, response_model = self.resolve_response_format(
            response_format,
            format_override=format_override,
        )
        system_instruction, contents = self._to_gemini_messages(
            messages, response_model, llm.enable_structured
        )
        timeout_ms = max(1, int(float(timeout_s or llm.timeout_s) * 1000))
        config = self._build_config(
            system_instruction=system_instruction,
            temperature=float(llm.temperature),
            timeout_ms=timeout_ms,
            schema=response_schema,
            enable_structured=bool(llm.enable_structured),
        )
        config = self._merge_config_kwargs(config=config, kwargs=kwargs)
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = str(getattr(response, "text", "") or "")
        usage = self._to_usage(getattr(response, "usage_metadata", None))
        if response_schema is None:
            return ChatTextResult(text=text, usage=usage)
        data = self._extract_json_object(resp=response, fallback_text=text)
        if response_model is not None:
            return ChatModelResult(
                text=text, data=response_model.model_validate(data), usage=usage
            )
        return ChatDictResult(text=text, data=data, usage=usage)

    @staticmethod
    def _build_config(
        *,
        system_instruction: str | None,
        temperature: float,
        timeout_ms: int,
        schema: dict[str, object] | None,
        enable_structured: bool,
    ) -> types.GenerateContentConfig:
        http_options = types.HttpOptions(timeout=int(timeout_ms))
        if schema is not None and enable_structured:
            return types.GenerateContentConfig(
                temperature=float(temperature),
                http_options=http_options,
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
                response_json_schema=schema,
            )
        if schema is not None:
            return types.GenerateContentConfig(
                temperature=float(temperature),
                http_options=http_options,
                system_instruction=system_instruction or None,
                response_mime_type="application/json",
            )
        return types.GenerateContentConfig(
            temperature=float(temperature),
            http_options=http_options,
            system_instruction=system_instruction or None,
        )

    @staticmethod
    def _merge_config_kwargs(
        *,
        config: types.GenerateContentConfig,
        kwargs: dict[str, Any],
    ) -> types.GenerateContentConfig:
        if not kwargs:
            return config
        dumped = config.model_dump(exclude_none=True)
        merged: dict[str, Any] = {}
        if isinstance(dumped, dict):
            merged.update(dumped)
        merged.update(kwargs)
        return types.GenerateContentConfig(**merged)

    def _to_gemini_messages(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | None = None,
        enable_structured: bool = True,
    ) -> tuple[str | None, list[types.ContentUnion]]:
        system_parts: list[str] = []
        contents: list[types.ContentUnion] = []
        if not enable_structured and response_model is not None:
            structure_prompt = self.get_format_instructions(response_model)
            if structure_prompt:
                system_parts.append(structure_prompt)
        for msg in messages:
            role = str(msg.get("role") or "user").strip().lower()
            text = str(msg.get("content") or "")
            if role == "system":
                if text:
                    system_parts.append(text)
                continue
            gem_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(role=gem_role, parts=[types.Part.from_text(text=text)])
            )
        if not contents:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text="")])
            )
        if not system_parts:
            return None, contents
        return "\n\n".join(system_parts), contents

    @staticmethod
    def _extract_json_object(
        *, resp: types.GenerateContentResponse, fallback_text: str
    ) -> dict[str, object]:
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            data: object = parsed
            if isinstance(data, _ModelDumpable):
                data = data.model_dump()
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return {str(k): v for k, v in data.items()}
        payload: object
        try:
            payload = json.loads(fallback_text)
        except json.JSONDecodeError:
            start = fallback_text.find("{")
            end = fallback_text.rfind("}")
            if 0 <= start < end:
                payload = json.loads(fallback_text[start : end + 1])
            else:
                raise
        if not isinstance(payload, dict):
            raise TypeError("structured LLM response must be a JSON object")
        return {str(k): v for k, v in payload.items()}

    @staticmethod
    def _to_usage(usage_meta: object) -> LLMUsage:
        if usage_meta is None:
            return LLMUsage()
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        total_tokens = int(
            getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens)
            or 0
        )
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


__all__ = ["GeminiClient", "GeminiModelConfig"]
