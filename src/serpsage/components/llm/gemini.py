from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar, overload
from typing_extensions import override

from google import genai
from google.genai import errors, types
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
    from serpsage.settings.models import OverviewModelSettings

TModel = TypeVar("TModel", bound=BaseModel)

class GeminiClient(LLMClientBase):
    def __init__(self, *, rt: Runtime, model_cfg: OverviewModelSettings) -> None:
        super().__init__(rt=rt)
        self._model_cfg = model_cfg

        llm = self._model_cfg
        timeout_ms = max(1, int(float(llm.timeout_s) * 1000))
        attempts = max(1, int(llm.max_retries) + 1)
        self.client = genai.Client(
            api_key=llm.api_key,
            http_options=types.HttpOptions(
                base_url=llm.base_url,
                headers=dict(llm.headers or {}),
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=attempts),
            ),
        )

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
    ) -> ChatModelResult[TModel]: ...

    @override
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        llm = self._model_cfg
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")

        response_schema: dict[str, Any] | None = None
        response_model: type[BaseModel] | None = None
        if response_format is None:
            response_schema = None
        elif isinstance(response_format, dict):
            response_schema = dict(response_format)
        elif isinstance(response_format, type) and issubclass(response_format, BaseModel):
            response_model = response_format
            response_schema = response_model.model_json_schema()
        else:
            raise TypeError(
                "response_format must be dict[str, Any] | type[BaseModel] | None"
            )

        system_instruction, contents = _to_gemini_messages(messages)
        timeout_ms = max(1, int(float(timeout_s or llm.timeout_s) * 1000))

        try:
            resp = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=_build_config(
                    system_instruction=system_instruction,
                    temperature=float(llm.temperature),
                    timeout_ms=timeout_ms,
                    schema=response_schema,
                    schema_strict=bool(llm.schema_strict),
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if (
                response_schema is not None
                and llm.schema_strict
                and _looks_like_schema_error(exc)
            ):
                resp = await self.client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=_build_config(
                        system_instruction=system_instruction,
                        temperature=float(llm.temperature),
                        timeout_ms=timeout_ms,
                        schema=response_schema,
                        schema_strict=False,
                    ),
                )
            else:
                raise

        usage = _to_usage(getattr(resp, "usage_metadata", None))

        text = getattr(resp, "text", "") or ""
        if not isinstance(text, str):
            raise TypeError("LLM response content is not a string")
        if response_schema is None:
            return ChatTextResult(text=text, usage=usage)
        data = _extract_json_object(resp=resp, fallback_text=text)
        if response_model is not None:
            model_data = response_model.model_validate(data)
            return ChatModelResult(text=text, data=model_data, usage=usage)
        return ChatDictResult(text=text, data=data, usage=usage)

def _build_config(
    *,
    system_instruction: str | None,
    temperature: float,
    timeout_ms: int,
    schema: dict[str, Any] | None,
    schema_strict: bool,
) -> types.GenerateContentConfig:
    cfg: dict[str, Any] = {
        "temperature": float(temperature),
        "http_options": types.HttpOptions(timeout=int(timeout_ms)),
    }
    if system_instruction:
        cfg["system_instruction"] = system_instruction
    if schema is not None:
        cfg["response_mime_type"] = "application/json"
        if schema_strict:
            cfg["response_json_schema"] = schema
    return types.GenerateContentConfig(**cfg)

def _to_gemini_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        text = str(msg.get("content") or "")
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        gem_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gem_role,
                parts=[types.Part.from_text(text=text)],
            )
        )

    if not contents:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="")],
            )
        )

    system_instruction = None
    if system_parts:
        system_instruction = "\n\n".join(system_parts)
    return system_instruction, contents

def _extract_json_object(
    *, resp: types.GenerateContentResponse, fallback_text: str
) -> dict[str, Any]:
    parsed = getattr(resp, "parsed", None)
    if parsed is not None:
        data: Any = parsed
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            return data

    payload: Any
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
    return payload

def _to_usage(usage_meta: Any) -> LLMUsage:
    if usage_meta is None:
        return LLMUsage()
    prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
    total_tokens = int(
        getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens) or 0
    )
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

def _looks_like_schema_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code is not None:
        try:
            sc = int(code)
        except Exception:  # noqa: BLE001
            sc = None
        if sc is not None and not (400 <= sc < 500):
            return False

    text = str(exc).lower()
    if "schema" in text:
        return True
    if "response_json_schema" in text or "response_schema" in text:
        return True
    if isinstance(exc, (errors.ClientError, errors.APIError)):
        return "invalid" in text and "response" in text
    return False

__all__ = ["GeminiClient"]
