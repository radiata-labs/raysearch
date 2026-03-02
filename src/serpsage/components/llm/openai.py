from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar, overload
from typing_extensions import override

import openai
from openai import AsyncOpenAI
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
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )
    from openai.types.completion_usage import CompletionUsage
    from openai.types.shared_params.response_format_json_object import (
        ResponseFormatJSONObject,
    )
    from openai.types.shared_params.response_format_json_schema import (
        ResponseFormatJSONSchema,
    )

    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import LLMModelSettings

TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIClient(LLMClientBase):
    def __init__(
        self, *, rt: Runtime, http: HttpClientBase, model_cfg: LLMModelSettings
    ) -> None:
        super().__init__(rt=rt)
        self.bind_deps(http)
        self._model_cfg = model_cfg
        llm = self._model_cfg
        self.client = AsyncOpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=float(llm.timeout_s),
            max_retries=int(llm.max_retries),
            default_headers=dict(llm.headers or {}),
            http_client=http.client,
        )

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        timeout_s: float | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
    ) -> ChatModelResult[TModel]: ...

    @override
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        llm = self._model_cfg
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")

        response_schema: dict[str, object] | None = None
        response_model: type[BaseModel] | None = None
        if response_format is None:
            response_schema = None
        elif isinstance(response_format, dict):
            response_schema = dict(response_format)
        elif isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            response_model = response_format
            response_schema = response_model.model_json_schema()
        else:
            raise TypeError(
                "response_format must be dict[str, object] | type[BaseModel] | None"
            )

        request_messages = _to_openai_messages(messages)
        request_timeout = float(timeout_s or llm.timeout_s)
        request_temp = float(llm.temperature)
        response_format_payload = _build_response_format_payload(
            schema=response_schema,
            strict=bool(llm.schema_strict),
        )

        try:
            resp = await self._create_completion(
                model=model,
                messages=request_messages,
                temperature=request_temp,
                timeout=request_timeout,
                response_format=response_format_payload,
            )
        except Exception as exc:  # noqa: BLE001
            if (
                response_schema is not None
                and llm.schema_strict
                and _looks_like_schema_error(exc)
            ):
                resp = await self._create_completion(
                    model=model,
                    messages=request_messages,
                    temperature=request_temp,
                    timeout=request_timeout,
                    response_format={"type": "json_object"},
                )
            else:
                raise

        usage_out = _to_usage(getattr(resp, "usage", None))
        content = _extract_text(resp)
        if not isinstance(content, str):
            raise TypeError("LLM response content is not a string")

        if response_schema is None:
            return ChatTextResult(text=content, usage=usage_out)

        data = _try_parse_json_object(content)
        if response_model is not None:
            model_data = response_model.model_validate(data)
            return ChatModelResult(text=content, data=model_data, usage=usage_out)
        return ChatDictResult(text=content, data=data, usage=usage_out)

    async def _create_completion(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        timeout: float,
        response_format: ResponseFormatJSONSchema | ResponseFormatJSONObject | None,
    ) -> ChatCompletion:
        if response_format is None:
            return await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
        )


def _to_openai_messages(
    messages: list[dict[str, str]],
) -> list[ChatCompletionMessageParam]:
    out: list[ChatCompletionMessageParam] = []
    for msg in messages:
        role = str(msg.get("role") or "user").strip().lower()
        content = str(msg.get("content") or "")
        if role == "system":
            out.append({"role": "system", "content": content})
            continue
        if role == "assistant":
            out.append({"role": "assistant", "content": content})
            continue
        out.append({"role": "user", "content": content})
    if not out:
        out.append({"role": "user", "content": ""})
    return out


def _build_response_format_payload(
    *,
    schema: dict[str, object] | None,
    strict: bool,
) -> ResponseFormatJSONSchema | ResponseFormatJSONObject | None:
    if schema is None:
        return None
    if strict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "SerpSageOverview",
                "schema": schema,
                "strict": True,
            },
        }
    return {"type": "json_object"}


def _extract_text(resp: ChatCompletion) -> str:
    choices = list(getattr(resp, "choices", []) or [])
    if not choices:
        return ""
    head = choices[0]
    message = getattr(head, "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return str(content or "")


def _to_usage(usage: CompletionUsage | None) -> LLMUsage:
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
    )


def _try_parse_json_object(content: str) -> dict[str, object]:
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


def _looks_like_schema_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    if "Invalid schema for response_format" in msg:
        return True
    if "additionalProperties" in msg and "response_format" in msg:
        return True
    if isinstance(exc, openai.BadRequestError):
        param = getattr(exc, "param", None)
        if param == "response_format":
            return True
        body = getattr(exc, "body", None)
        if _is_response_format_error_body(body):
            return True
    return False


def _is_response_format_error_body(body: object) -> bool:
    if not isinstance(body, dict):
        return False
    raw_error = body.get("error")
    if not isinstance(raw_error, dict):
        return False
    raw_param = raw_error.get("param")
    return isinstance(raw_param, str) and raw_param == "response_format"


__all__ = ["OpenAIClient"]
