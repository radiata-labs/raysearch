from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload
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
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import OverviewModelSettings

TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIClient(LLMClientBase):
    def __init__(
        self, *, rt: Runtime, http: HttpClientBase, model_cfg: OverviewModelSettings
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

        req: dict[str, Any] = {
            "model": model,
            "messages": cast("Any", messages),
            "temperature": float(llm.temperature),
            "timeout": float(timeout_s or llm.timeout_s),
        }
        if response_schema is not None:
            if llm.schema_strict:
                req["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SerpSageOverview",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            else:
                req["response_format"] = {"type": "json_object"}

        try:
            resp = await self.client.chat.completions.create(**req)
        except Exception as exc:  # noqa: BLE001
            if (
                response_schema is not None
                and llm.schema_strict
                and _looks_like_schema_error(exc)
            ):
                req["response_format"] = {"type": "json_object"}
                resp = await self.client.chat.completions.create(**req)
            else:
                raise

        usage = cast("CompletionUsage | None", getattr(resp, "usage", None))
        usage_out = LLMUsage()
        if usage is not None:
            usage_out = LLMUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            )
        content = ""
        choices = cast("list[Choice]", getattr(resp, "choices", None) or [])
        if choices:
            msg = getattr(choices[0], "message", None)
            content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            raise TypeError("LLM response content is not a string")

        if response_schema is None:
            return ChatTextResult(text=content, usage=usage_out)

        data = _try_parse_json_object(content)
        if response_model is not None:
            model_data = response_model.model_validate(data)
            return ChatModelResult(text=content, data=model_data, usage=usage_out)
        return ChatDictResult(text=content, data=data, usage=usage_out)


def _try_parse_json_object(content: str) -> dict[str, Any]:
    payload: Any
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
    return payload


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
        if (
            isinstance(body, dict)
            and body.get("error", {}).get("param") == "response_format"
        ):
            return True
    return False


__all__ = ["OpenAIClient"]
