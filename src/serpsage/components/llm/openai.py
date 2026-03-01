from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import openai
from openai import AsyncOpenAI

from serpsage.components.llm.base import LLMClientBase
from serpsage.models.llm import ChatResult, LLMUsage

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import OverviewModelSettings

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

        req: dict[str, Any] = {
            "model": model,
            "messages": cast("Any", messages),
            "temperature": float(llm.temperature),
            "timeout": float(timeout_s or llm.timeout_s),
        }
        if schema is not None:
            if llm.schema_strict:
                req["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SerpSageOverview",
                        "schema": schema,
                        "strict": True,
                    },
                }
            else:
                req["response_format"] = {"type": "json_object"}

        try:
            resp = await self.client.chat.completions.create(**req)
        except Exception as exc:  # noqa: BLE001
            if schema is not None and llm.schema_strict and _looks_like_schema_error(exc):
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

        data: object | None = None
        if schema is not None:
            data = _try_parse_json(content)
        return ChatResult(text=content, data=data, usage=usage_out)

def _try_parse_json(content: str) -> object:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            return json.loads(content[start : end + 1])
        raise

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
