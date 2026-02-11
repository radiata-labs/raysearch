from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import openai
from openai import AsyncOpenAI

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.llm import ChatJSONResult, LLMUsage
from serpsage.contracts.protocols import LLMClient

if TYPE_CHECKING:
    import httpx
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from serpsage.app.runtime import CoreRuntime


class OpenAIClient(WorkUnit, LLMClient):
    """LLMClient implemented via the official OpenAI Python SDK (async).

    Notes:
    - Reuses the injected `httpx.AsyncClient` for connection pooling.
    - Keeps the narrow `LLMClient.chat_json()` interface for easy swapping with
      other providers / local models.
    """

    def __init__(self, *, rt: CoreRuntime, http: httpx.AsyncClient) -> None:
        super().__init__(rt=rt)
        llm = self.settings.overview.llm
        # AsyncOpenAI is cheap, but we keep a single instance to reuse config.
        self.client = AsyncOpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=float(llm.timeout_s),
            max_retries=int(llm.max_retries),
            default_headers=dict(llm.headers or {}),
            http_client=http,
        )

    @override
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatJSONResult:
        llm = self.settings.overview.llm
        if not llm.api_key:
            raise RuntimeError("missing LLM api_key")

        response_format: dict[str, Any]
        if self.settings.overview.schema_strict:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "SerpSageOverview",
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}

        with self.span("llm.openai.chat_json", model=model) as sp:
            sp.set_attr("schema_strict", bool(self.settings.overview.schema_strict))
            sp.set_attr("timeout_s", float(timeout_s or llm.timeout_s))

            try:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=cast("Any", messages),
                    temperature=float(llm.temperature),
                    response_format=cast("Any", response_format),
                    timeout=float(timeout_s or llm.timeout_s),
                )
            except Exception as exc:  # noqa: BLE001
                # Some OpenAI-compatible providers reject strict schema validation.
                # If so, degrade to `json_object` so the pipeline can still validate
                # output with Pydantic (and self-heal in OverviewStep if needed).
                if self.settings.overview.schema_strict and _looks_like_schema_error(
                    exc
                ):
                    sp.add_event("llm.openai.schema_rejected_fallback_json_object")
                    resp = await self.client.chat.completions.create(
                        model=model,
                        messages=cast("Any", messages),
                        temperature=float(llm.temperature),
                        response_format=cast("Any", {"type": "json_object"}),
                        timeout=float(timeout_s or llm.timeout_s),
                    )
                else:
                    raise

            usage = cast("CompletionUsage | None", getattr(resp, "usage", None))
            usage_out = LLMUsage()
            if usage is not None:
                # Best-effort: usage shape may differ across providers.
                u: dict[str, int] = {}
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(usage, k, None)
                    if v is not None:
                        sp.set_attr(f"usage_{k}", int(v))
                        u[k] = int(v)
                usage_out = LLMUsage(
                    prompt_tokens=int(u.get("prompt_tokens", 0)),
                    completion_tokens=int(u.get("completion_tokens", 0)),
                    total_tokens=int(u.get("total_tokens", 0)),
                )

            content = ""
            choices = cast("list[Choice]", getattr(resp, "choices", None) or [])
            sp.set_attr("choices_count", int(len(choices)))
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", "") or ""
            if not isinstance(content, str):
                raise TypeError("LLM response content is not a string")
            sp.set_attr("response_chars", int(len(content)))

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}")
                if 0 <= start < end:
                    sp.add_event("llm.openai.json_salvage")
                    data = json.loads(content[start : end + 1])
                else:
                    raise

            if not isinstance(data, dict):
                raise TypeError("LLM JSON output is not an object")
            return ChatJSONResult(data=data, usage=usage_out)


def _looks_like_schema_error(exc: Exception) -> bool:
    msg = str(exc) or ""
    if "Invalid schema for response_format" in msg:
        return True
    if "additionalProperties" in msg and "response_format" in msg:
        return True

    # Official SDK error types.
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
