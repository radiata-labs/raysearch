from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from openai import AsyncOpenAI

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import LLMClient

if TYPE_CHECKING:
    import httpx

    from serpsage.app.runtime import CoreRuntime


class OpenAIOfficialLLMClient(WorkUnit, LLMClient):
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
            organization=llm.organization,
            project=llm.project,
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
    ) -> dict[str, Any]:
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

        max_out = int(self.settings.overview.max_output_tokens)
        with self.span("llm.openai.chat_json", model=model) as sp:
            sp.set_attr("schema_strict", bool(self.settings.overview.schema_strict))
            sp.set_attr("max_completion_tokens", max_out)
            sp.set_attr("timeout_s", float(timeout_s or llm.timeout_s))

            resp = await self.client.chat.completions.create(
                model=model,
                messages=cast("Any", messages),
                temperature=float(llm.temperature),
                max_completion_tokens=max_out,
                response_format=cast("Any", response_format),
                timeout=float(timeout_s or llm.timeout_s),
            )

            usage = getattr(resp, "usage", None)
            if usage is not None:
                # Best-effort: usage shape may differ across providers.
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(usage, k, None)
                    if v is not None:
                        sp.set_attr(f"usage_{k}", int(v))

            content = ""
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                content = getattr(msg, "content", "") or ""
            if not isinstance(content, str):
                raise TypeError("LLM response content is not a string")

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                start = content.find("{")
                end = content.rfind("}")
                if 0 <= start < end:
                    data = json.loads(content[start : end + 1])
                else:
                    raise

            if not isinstance(data, dict):
                raise TypeError("LLM JSON output is not an object")
            return data


__all__ = ["OpenAIOfficialLLMClient"]
