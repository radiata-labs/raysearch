from __future__ import annotations

import json
from typing import Any
from typing_extensions import override

import httpx

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import LLMClient


class NullLLMClient(WorkUnit, LLMClient):
    def __init__(self, *, rt) -> None:  # noqa: ANN001
        super().__init__(rt=rt)

    @override
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("LLM is not configured (missing api_key or disabled).")


class OpenAICompatLLMClient(WorkUnit, LLMClient):
    def __init__(
        self,
        *,
        rt,  # noqa: ANN001
        http: httpx.AsyncClient,
    ) -> None:
        super().__init__(rt=rt)
        self._http = http

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

        url = llm.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {llm.api_key}"}
        headers.update(dict(llm.headers or {}))

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

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": response_format,
        }

        resp = await self._http.post(
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout_s or float(llm.timeout_s)),
        )
        resp.raise_for_status()
        data = resp.json()

        content = (((data.get("choices") or [{}])[0]).get("message") or {}).get(
            "content"
        ) or ""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise TypeError("LLM response content is not a string/dict")

        # Some providers already return JSON in `content`. Parse it.
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object substring.
            start = content.find("{")
            end = content.rfind("}")
            if 0 <= start < end:
                return json.loads(content[start : end + 1])
            raise


__all__ = ["NullLLMClient", "OpenAICompatLLMClient"]
