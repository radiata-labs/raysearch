from __future__ import annotations

from typing import Any
from typing_extensions import override

import httpx
from pydantic import field_validator

from serpsage.components.http.base import HttpClientBase
from serpsage.components.provider.base import ProviderConfigBase, SearchProviderBase
from serpsage.dependencies import Depends
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
from serpsage.utils import clean_whitespace

_DEFAULT_SEARXNG_BASE_URL = "https://searx.be/search"


class SearxngProviderConfig(ProviderConfigBase):
    __setting_family__ = "provider"
    __setting_name__ = "searxng"

    base_url: str = _DEFAULT_SEARXNG_BASE_URL
    api_key: str | None = None
    user_agent: str = ""

    @field_validator("api_key")
    @classmethod
    def _normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        token = clean_whitespace(str(value))
        return token or None

    @field_validator("user_agent")
    @classmethod
    def _normalize_user_agent(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @classmethod
    @override
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = dict(raw)
        if env.get("SEARXNG_BASE_URL"):
            payload["base_url"] = env["SEARXNG_BASE_URL"]
        api_key = env.get("SEARXNG_API_KEY")
        if api_key:
            payload["api_key"] = api_key
        cf_id = env.get("SEARXNG_CF_ACCESS_CLIENT_ID")
        cf_secret = env.get("SEARXNG_CF_ACCESS_CLIENT_SECRET")
        if cf_id and cf_secret:
            payload.setdefault("headers", {})
            payload["headers"].setdefault("CF-Access-Client-Id", cf_id)
            payload["headers"].setdefault("CF-Access-Client-Secret", cf_secret)
        return payload


class SearxngProvider(SearchProviderBase[SearxngProviderConfig]):
    http: HttpClientBase = Depends()

    @override
    async def asearch(
        self,
        *,
        query: str,
        page: int = 1,
        language: str = "",
        **kwargs: Any,
    ) -> SearchProviderResponse:
        cfg = self.config
        if not query or not str(query).strip():
            raise ValueError("query must not be empty")
        payload: dict[str, str] = {"q": str(query), "format": "json"}
        if page > 1:
            payload["pageno"] = str(int(page))
        if language:
            payload["language"] = str(language)
        for key, value in kwargs.items():
            if value is None:
                continue
            payload[str(key)] = str(value)
        headers = dict(cfg.headers or {})
        if cfg.user_agent:
            headers.setdefault("User-Agent", str(cfg.user_agent))
        if cfg.api_key:
            headers.setdefault("Authorization", f"Bearer {cfg.api_key}")
        resp = await self.http.client.get(
            cfg.base_url,
            params=payload,
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            follow_redirects=bool(cfg.allow_redirects),
        )
        resp.raise_for_status()
        data = resp.json()
        raw_results = data.get("results", [])
        results: list[SearchProviderResult] = []
        if isinstance(raw_results, list):
            for index, raw in enumerate(raw_results, start=1):
                if not isinstance(raw, dict):
                    continue
                url = clean_whitespace(str(raw.get("url") or ""))
                if not url:
                    continue
                snippet = raw.get("content")
                if snippet is None:
                    snippet = raw.get("description")
                metadata = {
                    key: value
                    for key, value in raw.items()
                    if key not in {"url", "title", "content", "description", "engine"}
                }
                results.append(
                    SearchProviderResult(
                        url=url,
                        title=str(raw.get("title") or ""),
                        snippet=str(snippet or ""),
                        display_url=str(raw.get("pretty_url") or ""),
                        source_engine=str(raw.get("engine") or ""),
                        position=index,
                        metadata=metadata,
                    )
                )
        total_results = data.get("number_of_results")
        return SearchProviderResponse(
            provider_backend="searxng",
            query=str(query),
            page=int(page),
            language=str(language or ""),
            total_results=(
                int(total_results) if isinstance(total_results, (int, float)) else None
            ),
            suggestions=(
                list(data.get("suggestions"))
                if isinstance(data.get("suggestions"), list)
                else []
            ),
            results=results,
            metadata={
                key: value
                for key, value in data.items()
                if key not in {"results", "number_of_results", "suggestions"}
            },
        )


__all__ = ["SearxngProvider", "SearxngProviderConfig"]
