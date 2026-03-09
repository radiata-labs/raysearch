from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

import httpx

from serpsage.components.provider.base import SearchProviderBase
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.http.base import HttpClientBase
    from serpsage.core.runtime import Runtime


class SearxngProvider(SearchProviderBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        http: HttpClientBase,
    ) -> None:
        super().__init__(rt=rt)
        self.bind_deps(http)
        self._http = http.client

    @override
    async def asearch(
        self,
        *,
        query: str,
        page: int = 1,
        language: str = "",
        **kwargs: Any,
    ) -> SearchProviderResponse:
        provider = self.settings.provider.searxng
        if not query or not str(query).strip():
            raise ValueError("query must not be empty")
        # In 3.0 we treat api_key as optional; some instances might not require it.
        payload: dict[str, str] = {"q": str(query), "format": "json"}
        if page > 1:
            payload["pageno"] = str(int(page))
        if language:
            payload["language"] = str(language)
        for key, value in kwargs.items():
            if value is None:
                continue
            payload[str(key)] = str(value)
        headers = dict(provider.headers or {})
        headers.setdefault(
            "User-Agent",
            str(provider.user_agent or self.settings.fetch.user_agent),
        )
        if provider.api_key:
            headers.setdefault("Authorization", f"Bearer {provider.api_key}")
        resp = await self._http.get(
            provider.base_url,
            params=payload,
            headers=headers,
            timeout=httpx.Timeout(provider.timeout_s),
            follow_redirects=bool(provider.allow_redirects),
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


__all__ = ["SearxngProvider"]
