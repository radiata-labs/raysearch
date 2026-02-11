from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

import httpx

from serpsage.components.http import HttpClient
from serpsage.contracts.services import SearchProviderBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from serpsage.core.runtime import Runtime


class SearxngProvider(SearchProviderBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        http: HttpClient,
    ) -> None:
        super().__init__(rt=rt)
        self.bind_deps(http)
        self._http = http.client

    @override
    async def asearch(
        self, *, query: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]:
        se = self.settings.provider.searxng
        if not query or not str(query).strip():
            raise ValueError("query must not be empty")
        # In 3.0 we treat api_key as optional; some instances might not require it.

        payload: dict[str, str] = {"q": str(query), "format": "json"}
        if params:
            payload.update({k: str(v) for k, v in dict(params).items()})

        headers = dict(se.headers or {})
        headers.setdefault("User-Agent", self.settings.enrich.fetch.common.user_agent)
        if se.api_key:
            headers.setdefault("Authorization", f"Bearer {se.api_key}")

        resp = await self._http.get(
            se.base_url,
            params=payload,
            headers=headers,
            timeout=httpx.Timeout(se.timeout_s),
            follow_redirects=bool(se.allow_redirects),
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return list(results if isinstance(results, list) else [])


__all__ = ["SearxngProvider"]
