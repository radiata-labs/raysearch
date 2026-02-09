from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

import httpx

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import Cache, SearchProvider

if TYPE_CHECKING:
    from collections.abc import Mapping


class SearxngProvider(WorkUnit, SearchProvider):
    def __init__(
        self,
        *,
        rt,  # noqa: ANN001
        http: httpx.AsyncClient,
        cache: Cache,
    ) -> None:
        super().__init__(rt=rt)
        self._http = http
        self._cache = cache

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
        headers.setdefault("User-Agent", self.settings.enrich.fetch.user_agent)
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
