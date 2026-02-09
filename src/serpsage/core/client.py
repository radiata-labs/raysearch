"""SearxNG HTTP clients (sync + async).

This module provides two small clients that only fetch *raw* SearxNG JSON:

- :class:`SearxngClient`: sync HTTP via ``requests``.
- :class:`AsyncSearxngClient`: async HTTP via ``httpx`` (anyio backend).

All result processing/ranking/rendering is implemented in :class:`search_core.searcher.Searcher`
and :class:`search_core.searcher.AsyncSearcher`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import httpx
import requests

from serpsage.core.config import DEFAULT_BASE_URL, SearchConfig

if TYPE_CHECKING:
    from collections.abc import Mapping


class _SearxngClientBase:
    """Shared request construction and validation for SearxNG clients."""

    config: SearchConfig

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def _validate_query(self, query: str) -> None:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

    def _validate_api_key_requirement(self) -> None:
        if (
            self.config.searxng.base_url == DEFAULT_BASE_URL
            and not self.config.searxng.search_api_key
        ):
            raise ValueError(
                "SEARCH_API_KEY is required when using the default base_url."
            )

    @staticmethod
    def _build_payload(
        query: str, params: Mapping[str, object] | None
    ) -> dict[str, str]:
        payload: dict[str, str] = {"q": query, "format": "json"}
        if params:
            payload.update({k: str(v) for k, v in params.items()})
        return payload


class SearxngClient(_SearxngClientBase):
    """Sync HTTP client for SearxNG search.

    This client only fetches raw results. Post-processing lives in SearchPipeline.
    """

    session: requests.Session

    def __init__(
        self,
        config: SearchConfig,
        session: requests.Session | None = None,
    ) -> None:
        super().__init__(config)
        self.session = session or requests.Session()

    def search(self, query: str, *, params: Mapping[str, object] | None = None) -> dict:
        """Search via SearxNG and return raw JSON."""

        self._validate_query(query)
        self._validate_api_key_requirement()
        payload = self._build_payload(query, params)

        response = self.session.get(
            self.config.searxng.base_url,
            params=payload,
            headers=self.config.searxng.build_headers(),
            timeout=self.config.searxng.timeout,
            allow_redirects=self.config.searxng.allow_redirects,
        )

        if response.is_redirect or response.status_code in {301, 302, 303, 307, 308}:
            raise RuntimeError(
                f"Redirected to: {response.headers.get('location')} (access not passed)"
            )

        response.raise_for_status()
        return response.json()


class AsyncSearxngClient(_SearxngClientBase):
    """Async HTTP client for SearxNG search (httpx/anyio)."""

    def __init__(
        self,
        config: SearchConfig,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._async_client: httpx.AsyncClient | None = async_client
        self._owns_async_client = async_client is None

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
        return self._async_client

    async def asearch(
        self, query: str, *, params: Mapping[str, object] | None = None
    ) -> dict:
        """Async search via SearxNG and return raw JSON."""

        self._validate_query(query)
        self._validate_api_key_requirement()
        payload = self._build_payload(query, params)

        allow_redirects = bool(self.config.searxng.allow_redirects)
        resp = await self._get_async_client().get(
            self.config.searxng.base_url,
            params=payload,
            headers=self.config.searxng.build_headers(),
            timeout=self.config.searxng.timeout,
            follow_redirects=allow_redirects,
        )

        if not allow_redirects and (
            resp.is_redirect
            or resp.status_code in {301, 302, 303, 307, 308}
            or (300 <= int(resp.status_code) < 400)
        ):
            raise RuntimeError(
                f"Redirected to: {resp.headers.get('location')} (access not passed)"
            )

        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        """Close the internally-owned ``httpx.AsyncClient`` if present."""
        if self._async_client is None:
            return
        if not self._owns_async_client:
            return
        await self._async_client.aclose()
        self._async_client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()


__all__ = ["SearxngClient", "AsyncSearxngClient"]
