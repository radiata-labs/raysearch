from __future__ import annotations

from typing_extensions import override

import httpx

from serpsage.components.http.base import HttpClientBase, HttpClientConfig


class HttpClient(HttpClientBase[HttpClientConfig]):
    def __init__(
        self,
    ) -> None:
        registry = self.require_components()
        override_client = registry.http_override()
        if isinstance(override_client, httpx.AsyncClient):
            self._client = override_client
            self._owns_client = False
            return
        limits = httpx.Limits(
            max_connections=int(self.config.max_connections),
            max_keepalive_connections=int(self.config.max_keepalive_connections),
            keepalive_expiry=float(self.config.keepalive_expiry_s),
        )
        self._client = httpx.AsyncClient(
            proxy=self.config.proxy,
            trust_env=bool(self.config.trust_env),
            limits=limits,
        )
        self._owns_client = True

    @property
    @override
    def client(self) -> httpx.AsyncClient:
        return self._client

    @override
    async def on_close(self) -> None:
        if not self._owns_client:
            return
        await self._client.aclose()


__all__ = ["HttpClient"]
