from __future__ import annotations

from typing_extensions import override

import httpx

from serpsage.components.base import ComponentMeta
from serpsage.components.http.base import HttpClientBase, HttpClientConfig
from serpsage.components.registry import register_component
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject


@register_component(
    meta=ComponentMeta(
        family="http",
        name="httpx",
        version="1.0.0",
        summary="Shared httpx async client.",
        provides=("http.client",),
        config_model=HttpClientConfig,
    )
)
class HttpClient(HttpClientBase):
    meta = ComponentMeta(
        family="http",
        name="httpx",
        version="1.0.0",
        summary="Shared httpx async client.",
        provides=("http.client",),
        config_model=HttpClientConfig,
    )

    def __init__(
        self,
        *,
        rt: Runtime = Inject(),
        config: HttpClientConfig = Inject(),
    ) -> None:
        super().__init__(rt=rt, config=config)
        components = getattr(rt, "components", None)
        override_client = components.http_override() if components is not None else None
        if isinstance(override_client, httpx.AsyncClient):
            self._client = override_client
            self._owns_client = False
            return
        limits = httpx.Limits(
            max_connections=int(config.max_connections),
            max_keepalive_connections=int(config.max_keepalive_connections),
            keepalive_expiry=float(config.keepalive_expiry_s),
        )
        self._client = httpx.AsyncClient(
            proxy=config.proxy,
            trust_env=bool(config.trust_env),
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
