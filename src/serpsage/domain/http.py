from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import httpx

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.core.runtime import Overrides, Runtime


class HttpClient(WorkUnit):
    def __init__(
        self,
        *,
        rt: Runtime,
        ov: Overrides,
    ) -> None:
        super().__init__(rt=rt)
        if ov.http is not None:
            self._client = ov.http
            self._owns_client = False
        else:
            cfg = rt.settings.http
            limits = httpx.Limits(
                max_connections=int(cfg.max_connections),
                max_keepalive_connections=int(cfg.max_keepalive_connections),
                keepalive_expiry=float(cfg.keepalive_expiry_s),
            )
            self._client = httpx.AsyncClient(
                proxy=cfg.proxy,
                trust_env=bool(cfg.trust_env),
                limits=limits,
            )
            self._owns_client = True

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    @override
    async def on_close(self) -> None:
        if not self._owns_client:
            return
        await self._client.aclose()


__all__ = ["HttpClient"]
