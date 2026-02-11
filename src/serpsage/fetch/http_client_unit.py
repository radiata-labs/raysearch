from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import httpx

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class HttpClientUnit(WorkUnit):
    def __init__(
        self,
        *,
        rt: Runtime,
        client: httpx.AsyncClient | None = None,
        owns_client: bool | None = None,
    ) -> None:
        super().__init__(rt=rt)
        if client is None:
            self._client = httpx.AsyncClient()
            self._owns_client = True if owns_client is None else bool(owns_client)
            return
        self._client = client
        self._owns_client = False if owns_client is None else bool(owns_client)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    @override
    async def on_close(self) -> None:
        if not self._owns_client:
            return
        await self._client.aclose()


__all__ = ["HttpClientUnit"]
