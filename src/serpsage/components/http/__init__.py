from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from serpsage.components.http.client import HttpClient

if TYPE_CHECKING:
    from serpsage.core.runtime import Overrides, Runtime


def build_http_client(*, rt: Runtime, ov: Overrides) -> HttpClient:
    if ov.http is not None:
        return HttpClient(rt=rt, client=ov.http, owns_client=False)

    cfg = rt.settings.http
    limits = httpx.Limits(
        max_connections=int(cfg.max_connections),
        max_keepalive_connections=int(cfg.max_keepalive_connections),
        keepalive_expiry=float(cfg.keepalive_expiry_s),
    )
    client = httpx.AsyncClient(
        proxy=cfg.proxy,
        trust_env=bool(cfg.trust_env),
        limits=limits,
    )
    return HttpClient(rt=rt, client=client, owns_client=True)


__all__ = ["HttpClient", "build_http_client"]
