from __future__ import annotations

from typing import Any, cast

from serpsage.components.http.base import HttpClientBase


def build_http_client(
    *,
    rt: Any,
    overrides: Any | None = None,
) -> HttpClientBase:
    _ = overrides
    return cast("HttpClientBase", rt.services.require(HttpClientBase))


__all__ = ["HttpClientBase", "build_http_client"]
