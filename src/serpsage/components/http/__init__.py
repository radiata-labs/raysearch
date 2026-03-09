from __future__ import annotations

from typing import Any

from serpsage.components.http.base import HttpClientBase


def build_http_client(
    *,
    rt: Any,
    overrides: Any | None = None,
) -> HttpClientBase:
    _ = overrides
    return rt.components.resolve_default("http", expected_type=HttpClientBase)


__all__ = ["HttpClientBase", "build_http_client"]
