from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.http.client import HttpClient

if TYPE_CHECKING:
    from serpsage.contracts.services import HttpClientBase
    from serpsage.core.runtime import Overrides, Runtime


def build_http_client(
    *,
    rt: Runtime,
    overrides: Overrides | None = None,
) -> HttpClientBase:
    return HttpClient(rt=rt, overrides=overrides)


__all__ = ["build_http_client"]
