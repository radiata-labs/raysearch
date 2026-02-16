from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.fetch.base import FetcherBase

if TYPE_CHECKING:
    from serpsage.components.http import HttpClientBase
    from serpsage.components.rate_limit import RateLimiterBase
    from serpsage.core.runtime import Runtime


def build_fetcher(
    *,
    rt: Runtime,
    rate_limiter: RateLimiterBase,
    http: HttpClientBase,
) -> FetcherBase:
    backend = str(rt.settings.fetch.backend or "auto").lower()

    curl_fetcher = None
    if backend in {"auto", "curl_cffi"}:
        from serpsage.components.fetch.curl_cffi import (
            CURL_CFFI_AVAILABLE,
            CurlCffiFetcher,
        )

        if CURL_CFFI_AVAILABLE:
            try:
                curl_fetcher = CurlCffiFetcher(rt=rt)
            except Exception:
                curl_fetcher = None

    playwright_fetcher = None
    if backend in {"auto", "playwright"}:
        from serpsage.components.fetch.playwright import (
            PLAYWRIGHT_AVAILABLE,
            PlaywrightFetcher,
        )

        if PLAYWRIGHT_AVAILABLE:
            try:
                playwright_fetcher = PlaywrightFetcher(rt=rt)
            except Exception:
                playwright_fetcher = None

    if backend == "curl_cffi":
        if curl_fetcher is None:
            raise RuntimeError(
                "fetch backend `curl_cffi` is unavailable: install curl_cffi"
            )
        return curl_fetcher

    if backend == "playwright":
        if playwright_fetcher is None:
            raise RuntimeError(
                "fetch backend `playwright` is unavailable: install playwright and browsers"
            )
        return playwright_fetcher

    if backend == "auto":
        if curl_fetcher is None:
            raise RuntimeError(
                "fetch backend `auto` requires curl_cffi runtime dependencies"
            )
        if playwright_fetcher is None:
            raise RuntimeError(
                "fetch backend `auto` requires playwright runtime dependencies"
            )
        from serpsage.components.fetch.auto import AutoFetcher

        assert curl_fetcher is not None
        assert playwright_fetcher is not None
        return AutoFetcher(
            rt=rt,
            rate_limiter=rate_limiter,
            http=http,
            curl_fetcher=curl_fetcher,
            playwright_fetcher=playwright_fetcher,
        )

    raise ValueError(
        f"unsupported fetch backend `{backend}`; expected curl_cffi|playwright|auto"
    )


__all__ = [
    "FetcherBase",
    "build_fetcher",
]
