from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.components.cache import CacheBase
    from serpsage.contracts.services import FetcherBase, RateLimiterBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.http import HttpClient


def build_fetcher(
    *,
    rt: Runtime,
    cache: CacheBase,
    rate_limiter: RateLimiterBase,
    http: HttpClient,
) -> FetcherBase:
    fetch_cfg = rt.settings.enrich.fetch
    backend = str(fetch_cfg.backend or "auto").lower()

    httpx_fetcher = None
    if backend in {"auto", "httpx"}:
        from serpsage.components.fetch.http import HttpxFetcher

        httpx_fetcher = HttpxFetcher(rt=rt, http=http)

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

    if backend == "httpx":
        assert httpx_fetcher is not None
        return httpx_fetcher

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
        if bool(fetch_cfg.playwright.enabled) and playwright_fetcher is None:
            raise RuntimeError(
                "fetch backend `auto` requires playwright when enrich.fetch.playwright.enabled=true"
            )
        from serpsage.components.fetch.auto import AutoFetcher

        assert httpx_fetcher is not None
        return AutoFetcher(
            rt=rt,
            cache=cache,
            rate_limiter=rate_limiter,
            httpx_fetcher=httpx_fetcher,
            curl_fetcher=curl_fetcher,
            playwright_fetcher=playwright_fetcher,
        )

    raise ValueError(
        f"unsupported fetch backend `{backend}`; expected httpx|curl_cffi|playwright|auto"
    )


__all__ = [
    "build_fetcher",
]
