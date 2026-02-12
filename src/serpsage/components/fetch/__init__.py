from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.components.cache import CacheBase
    from serpsage.components.extract import ExtractorBase
    from serpsage.contracts.services import FetcherBase, RateLimiterBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.http import HttpClient


def build_fetcher(
    *,
    rt: Runtime,
    cache: CacheBase,
    rate_limiter: RateLimiterBase,
    extractor: ExtractorBase,
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

    if backend == "httpx":
        assert httpx_fetcher is not None
        return httpx_fetcher

    if backend == "curl_cffi":
        if curl_fetcher is None:
            raise RuntimeError(
                "fetch backend `curl_cffi` is unavailable: install curl_cffi"
            )
        return curl_fetcher

    if backend == "auto":
        from serpsage.components.fetch.auto import AutoFetcher

        assert httpx_fetcher is not None
        return AutoFetcher(
            rt=rt,
            cache=cache,
            rate_limiter=rate_limiter,
            httpx_fetcher=httpx_fetcher,
            curl_fetcher=curl_fetcher,
            extractor=extractor,
        )

    raise ValueError(
        f"unsupported fetch backend `{backend}`; expected httpx|curl_cffi|auto"
    )


__all__ = [
    "build_fetcher",
]
