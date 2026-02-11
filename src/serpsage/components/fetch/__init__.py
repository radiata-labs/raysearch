from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from serpsage.components.fetch.auto import AutoFetcher
from serpsage.components.fetch.curl_cffi import CURL_CFFI_AVAILABLE, CurlCffiFetcher
from serpsage.components.fetch.http import HttpxFetcher
from serpsage.components.fetch.http_client_unit import HttpClientUnit

if TYPE_CHECKING:
    from serpsage.components.cache import CacheBase
    from serpsage.components.extract import ExtractorBase
    from serpsage.components.fetch.rate_limit import RateLimiter
    from serpsage.contracts.services import FetcherBase
    from serpsage.core.runtime import Overrides, Runtime


def build_fetcher(
    *,
    rt: Runtime,
    cache: CacheBase,
    rate_limiter: RateLimiter,
    extractor: ExtractorBase,
    ov: Overrides,
) -> FetcherBase:
    fetch_cfg = rt.settings.enrich.fetch
    common = fetch_cfg.common
    backend = str(fetch_cfg.backend or "auto").lower()
    fetch_http_unit = HttpClientUnit(
        rt=rt,
        client=ov.fetch_http
        or httpx.AsyncClient(
            proxy=common.proxy,
            timeout=httpx.Timeout(float(common.timeout_s)),
            follow_redirects=bool(common.follow_redirects),
            max_redirects=int(common.max_redirects),
            trust_env=False,
        ),
        owns_client=ov.fetch_http is None,
    )

    httpx_fetcher = HttpxFetcher(rt=rt, http=fetch_http_unit)

    if backend == "httpx":
        return httpx_fetcher

    curl_fetcher = None
    if backend in {"auto", "curl_cffi"}:
        if CURL_CFFI_AVAILABLE:
            try:
                curl_fetcher = CurlCffiFetcher(rt=rt)
            except Exception:
                curl_fetcher = None

    if backend == "curl_cffi":
        if curl_fetcher is None:
            raise RuntimeError(
                "fetch backend `curl_cffi` is unavailable: install curl_cffi"
            )
        return curl_fetcher

    if backend == "auto":
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
    "AutoFetcher",
    "CurlCffiFetcher",
    "HttpxFetcher",
    "HttpClientUnit",
    "build_fetcher",
]
