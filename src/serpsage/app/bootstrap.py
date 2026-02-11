from __future__ import annotations

import time
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING
from typing_extensions import override

import httpx

from serpsage.app.engine import Engine
from serpsage.cache.null import NullCache
from serpsage.cache.sqlite import SqliteCache
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import (
    CacheBase,
    ExtractorBase,
    FetcherBase,
    LLMClientBase,
    PipelineStepBase,
    RankerBase,
    SearchProviderBase,
)
from serpsage.core.runtime import ComponentOverrides, CoreRuntime
from serpsage.domain.dedupe import ResultDeduper
from serpsage.domain.enrich import Enricher
from serpsage.domain.filter import ResultFilterer
from serpsage.domain.normalize import ResultNormalizer
from serpsage.domain.overview import OverviewBuilder
from serpsage.domain.rerank import Reranker
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.extract.html_main import MainContentHtmlExtractor
from serpsage.fetch.auto import AutoFetcher
from serpsage.fetch.curl_cffi import CURL_CFFI_AVAILABLE, CurlCffiFetcher
from serpsage.fetch.http import HttpxFetcher
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.overview.null import NullLLMClient
from serpsage.overview.openai import OpenAIClient
from serpsage.pipeline.steps import (
    DedupeStep,
    EnrichStep,
    FilterStep,
    NormalizeStep,
    OverviewStep,
    RankStep,
    RerankStep,
    SearchStep,
)
from serpsage.provider.searxng import SearxngProvider
from serpsage.rank.blend import BlendRanker
from serpsage.telemetry.trace import NoopTelemetry, TraceTelemetry

if TYPE_CHECKING:
    from serpsage.settings.models import AppSettings


class SystemClock(ClockBase):
    @override
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def build_runtime(
    *, settings: AppSettings, overrides: ComponentOverrides | None = None
) -> CoreRuntime:
    ov = overrides or ComponentOverrides()
    clock = ov.clock or SystemClock()
    telemetry = ov.telemetry or (
        TraceTelemetry(settings.telemetry, clock=clock)
        if settings.telemetry.enabled
        else NoopTelemetry()
    )
    return CoreRuntime(settings=settings, telemetry=telemetry, clock=clock)


def build_engine(
    *, settings: AppSettings, overrides: ComponentOverrides | None = None
) -> Engine:
    ov = overrides or ComponentOverrides()
    rt = build_runtime(settings=settings, overrides=ov)

    stack = AsyncExitStack()

    http = ov.http or httpx.AsyncClient()
    owns_http = ov.http is None
    if owns_http:
        stack.push_async_callback(http.aclose)

    cache: CacheBase = ov.cache or (
        SqliteCache(rt=rt) if settings.cache.enabled else NullCache(rt=rt)
    )
    owns_cache = ov.cache is None
    if owns_cache:
        stack.push_async_callback(cache.aclose)

    rate_limiter = ov.rate_limiter or RateLimiter(rt=rt)
    provider: SearchProviderBase = ov.provider or SearxngProvider(rt=rt, http=http)
    extractor: ExtractorBase = ov.extractor or (
        MainContentHtmlExtractor(rt=rt)
        if (settings.enrich.extractor.kind or "main_content") == "main_content"
        else BasicHtmlExtractor(rt=rt)
    )

    fetcher: FetcherBase
    if ov.fetcher is not None:
        fetcher = ov.fetcher
    else:
        fetch_cfg = settings.enrich.fetch
        fetch_http = httpx.AsyncClient(
            proxy=getattr(fetch_cfg, "proxy", None),
            timeout=httpx.Timeout(float(fetch_cfg.timeout_s)),
            follow_redirects=bool(fetch_cfg.follow_redirects),
            max_redirects=int(getattr(fetch_cfg, "max_redirects", 10)),
            trust_env=False,
        )
        stack.push_async_callback(fetch_http.aclose)

        httpx_fetcher = HttpxFetcher(rt=rt, http=fetch_http)

        curl_fetcher = None
        if str(getattr(fetch_cfg, "strategy", "auto") or "auto") in {
            "auto",
            "curl_cffi",
        }:
            if CURL_CFFI_AVAILABLE:
                try:
                    curl_fetcher = CurlCffiFetcher(rt=rt)
                except Exception:
                    curl_fetcher = None
            if curl_fetcher is not None:
                stack.push_async_callback(curl_fetcher.aclose)

        fetcher = AutoFetcher(
            rt=rt,
            cache=cache,
            rate_limiter=rate_limiter,
            httpx_fetcher=httpx_fetcher,
            curl_fetcher=curl_fetcher,
            extractor=extractor,
        )
    ranker: RankerBase = ov.ranker or BlendRanker(rt=rt)

    llm: LLMClientBase = ov.llm or (
        OpenAIClient(rt=rt, http=http)
        if settings.overview.enabled and settings.overview.llm.api_key
        else NullLLMClient(rt=rt)
    )

    normalizer = ResultNormalizer(rt=rt)
    filterer = ResultFilterer(rt=rt)
    deduper = ResultDeduper(rt=rt)
    enricher = Enricher(rt=rt, fetcher=fetcher, extractor=extractor, ranker=ranker)
    reranker = Reranker(rt=rt, ranker=ranker)
    overview_builder = OverviewBuilder(rt=rt, llm=llm)

    steps: list[PipelineStepBase] = [
        SearchStep(rt=rt, provider=provider, cache=cache),
        NormalizeStep(rt=rt, normalizer=normalizer),
        FilterStep(rt=rt, filterer=filterer),
        DedupeStep(rt=rt, deduper=deduper),
        RankStep(rt=rt, ranker=ranker),
        EnrichStep(rt=rt, enricher=enricher),
        RerankStep(rt=rt, reranker=reranker),
    ]
    overview_step: PipelineStepBase = OverviewStep(
        rt=rt, builder=overview_builder, cache=cache
    )

    init_targets = [
        cache,
        rate_limiter,
        provider,
        extractor,
        fetcher,
        ranker,
        llm,
        normalizer,
        filterer,
        deduper,
        enricher,
        reranker,
        overview_builder,
        *steps,
        overview_step,
    ]

    async def _init() -> None:
        seen: set[int] = set()
        for obj in init_targets:
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            fn = getattr(obj, "ainit", None)
            if fn is None:
                continue
            out = fn()
            if hasattr(out, "__await__"):
                await out

    async def _close() -> None:
        await stack.aclose()

    return Engine(
        rt=rt,
        steps=steps,
        overview_step=overview_step,
        ainit_hook=_init,
        aclose_hook=_close,
    )


__all__ = ["ComponentOverrides", "build_engine", "build_runtime"]
