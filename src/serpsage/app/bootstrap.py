from __future__ import annotations

import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from serpsage.app.engine import Engine
from serpsage.app.runtime import CoreRuntime
from serpsage.cache.null import NullCache
from serpsage.cache.sqlite import SqliteCache
from serpsage.domain.dedupe import ResultDeduper
from serpsage.domain.enrich import Enricher
from serpsage.domain.filter import ResultFilterer
from serpsage.domain.normalize import ResultNormalizer
from serpsage.domain.overview import OverviewBuilder
from serpsage.domain.rerank import Reranker
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.extract.html_main import MainContentHtmlExtractor
from serpsage.fetch.http import HttpFetcher
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.overview.null import NullLLMClient
from serpsage.overview.openai import OpenAIClient
from serpsage.pipeline.builtins import (
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
    from serpsage.contracts.protocols import (
        Cache,
        Clock,
        Extractor,
        Fetcher,
        LLMClient,
        Ranker,
        SearchProvider,
        Telemetry,
    )
    from serpsage.pipeline.steps import Step
    from serpsage.settings.models import AppSettings


class SystemClock:
    def now_ms(self) -> int:

        return int(time.time() * 1000)


@dataclass
class Overrides:
    # Low-level infra
    http: httpx.AsyncClient | None = None
    telemetry: Telemetry | None = None
    clock: Clock | None = None

    # Core components
    cache: Cache | None = None
    rate_limiter: RateLimiter | None = None
    provider: SearchProvider | None = None
    fetcher: Fetcher | None = None
    extractor: Extractor | None = None
    ranker: Ranker | None = None
    llm: LLMClient | None = None


def build_runtime(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> CoreRuntime:
    ov = overrides or Overrides()
    clock = ov.clock or SystemClock()
    telemetry = ov.telemetry or (
        TraceTelemetry(settings.telemetry, clock=clock)
        if settings.telemetry.enabled
        else NoopTelemetry()
    )
    return CoreRuntime(settings=settings, telemetry=telemetry, clock=clock)


def build_engine(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Engine:
    ov = overrides or Overrides()
    rt = build_runtime(settings=settings, overrides=ov)

    stack = AsyncExitStack()

    http = ov.http or httpx.AsyncClient()
    owns_http = ov.http is None
    if owns_http:
        # Close last (LIFO) after dependent resources.
        stack.push_async_callback(http.aclose)

    cache: Cache = ov.cache or (
        SqliteCache(rt=rt) if settings.cache.enabled else NullCache(rt=rt)
    )
    owns_cache = ov.cache is None
    if owns_cache:
        stack.push_async_callback(cache.aclose)

    rate_limiter = ov.rate_limiter or RateLimiter(rt=rt)
    provider: SearchProvider = ov.provider or SearxngProvider(rt=rt, http=http)
    fetcher: Fetcher = ov.fetcher or HttpFetcher(
        rt=rt, http=http, cache=cache, rate_limiter=rate_limiter
    )
    extractor: Extractor = ov.extractor or (
        MainContentHtmlExtractor(rt=rt)
        if (settings.enrich.extractor.kind or "main_content") == "main_content"
        else BasicHtmlExtractor(rt=rt)
    )
    ranker: Ranker = ov.ranker or BlendRanker(rt=rt)

    llm: LLMClient = ov.llm or (
        OpenAIClient(rt=rt, http=http)
        if settings.overview.enabled and settings.overview.llm.api_key
        else NullLLMClient(rt=rt)
    )

    # Domain services (cohesive logic holders).
    normalizer = ResultNormalizer(rt=rt)
    filterer = ResultFilterer(rt=rt)
    deduper = ResultDeduper(rt=rt)
    enricher = Enricher(rt=rt, fetcher=fetcher, extractor=extractor, ranker=ranker)
    reranker = Reranker(rt=rt, ranker=ranker)
    overview_builder = OverviewBuilder(rt=rt, llm=llm)

    steps: list[Step] = [
        SearchStep(rt=rt, provider=provider, cache=cache),
        NormalizeStep(rt=rt, normalizer=normalizer),
        FilterStep(rt=rt, filterer=filterer),
        DedupeStep(rt=rt, deduper=deduper),
        RankStep(rt=rt, ranker=ranker),
        EnrichStep(rt=rt, enricher=enricher),
        RerankStep(rt=rt, reranker=reranker),
    ]
    overview_step: Step = OverviewStep(rt=rt, builder=overview_builder, cache=cache)

    async def _close() -> None:
        await stack.aclose()

    return Engine(rt=rt, steps=steps, overview_step=overview_step, aclose_hook=_close)


__all__ = ["Overrides", "build_engine", "build_runtime"]
