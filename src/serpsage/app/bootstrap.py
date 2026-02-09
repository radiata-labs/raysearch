from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from serpsage.app.engine import Engine
from serpsage.app.runtime import CoreRuntime
from serpsage.cache.sqlite import NullCache, SqliteCache
from serpsage.domain.dedupe import ResultDeduper
from serpsage.domain.enrich import Enricher
from serpsage.domain.filter import ResultFilterer
from serpsage.domain.normalize import ResultNormalizer
from serpsage.domain.overview import OverviewBuilder
from serpsage.domain.rerank import Reranker
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.fetch.http import HttpFetcher
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.overview.openai_compat import NullLLMClient, OpenAICompatLLMClient
from serpsage.pipeline.dedupe import DedupeStep
from serpsage.pipeline.enrich import EnrichStep
from serpsage.pipeline.filter import FilterStep
from serpsage.pipeline.normalize import NormalizeStep
from serpsage.pipeline.overview import OverviewStep
from serpsage.pipeline.rank import RankStep
from serpsage.pipeline.rerank import RerankStep
from serpsage.pipeline.search import SearchStep
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
        TraceTelemetry(settings.telemetry)
        if settings.telemetry.enabled
        else NoopTelemetry()
    )
    return CoreRuntime(settings=settings, telemetry=telemetry, clock=clock)


def build_engine(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Engine:
    ov = overrides or Overrides()
    rt = build_runtime(settings=settings, overrides=ov)

    http = ov.http or httpx.AsyncClient()
    owns_http = ov.http is None

    cache: Cache = ov.cache or (
        SqliteCache(rt=rt) if settings.cache.enabled else NullCache(rt=rt)
    )
    rate_limiter = ov.rate_limiter or RateLimiter(rt=rt)
    provider: SearchProvider = ov.provider or SearxngProvider(
        rt=rt, http=http, cache=cache
    )
    fetcher: Fetcher = ov.fetcher or HttpFetcher(
        rt=rt, http=http, cache=cache, rate_limiter=rate_limiter
    )
    extractor: Extractor = ov.extractor or BasicHtmlExtractor(rt=rt)
    ranker: Ranker = ov.ranker or BlendRanker(rt=rt)

    llm: LLMClient = ov.llm or (
        OpenAICompatLLMClient(rt=rt, http=http)
        if settings.overview.enabled and settings.overview.llm.api_key
        else NullLLMClient(rt=rt)
    )

    # Domain services (cohesive logic holders).
    normalizer = ResultNormalizer(rt=rt)
    filterer = ResultFilterer(rt=rt)
    deduper = ResultDeduper(rt=rt)
    enricher = Enricher(rt=rt, fetcher=fetcher, extractor=extractor, ranker=ranker)
    reranker = Reranker(rt=rt, ranker=ranker)
    overview_builder = OverviewBuilder(rt=rt)

    steps = [
        SearchStep(rt=rt, provider=provider, cache=cache),
        NormalizeStep(rt=rt, normalizer=normalizer),
        FilterStep(rt=rt, filterer=filterer),
        DedupeStep(rt=rt, deduper=deduper),
        RankStep(rt=rt, ranker=ranker),
        EnrichStep(rt=rt, enricher=enricher),
        RerankStep(rt=rt, reranker=reranker),
    ]
    overview_step = OverviewStep(rt=rt, llm=llm, builder=overview_builder)

    async def _close() -> None:
        # Best-effort close in dependency order.
        if hasattr(cache, "aclose"):
            await cache.aclose()  # pyright: ignore[reportAttributeAccessIssue]
        if owns_http:
            await http.aclose()

    return Engine(rt=rt, steps=steps, overview_step=overview_step, aclose_hook=_close)


__all__ = ["Overrides", "build_engine", "build_runtime"]
