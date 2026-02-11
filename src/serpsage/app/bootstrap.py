from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast
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
from serpsage.core.runtime import Overrides, Runtime
from serpsage.core.workunit import WorkUnit
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
from serpsage.fetch.http_client_unit import HttpClientUnit
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
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Runtime:
    ov = overrides or Overrides()
    clock = ov.clock or SystemClock()
    telemetry = ov.telemetry or (
        TraceTelemetry(settings.telemetry, clock=clock)
        if settings.telemetry.enabled
        else NoopTelemetry()
    )
    return Runtime(settings=settings, telemetry=telemetry, clock=clock)


def build_engine(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Engine:
    ov = overrides or Overrides()
    rt = build_runtime(settings=settings, overrides=ov)

    _validate_override_workunits(ov)

    shared_http_unit: HttpClientUnit | None = None

    def get_shared_http_unit() -> HttpClientUnit:
        nonlocal shared_http_unit
        if shared_http_unit is None:
            shared_http_unit = HttpClientUnit(
                rt=rt,
                client=ov.http,
                owns_client=ov.http is None,
            )
        return shared_http_unit

    cache: CacheBase = ov.cache or (
        SqliteCache(rt=rt) if settings.cache.enabled else NullCache(rt=rt)
    )
    rate_limiter: RateLimiter = (
        cast("RateLimiter", ov.rate_limiter)
        if ov.rate_limiter is not None
        else RateLimiter(rt=rt)
    )
    provider: SearchProviderBase = ov.provider or SearxngProvider(
        rt=rt, http=get_shared_http_unit()
    )
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
        fetch_http_unit = HttpClientUnit(
            rt=rt,
            client=ov.fetch_http
            or httpx.AsyncClient(
                proxy=getattr(fetch_cfg, "proxy", None),
                timeout=httpx.Timeout(float(fetch_cfg.timeout_s)),
                follow_redirects=bool(fetch_cfg.follow_redirects),
                max_redirects=int(getattr(fetch_cfg, "max_redirects", 10)),
                trust_env=False,
            ),
            owns_client=ov.fetch_http is None,
        )

        httpx_fetcher = HttpxFetcher(rt=rt, http=fetch_http_unit)

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
        OpenAIClient(rt=rt, http=get_shared_http_unit())
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

    return Engine(
        rt=rt,
        steps=steps,
        overview_step=overview_step,
    )


def _validate_override_workunits(ov: Overrides) -> None:
    _ensure_workunit_override("cache", ov.cache)
    _ensure_workunit_override("rate_limiter", ov.rate_limiter)
    _ensure_workunit_override("provider", ov.provider)
    _ensure_workunit_override("fetcher", ov.fetcher)
    _ensure_workunit_override("extractor", ov.extractor)
    _ensure_workunit_override("ranker", ov.ranker)
    _ensure_workunit_override("llm", ov.llm)


def _ensure_workunit_override(name: str, obj: object | None) -> None:
    if obj is None:
        return
    if not isinstance(obj, WorkUnit):
        raise TypeError(
            f"override `{name}` must be a WorkUnit, got `{type(obj).__name__}`"
        )
    if not bool(getattr(obj, "_wu_bootstrapped", False)):
        raise TypeError(
            f"override `{name}` must call WorkUnit.__init__(rt=...) via super().__init__"
        )


__all__ = ["Overrides", "build_engine", "build_runtime"]
