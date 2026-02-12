from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.engine import Engine
from serpsage.components import (
    build_cache,
    build_extractor,
    build_fetcher,
    build_overview_client,
    build_provider,
    build_ranker,
    build_rate_limiter,
)
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.domain import (
    Deduper,
    Enricher,
    Filterer,
    HttpClient,
    Normalizer,
    OverviewBuilder,
    Reranker,
)
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
from serpsage.telemetry.trace import (
    NoopTelemetry,
    TraceTelemetry,
)

if TYPE_CHECKING:
    from serpsage.contracts.services import (
        CacheBase,
        ExtractorBase,
        FetcherBase,
        LLMClientBase,
        PipelineStepBase,
        RankerBase,
        RateLimiterBase,
        SearchProviderBase,
    )
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

    shared_http_unit: HttpClient | None = None

    def get_shared_http_unit() -> HttpClient:
        nonlocal shared_http_unit
        if shared_http_unit is None:
            shared_http_unit = HttpClient(rt=rt, ov=ov)
        return shared_http_unit

    cache: CacheBase = ov.cache or build_cache(rt=rt)
    rate_limiter: RateLimiterBase = ov.rate_limiter or build_rate_limiter(rt=rt)
    provider: SearchProviderBase = ov.provider or build_provider(
        rt=rt, http=get_shared_http_unit()
    )
    extractor: ExtractorBase = ov.extractor or build_extractor(rt=rt)

    fetcher: FetcherBase = ov.fetcher or build_fetcher(
        rt=rt,
        cache=cache,
        rate_limiter=rate_limiter,
        extractor=extractor,
        http=get_shared_http_unit(),
    )
    ranker: RankerBase = ov.ranker or build_ranker(rt=rt)
    llm: LLMClientBase = ov.llm or build_overview_client(
        rt=rt, http=get_shared_http_unit()
    )

    normalizer = Normalizer(rt=rt)
    filterer = Filterer(rt=rt)
    deduper = Deduper(rt=rt)
    enricher = Enricher(rt=rt, fetcher=fetcher, extractor=extractor, ranker=ranker)
    reranker = Reranker(rt=rt)
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
