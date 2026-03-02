from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.engine import Engine
from serpsage.components import (
    build_cache,
    build_extractor,
    build_fetcher,
    build_http_client,
    build_overview_client,
    build_provider,
    build_ranker,
    build_rate_limiter,
    build_telemetry,
)
from serpsage.core.runtime import ClockBase, Overrides, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.models.pipeline import (
    AnswerStepContext,
    FetchStepContext,
    ResearchStepContext,
    SearchStepContext,
)
from serpsage.steps.answer import AnswerGenerateStep, AnswerPlanStep, AnswerSearchStep
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.fetch import (
    FetchAbstractBuildStep,
    FetchAbstractRankStep,
    FetchExtractStep,
    FetchFinalizeStep,
    FetchLoadStep,
    FetchOverviewStep,
    FetchParallelEnrichStep,
    FetchPrepareStep,
)
from serpsage.steps.research import (
    ResearchContentStep,
    ResearchDecideStep,
    ResearchFinalizeStep,
    ResearchLoopStep,
    ResearchOverviewStep,
    ResearchPlanStep,
    ResearchPrepareStep,
    ResearchRenderStep,
    ResearchSearchStep,
    ResearchSubreportStep,
    ResearchThemeStep,
)
from serpsage.steps.search import (
    SearchExpandStep,
    SearchFetchStep,
    SearchFinalizeStep,
    SearchOptimizeStep,
    SearchPrepareStep,
    SearchRankStep,
    SearchStep,
)

if TYPE_CHECKING:
    from serpsage.components.cache import CacheBase
    from serpsage.components.extract import ExtractorBase
    from serpsage.components.fetch import FetcherBase
    from serpsage.components.llm import LLMClientBase
    from serpsage.components.provider import SearchProviderBase
    from serpsage.components.rank import RankerBase
    from serpsage.components.rate_limit import RateLimiterBase
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
    return Runtime(settings=settings, clock=clock, telemetry=None)


def build_engine(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Engine:
    ov = overrides or Overrides()
    rt = build_runtime(settings=settings, overrides=ov)
    _validate_override_workunits(ov)
    telemetry = build_telemetry(rt=rt, overrides=ov)
    rt = rt.model_copy(update={"telemetry": telemetry})
    shared_http_unit = build_http_client(rt=rt, overrides=ov)
    cache: CacheBase = ov.cache or build_cache(rt=rt)
    rate_limiter: RateLimiterBase = ov.rate_limiter or build_rate_limiter(rt=rt)
    provider: SearchProviderBase = ov.provider or build_provider(
        rt=rt, http=shared_http_unit
    )
    extractor: ExtractorBase = ov.extractor or build_extractor(rt=rt)
    fetcher: FetcherBase = ov.fetcher or build_fetcher(
        rt=rt,
        rate_limiter=rate_limiter,
        http=shared_http_unit,
    )
    ranker: RankerBase = ov.ranker or build_ranker(rt=rt)
    llm: LLMClientBase = ov.llm or build_overview_client(rt=rt, http=shared_http_unit)
    child_fetch_steps: list[StepBase[FetchStepContext]] = [
        FetchPrepareStep(rt=rt),
        FetchLoadStep(rt=rt, fetcher=fetcher, cache=cache),
        FetchExtractStep(rt=rt, extractor=extractor),
        FetchAbstractBuildStep(rt=rt),
        FetchAbstractRankStep(rt=rt, ranker=ranker),
        FetchOverviewStep(rt=rt, llm=llm, cache=cache),
        FetchFinalizeStep(rt=rt),
    ]
    child_fetch_runner = RunnerBase[FetchStepContext](
        rt=rt, steps=child_fetch_steps, kind="child_fetch"
    )
    fetch_steps: list[StepBase[FetchStepContext]] = [
        FetchPrepareStep(rt=rt),
        FetchLoadStep(rt=rt, fetcher=fetcher, cache=cache),
        FetchExtractStep(rt=rt, extractor=extractor),
        FetchAbstractBuildStep(rt=rt),
        FetchAbstractRankStep(rt=rt, ranker=ranker),
        FetchParallelEnrichStep(
            rt=rt,
            llm=llm,
            cache=cache,
            fetch_runner=child_fetch_runner,
            ranker=ranker,
        ),
        FetchFinalizeStep(rt=rt),
    ]
    fetch_runner = RunnerBase[FetchStepContext](rt=rt, steps=fetch_steps, kind="fetch")
    search_steps: list[StepBase[SearchStepContext]] = [
        SearchPrepareStep(rt=rt),
        SearchOptimizeStep(rt=rt, llm=llm),
        SearchExpandStep(rt=rt, llm=llm),
        SearchStep(rt=rt, provider=provider, ranker=ranker),
        SearchFetchStep(rt=rt, fetch_runner=fetch_runner),
        SearchRankStep(rt=rt, ranker=ranker),
        SearchFinalizeStep(rt=rt),
    ]
    search_runner = RunnerBase[SearchStepContext](
        rt=rt, steps=search_steps, kind="search"
    )
    answer_steps: list[StepBase[AnswerStepContext]] = [
        AnswerPlanStep(rt=rt, llm=llm),
        AnswerSearchStep(rt=rt, search_runner=search_runner),
        AnswerGenerateStep(rt=rt, llm=llm),
    ]
    answer_runner = RunnerBase[AnswerStepContext](
        rt=rt, steps=answer_steps, kind="search"
    )
    research_round_steps: list[StepBase[ResearchStepContext]] = [
        ResearchPlanStep(rt=rt, llm=llm),
        ResearchSearchStep(rt=rt, search_runner=search_runner),
        ResearchOverviewStep(rt=rt, llm=llm),
        ResearchContentStep(rt=rt, llm=llm),
        ResearchDecideStep(rt=rt, llm=llm),
    ]
    research_round_runner = RunnerBase[ResearchStepContext](
        rt=rt,
        steps=research_round_steps,
        kind="search",
    )
    research_steps: list[StepBase[ResearchStepContext]] = [
        ResearchPrepareStep(rt=rt),
        ResearchThemeStep(rt=rt, llm=llm),
        ResearchLoopStep(
            rt=rt,
            llm=llm,
            round_runner=research_round_runner,
            render_step=ResearchSubreportStep(rt=rt, llm=llm),
        ),
        ResearchRenderStep(rt=rt, llm=llm),
        ResearchFinalizeStep(rt=rt),
    ]
    research_runner = RunnerBase[ResearchStepContext](
        rt=rt, steps=research_steps, kind="search"
    )
    return Engine(
        rt=rt,
        search_runner=search_runner,
        fetch_runner=fetch_runner,
        answer_runner=answer_runner,
        research_runner=research_runner,
    )


def _validate_override_workunits(ov: Overrides) -> None:
    _ensure_workunit_override("cache", ov.cache)
    _ensure_workunit_override("rate_limiter", ov.rate_limiter)
    _ensure_workunit_override("provider", ov.provider)
    _ensure_workunit_override("fetcher", ov.fetcher)
    _ensure_workunit_override("extractor", ov.extractor)
    _ensure_workunit_override("ranker", ov.ranker)
    _ensure_workunit_override("llm", ov.llm)
    _ensure_workunit_override("telemetry", ov.telemetry)


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
