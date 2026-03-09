from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.engine import Engine
from serpsage.components import (
    BuiltinComponentDiscovery,
    ComponentContainer,
    build_telemetry,
    get_component_registry,
)
from serpsage.core.runtime import ClockBase, Overrides, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.models.steps.answer import AnswerStepContext
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.models.steps.research import ResearchStepContext
from serpsage.models.steps.search import SearchStepContext
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
    ResearchFetchStep,
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
    env = dict(settings.runtime_env or {})
    return Runtime(
        settings=settings, clock=clock, telemetry=None, components=None, env=env
    )


def build_component_container(
    *,
    settings: AppSettings,
    overrides: Overrides | None = None,
) -> tuple[Runtime, ComponentContainer]:
    BuiltinComponentDiscovery.discover()
    rt = build_runtime(settings=settings, overrides=overrides)
    container = ComponentContainer(
        settings=settings,
        registry=get_component_registry(),
        overrides=overrides,
        env=rt.env,
    )
    rt = rt.model_copy(update={"components": container})
    container.attach_runtime(rt)
    telemetry = build_telemetry(rt=rt, overrides=overrides)
    rt = rt.model_copy(update={"telemetry": telemetry})
    container.attach_runtime(rt)
    return rt, container


def build_engine(
    *, settings: AppSettings, overrides: Overrides | None = None
) -> Engine:
    ov = overrides or Overrides()
    _validate_override_workunits(ov)
    rt, container = build_component_container(settings=settings, overrides=ov)
    child_fetch_steps: list[StepBase[FetchStepContext]] = [
        container.instantiate_class(FetchPrepareStep),
        container.instantiate_class(FetchLoadStep),
        container.instantiate_class(FetchExtractStep),
        container.instantiate_class(FetchAbstractBuildStep),
        container.instantiate_class(FetchAbstractRankStep),
        container.instantiate_class(FetchOverviewStep),
        container.instantiate_class(FetchFinalizeStep),
    ]
    child_fetch_runner = RunnerBase[FetchStepContext](
        rt=rt, steps=child_fetch_steps, kind="child_fetch"
    )
    fetch_steps: list[StepBase[FetchStepContext]] = [
        container.instantiate_class(FetchPrepareStep),
        container.instantiate_class(FetchLoadStep),
        container.instantiate_class(FetchExtractStep),
        container.instantiate_class(FetchAbstractBuildStep),
        container.instantiate_class(FetchAbstractRankStep),
        container.instantiate_class(
            FetchParallelEnrichStep,
            explicit_kwargs={"fetch_runner": child_fetch_runner},
        ),
        container.instantiate_class(FetchFinalizeStep),
    ]
    fetch_runner = RunnerBase[FetchStepContext](rt=rt, steps=fetch_steps, kind="fetch")
    search_steps: list[StepBase[SearchStepContext]] = [
        container.instantiate_class(SearchPrepareStep),
        container.instantiate_class(SearchOptimizeStep),
        container.instantiate_class(SearchExpandStep),
        container.instantiate_class(SearchStep),
        container.instantiate_class(
            SearchFetchStep,
            explicit_kwargs={"fetch_runner": fetch_runner},
        ),
        container.instantiate_class(SearchRankStep),
        container.instantiate_class(SearchFinalizeStep),
    ]
    search_runner = RunnerBase[SearchStepContext](
        rt=rt, steps=search_steps, kind="search"
    )
    answer_steps: list[StepBase[AnswerStepContext]] = [
        container.instantiate_class(AnswerPlanStep),
        container.instantiate_class(
            AnswerSearchStep,
            explicit_kwargs={"search_runner": search_runner},
        ),
        container.instantiate_class(AnswerGenerateStep),
    ]
    answer_runner = RunnerBase[AnswerStepContext](
        rt=rt, steps=answer_steps, kind="search"
    )
    research_round_steps: list[StepBase[ResearchStepContext]] = [
        container.instantiate_class(ResearchPlanStep),
        container.instantiate_class(
            ResearchFetchStep,
            explicit_kwargs={
                "fetch_runner": fetch_runner,
                "phase": "pre",
            },
        ),
        container.instantiate_class(
            ResearchSearchStep,
            explicit_kwargs={"search_runner": search_runner},
        ),
        container.instantiate_class(
            ResearchFetchStep,
            explicit_kwargs={
                "fetch_runner": fetch_runner,
                "phase": "post",
            },
        ),
        container.instantiate_class(ResearchOverviewStep),
        container.instantiate_class(ResearchContentStep),
        container.instantiate_class(ResearchDecideStep),
    ]
    research_round_runner = RunnerBase[ResearchStepContext](
        rt=rt,
        steps=research_round_steps,
        kind="search",
    )
    subreport_step = container.instantiate_class(ResearchSubreportStep)
    research_steps: list[StepBase[ResearchStepContext]] = [
        container.instantiate_class(ResearchPrepareStep),
        container.instantiate_class(ResearchThemeStep),
        container.instantiate_class(
            ResearchLoopStep,
            explicit_kwargs={
                "round_runner": research_round_runner,
                "render_step": subreport_step,
            },
        ),
        container.instantiate_class(ResearchRenderStep),
        container.instantiate_class(ResearchFinalizeStep),
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


__all__ = ["Overrides", "build_component_container", "build_engine", "build_runtime"]
