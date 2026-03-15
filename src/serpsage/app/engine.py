from __future__ import annotations

import asyncio
import threading
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.cache.base import CacheBase
from serpsage.components.crawl.base import CrawlerBase
from serpsage.components.extract.base import ExtractorBase
from serpsage.components.http.base import HttpClientBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.llm.router import LLM_ROUTES_TOKEN
from serpsage.components.loads import (
    ComponentRegistry,
    load_components,
    materialize_settings,
)
from serpsage.components.metering import MeteringEmitterBase
from serpsage.components.metering.base import MeteringSinkBase
from serpsage.components.metering.emitter import (
    METERING_SINKS_TOKEN,
    NullMeteringEmitter,
)
from serpsage.components.provider.base import PROVIDER_ROUTES_TOKEN, SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.tracking import TrackingEmitterBase
from serpsage.components.tracking.base import TrackingSinkBase
from serpsage.components.tracking.emitter import (
    TRACKING_SINKS_TOKEN,
    NullTrackingEmitter,
)
from serpsage.core.overrides import Overrides
from serpsage.core.workunit import ClockBase, WorkUnit
from serpsage.dependencies import (
    ANSWER_RUNNER,
    CHILD_FETCH_RUNNER,
    FETCH_RUNNER,
    RESEARCH_ROUND_RUNNER,
    RESEARCH_RUNNER,
    RESEARCH_SUBREPORT_STEP,
    SEARCH_RUNNER,
    Depends,
    solve_dependencies,
)
from serpsage.models.app.response import (
    AnswerResponse,
    FetchResponse,
    FetchStatusError,
    FetchStatusItem,
    ResearchResponse,
    SearchResponse,
)
from serpsage.models.steps.answer import AnswerStepContext
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.models.steps.research import ResearchStepContext
from serpsage.models.steps.search import SearchStepContext
from serpsage.settings.models import AppSettings
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
    SearchFetchStep,
    SearchFinalizeStep,
    SearchPrepareStep,
    SearchQueryPlanStep,
    SearchRerankStep,
    SearchStep,
)

if TYPE_CHECKING:
    from serpsage.models.app.request import (
        AnswerRequest,
        FetchRequest,
        ResearchRequest,
        SearchRequest,
    )


class SystemClock(ClockBase):
    @override
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class Engine(WorkUnit):
    """Async-only engine with integrated bootstrap/search/fetch/answer/research paths."""

    def __init__(
        self,
        *,
        search_runner: RunnerBase[SearchStepContext] = Depends(SEARCH_RUNNER),
        fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER),
        answer_runner: RunnerBase[AnswerStepContext] = Depends(ANSWER_RUNNER),
        research_runner: RunnerBase[ResearchStepContext] = Depends(RESEARCH_RUNNER),
    ) -> None:
        self._search_runner = search_runner
        self._fetch_runner = fetch_runner
        self._answer_runner = answer_runner
        self._research_runner = research_runner
        self.bind_deps(
            self.tracker,
            self.meter,
            self._search_runner,
            self._fetch_runner,
            self._answer_runner,
            self._research_runner,
        )

    @classmethod
    def from_settings(
        cls,
        setting_file: str | None = None,
        *,
        settings: AppSettings | dict[str, Any] | None = None,
        overrides: Overrides | None = None,
    ) -> Engine:
        if settings is None:
            from serpsage.settings.load import load_settings  # noqa: PLC0415

            settings = load_settings(path=setting_file)
        overrides = overrides or Overrides()
        for name, override_value in {
            "cache": overrides.cache,
            "rate_limiter": overrides.rate_limiter,
            "provider": overrides.provider,
            "crawler": overrides.crawler,
            "extractor": overrides.extractor,
            "ranker": overrides.ranker,
            "llm": overrides.llm,
            "tracker": overrides.tracker,
            "meter": overrides.meter,
        }.items():
            if override_value is None:
                continue
            if not isinstance(override_value, WorkUnit):
                raise TypeError(
                    "override "
                    f"`{name}` must be a WorkUnit, got `{type(override_value).__name__}`"
                )
            if not bool(getattr(override_value, "_wu_bootstrapped", False)):
                raise TypeError(
                    f"override `{name}` must already be bootstrapped as a WorkUnit"
                )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(cls._build_engine, settings, overrides)

        payload: Any = None
        failure: BaseException | None = None

        def _worker() -> None:
            nonlocal payload, failure
            try:
                payload = anyio.run(cls._build_engine, settings, overrides)
            except BaseException as exc:  # noqa: BLE001
                failure = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join()
        if failure is not None:
            raise failure
        if payload is None:
            raise RuntimeError("failed to assemble engine")
        return cast("Engine", payload)

    @classmethod
    async def _build_engine(
        cls,
        settings: AppSettings | dict[str, Any],
        overrides: Overrides,
    ) -> Engine:
        components = load_components()
        normalized_settings = materialize_settings(
            settings=settings,
            components=components,
        )
        clock = overrides.clock or SystemClock()
        registry = ComponentRegistry(
            settings=normalized_settings,
            components=components,
            overrides=overrides,
            env=dict(normalized_settings.runtime_env or {}),
        )
        cache: dict[Any, Any] = {
            AppSettings: normalized_settings,
            ClockBase: clock,
            ComponentRegistry: registry,
        }
        for spec in registry.all_specs():
            cache[spec.config_cls] = spec.config
            if not bool(spec.config.enabled):
                cache[spec.cls] = None

        async def solve(
            dependent: Any,
            *,
            use_cache: bool = True,
            **named: object,
        ) -> Any:
            marker = object()
            previous: dict[str, object] = {}
            for key, value in named.items():
                previous[key] = cache.get(key, marker)
                cache[key] = value
            try:
                return await solve_dependencies(
                    dependent,
                    use_cache=use_cache,
                    dependency_cache=cache,
                )
            finally:
                for key, value in previous.items():
                    if value is marker:
                        cache.pop(key, None)
                    else:
                        cache[key] = value

        async def solve_transient(dependent: Any, **named: object) -> Any:
            resolved = await solve(dependent, use_cache=False, **named)
            cache.pop(dependent, None)
            return resolved

        async def solve_many(
            *dependents: type[StepBase[Any]],
        ) -> list[StepBase[Any]]:
            out: list[StepBase[Any]] = []
            for dependent in dependents:
                instance = await solve(dependent)
                if not isinstance(instance, StepBase):
                    raise TypeError(
                        f"{dependent.__name__} did not resolve to a StepBase"
                    )
                out.append(instance)
            return out

        for family, contract, override_value in (
            ("http", HttpClientBase, None),
            ("cache", CacheBase, overrides.cache),
            ("rate_limit", RateLimiterBase, overrides.rate_limiter),
            ("extract", ExtractorBase, overrides.extractor),
            ("rank", RankerBase, overrides.ranker),
            ("crawl", CrawlerBase, overrides.crawler),
        ):
            if override_value is not None:
                cache[contract] = override_value
                cache[type(override_value)] = override_value
                continue
            enabled_specs = registry.enabled_specs(family)
            if not enabled_specs:
                continue
            cache[contract] = await solve(registry.default_spec(family).cls)

        tracking_sinks: list[object] = []
        for spec in registry.enabled_specs("tracking"):
            if not issubclass(spec.cls, TrackingSinkBase):
                continue
            sink = await solve(spec.cls)
            if sink is not None:
                tracking_sinks.append(sink)
        cache[TRACKING_SINKS_TOKEN] = tuple(tracking_sinks)
        tracker_emitter: TrackingEmitterBase[Any]
        if registry.enabled_specs("tracking") and overrides.tracker is None:
            tracker_emitter = await solve(registry.default_spec("tracking").cls)
        elif overrides.tracker is not None:
            tracker_emitter = overrides.tracker
        else:
            tracker_emitter = await solve_transient(NullTrackingEmitter)
        cache[TrackingEmitterBase] = tracker_emitter
        cache[type(tracker_emitter)] = tracker_emitter

        metering_sinks: list[object] = []
        for spec in registry.enabled_specs("metering"):
            if not issubclass(spec.cls, MeteringSinkBase):
                continue
            sink = await solve(spec.cls)
            if sink is not None:
                metering_sinks.append(sink)
        cache[METERING_SINKS_TOKEN] = tuple(metering_sinks)
        meter_emitter: MeteringEmitterBase[Any]
        if registry.enabled_specs("metering") and overrides.meter is None:
            meter_emitter = await solve(registry.default_spec("metering").cls)
        elif overrides.meter is not None:
            meter_emitter = overrides.meter
        else:
            meter_emitter = await solve_transient(NullMeteringEmitter)
        cache[MeteringEmitterBase] = meter_emitter
        cache[type(meter_emitter)] = meter_emitter

        provider_routes: list[object] = []
        for spec in registry.enabled_specs("provider"):
            if spec.name == "blend":
                continue
            if not issubclass(spec.cls, SearchProviderBase):
                continue
            route = await solve(spec.cls)
            if route is not None:
                provider_routes.append(route)
        cache[PROVIDER_ROUTES_TOKEN] = tuple(provider_routes)
        if overrides.provider is not None:
            cache[SearchProviderBase] = overrides.provider
            cache[type(overrides.provider)] = overrides.provider
        elif registry.enabled_specs("provider"):
            cache[SearchProviderBase] = await solve(
                registry.default_spec("provider").cls
            )

        llm_routes: list[object] = []
        for spec in registry.enabled_specs("llm"):
            if spec.cls.__name__ == "RoutedLLMClient":
                continue
            if not issubclass(spec.cls, LLMClientBase):
                continue
            route = await solve(spec.cls)
            if route is not None:
                llm_routes.append(route)
        cache[LLM_ROUTES_TOKEN] = tuple(llm_routes)
        if overrides.llm is not None:
            cache[LLMClientBase] = overrides.llm
            cache[type(overrides.llm)] = overrides.llm
        elif registry.enabled_specs("llm"):
            cache[LLMClientBase] = await solve(registry.default_spec("llm").cls)

        cache[CHILD_FETCH_RUNNER] = await solve_transient(
            RunnerBase,
            steps=await solve_many(
                FetchPrepareStep,
                FetchLoadStep,
                FetchExtractStep,
                FetchAbstractBuildStep,
                FetchAbstractRankStep,
                FetchOverviewStep,
                FetchFinalizeStep,
            ),
            kind="child_fetch",
        )
        cache[FETCH_RUNNER] = await solve_transient(
            RunnerBase,
            steps=await solve_many(
                FetchPrepareStep,
                FetchLoadStep,
                FetchExtractStep,
                FetchAbstractBuildStep,
                FetchAbstractRankStep,
                FetchParallelEnrichStep,
                FetchFinalizeStep,
            ),
            kind="fetch",
        )
        cache[SEARCH_RUNNER] = await solve_transient(
            RunnerBase,
            steps=await solve_many(
                SearchPrepareStep,
                SearchQueryPlanStep,
                SearchStep,
                SearchFetchStep,
                SearchRerankStep,
                SearchFinalizeStep,
            ),
            kind="search",
        )
        cache[ANSWER_RUNNER] = await solve_transient(
            RunnerBase,
            steps=await solve_many(
                AnswerPlanStep,
                AnswerSearchStep,
                AnswerGenerateStep,
            ),
            kind="search",
        )
        cache[RESEARCH_ROUND_RUNNER] = await solve_transient(
            RunnerBase,
            steps=[
                await solve(ResearchPlanStep),
                await solve_transient(ResearchFetchStep, phase="pre"),
                await solve(ResearchSearchStep),
                await solve_transient(ResearchFetchStep, phase="post"),
                await solve(ResearchOverviewStep),
                await solve(ResearchContentStep),
                await solve(ResearchDecideStep),
            ],
            kind="search",
        )
        cache[RESEARCH_SUBREPORT_STEP] = await solve(ResearchSubreportStep)
        cache[RESEARCH_RUNNER] = await solve_transient(
            RunnerBase,
            steps=await solve_many(
                ResearchPrepareStep,
                ResearchThemeStep,
                ResearchLoopStep,
                ResearchRenderStep,
                ResearchFinalizeStep,
            ),
            kind="search",
        )

        engine = await solve(cls)
        if not isinstance(engine, cls):
            raise TypeError("engine was not constructed")
        return engine

    async def search(self, req: SearchRequest) -> SearchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self.tracker.info(
            name="engine.request.started",
            request_id=request_id,
            step="engine.search",
            data={"request_kind": "search"},
        )
        ctx = SearchStepContext(
            request=req,
            request_id=request_id,
            response=SearchResponse(
                request_id=request_id,
                search_mode=req.mode,
                results=[],
            ),
        )
        try:
            ctx = await self._search_runner.run(ctx)
            ctx.response.search_mode = ctx.request.mode
            ctx.response.results = list(ctx.output.results)
            response = ctx.response
            await self.tracker.info(
                name="engine.request.completed",
                request_id=request_id,
                step="engine.search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                data={
                    "request_kind": "search",
                    "result_count": len(response.results),
                },
            )
            return response
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="engine.request.failed",
                request_id=request_id,
                step="engine.search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "request_kind": "search",
                },
            )
            raise
        finally:
            await self.meter.record(
                name="request",
                request_id=request_id,
                key=f"{request_id}:request:search",
                unit="request",
            )
            self._pop_request_context(token)

    async def fetch(self, req: FetchRequest) -> FetchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self.tracker.info(
            name="engine.request.started",
            request_id=request_id,
            step="engine.fetch",
            data={
                "request_kind": "fetch",
                "url_count": len(req.urls),
            },
        )
        try:
            contexts: list[FetchStepContext] = []
            for idx, url in enumerate(req.urls):
                fetch_ctx = FetchStepContext(
                    request=req,
                    request_id=request_id,
                    response=FetchResponse(
                        request_id=request_id,
                        results=[],
                        statuses=[],
                    ),
                    url=url,
                    url_index=idx,
                )
                fetch_ctx.related.enabled = True
                fetch_ctx.page.crawl_mode = req.crawl_mode
                fetch_ctx.page.crawl_timeout_s = float(req.crawl_timeout or 0.0)
                fetch_ctx.related.link_limit = (
                    req.others.max_links if req.others is not None else None
                )
                fetch_ctx.related.image_limit = (
                    req.others.max_image_links if req.others is not None else None
                )
                contexts.append(fetch_ctx)
            if contexts:
                contexts = await self._fetch_runner.run_batch(contexts)
            results = [
                ctx.result
                for ctx in contexts
                if not ctx.error.failed and ctx.result is not None
            ]
            statuses = [
                FetchStatusItem(
                    url=str(url),
                    status="error",
                    error=FetchStatusError(
                        tag="SOURCE_NOT_AVAILABLE",
                        detail="not processed",
                    ),
                )
                for url in req.urls
            ]
            for ctx_item in contexts:
                idx = int(ctx_item.url_index)
                if idx < 0 or idx >= len(statuses):
                    continue
                success = bool(
                    ctx_item.result is not None and not ctx_item.error.failed
                )
                statuses[idx] = FetchStatusItem(
                    url=str(ctx_item.url),
                    status="success" if success else "error",
                    error=(
                        None
                        if success
                        else FetchStatusError(
                            tag=ctx_item.error.tag,
                            detail=ctx_item.error.detail,
                        )
                    ),
                )
            success_count = sum(1 for item in statuses if str(item.status) == "success")
            error_count = max(0, len(statuses) - success_count)
            response = FetchResponse(
                request_id=request_id,
                results=results,
                statuses=statuses,
            )
            await self.tracker.info(
                name="engine.request.completed",
                request_id=request_id,
                step="engine.fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                data={
                    "request_kind": "fetch",
                    "result_count": len(response.results),
                    "success_count": int(success_count),
                    "error_count": int(error_count),
                    "url_count": len(req.urls),
                },
            )
            return response
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="engine.request.failed",
                request_id=request_id,
                step="engine.fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "request_kind": "fetch",
                },
            )
            raise
        finally:
            await self.meter.record(
                name="request",
                request_id=request_id,
                key=f"{request_id}:request:fetch",
                unit="request",
            )
            self._pop_request_context(token)

    async def answer(self, req: AnswerRequest) -> AnswerResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self.tracker.info(
            name="engine.request.started",
            request_id=request_id,
            step="engine.answer",
            data={"request_kind": "answer"},
        )
        ctx = AnswerStepContext(
            request=req,
            request_id=request_id,
            response=AnswerResponse(
                request_id=request_id,
                answer={},
                citations=[],
            ),
        )
        try:
            ctx = await self._answer_runner.run(ctx)
            answer: str | object
            if isinstance(req.json_schema, dict):
                answer = (
                    ctx.output.answers if isinstance(ctx.output.answers, dict) else {}
                )
            else:
                answer = (
                    str(ctx.output.answers)
                    if isinstance(ctx.output.answers, str)
                    else ""
                )
            ctx.response.answer = answer
            ctx.response.citations = list(ctx.output.citations)
            response = ctx.response
            await self.tracker.info(
                name="engine.request.completed",
                request_id=request_id,
                step="engine.answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                data={
                    "request_kind": "answer",
                    "citation_count": len(response.citations),
                    "has_answer": bool(
                        response.answer if isinstance(response.answer, str) else True
                    ),
                },
            )
            return response
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="engine.request.failed",
                request_id=request_id,
                step="engine.answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "request_kind": "answer",
                },
            )
            raise
        finally:
            await self.meter.record(
                name="request",
                request_id=request_id,
                key=f"{request_id}:request:answer",
                unit="request",
            )
            self._pop_request_context(token)

    async def research(self, req: ResearchRequest) -> ResearchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self.tracker.info(
            name="engine.request.started",
            request_id=request_id,
            step="engine.research",
            data={"request_kind": "research"},
        )
        ctx = ResearchStepContext(
            request=req,
            request_id=request_id,
            response=ResearchResponse(
                request_id=request_id,
                content="",
                structured=None,
            ),
        )
        try:
            ctx = await self._research_runner.run(ctx)
            ctx.response.content = ctx.result.content
            ctx.response.structured = ctx.result.structured
            response = ctx.response
            await self.tracker.info(
                name="engine.request.completed",
                request_id=request_id,
                step="engine.research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                data={
                    "request_kind": "research",
                    "content_chars": len(str(response.content or "")),
                    "has_structured": response.structured is not None,
                },
            )
            return response
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="engine.request.failed",
                request_id=request_id,
                step="engine.research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "request_kind": "research",
                },
            )
            raise
        finally:
            await self.meter.record(
                name="request",
                request_id=request_id,
                key=f"{request_id}:request:research",
                unit="request",
            )
            self._pop_request_context(token)

    def _push_request_context(self, request_id: str) -> object:
        tracker_token: object = None
        meter_token: object = None
        try:
            tracker_token = self.tracker.push_request_context(request_id=request_id)
        except Exception:
            tracker_token = None
        try:
            meter_token = self.meter.push_request_context(request_id=request_id)
        except Exception:
            meter_token = None
        return (tracker_token, meter_token)

    def _pop_request_context(self, token: object) -> None:
        tracker_token: object = None
        meter_token: object = None
        if isinstance(token, tuple) and len(token) == 2:
            tracker_token, meter_token = token
        else:
            tracker_token = token
        with suppress(Exception):
            self.tracker.pop_request_context(tracker_token)
        with suppress(Exception):
            self.meter.pop_request_context(meter_token)


__all__ = ["Engine", "Overrides"]
