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
from serpsage.components.provider.base import PROVIDER_ROUTES_TOKEN, SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.components.telemetry.base import EventSinkBase
from serpsage.components.telemetry.emitter import TELEMETRY_SINKS_TOKEN
from serpsage.core.runtime import ClockBase, Overrides, Runtime
from serpsage.core.workunit import WorkUnit
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
from serpsage.models.components.telemetry import MeterPayload
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
            self.telemetry,
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
            "telemetry": overrides.telemetry,
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
                    f"override `{name}` must have a bootstrapped WorkUnit runtime"
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
        runtime = Runtime(
            settings=normalized_settings,
            clock=(overrides.clock or SystemClock()),
            env=dict(normalized_settings.runtime_env or {}),
        )
        registry = ComponentRegistry(
            settings=normalized_settings,
            components=components,
            overrides=overrides,
            env=runtime.env,
        )
        runtime.components = registry
        cache: dict[Any, Any] = {
            AppSettings: runtime.settings,
            ClockBase: runtime.clock,
            Runtime: runtime,
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

        telemetry_sinks: list[object] = []
        for spec in registry.enabled_specs("telemetry"):
            if not issubclass(spec.cls, EventSinkBase):
                continue
            sink = await solve(spec.cls)
            if sink is not None:
                telemetry_sinks.append(sink)
        cache[TELEMETRY_SINKS_TOKEN] = tuple(telemetry_sinks)
        if overrides.telemetry is not None:
            cache[TelemetryEmitterBase] = overrides.telemetry
            cache[type(overrides.telemetry)] = overrides.telemetry
        elif registry.enabled_specs("telemetry"):
            cache[TelemetryEmitterBase] = await solve(
                registry.default_spec("telemetry").cls
            )

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

        telemetry = cache.get(TelemetryEmitterBase)
        if isinstance(telemetry, TelemetryEmitterBase):
            runtime.telemetry = telemetry

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
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="search",
            attrs={"request_kind": "search"},
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
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "search",
                    "result_count": len(response.results),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="search")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="search",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "search"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="search",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def fetch(self, req: FetchRequest) -> FetchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="fetch",
            attrs={
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
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "fetch",
                    "result_count": len(response.results),
                    "success_count": int(success_count),
                    "error_count": int(error_count),
                    "url_count": len(req.urls),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="fetch")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="fetch",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "fetch"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="fetch",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def answer(self, req: AnswerRequest) -> AnswerResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="answer",
            attrs={"request_kind": "answer"},
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
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "answer",
                    "citation_count": len(response.citations),
                    "has_answer": bool(
                        response.answer if isinstance(response.answer, str) else True
                    ),
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="answer")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="answer",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "answer"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="answer",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def research(self, req: ResearchRequest) -> ResearchResponse:
        request_id = uuid.uuid4().hex
        started_ms = int(self.clock.now_ms())
        token = self._push_request_context(request_id)
        await self._emit_safe(
            event_name="request.start",
            status="start",
            request_id=request_id,
            component="engine",
            stage="research",
            attrs={"request_kind": "research"},
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
            await self._emit_safe(
                event_name="request.end",
                status="ok",
                request_id=request_id,
                component="engine",
                stage="research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "request_kind": "research",
                    "content_chars": len(str(response.content or "")),
                    "has_structured": response.structured is not None,
                },
            )
            await self._emit_request_meter(request_id=request_id, stage="research")
            return response
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="request.error",
                status="error",
                request_id=request_id,
                component="engine",
                stage="research",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={"request_kind": "research"},
            )
            await self._emit_request_meter(
                request_id=request_id,
                stage="research",
                status="error",
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._pop_request_context(token)

    async def _emit_request_meter(
        self,
        *,
        request_id: str,
        stage: str,
        status: str = "ok",
        error_type: str = "",
    ) -> None:
        await self._emit_safe(
            event_name="meter.usage.request",
            status="error" if status == "error" else "ok",
            request_id=request_id,
            component="engine",
            stage=stage,
            error_type=error_type,
            idempotency_key=f"{request_id}:meter.usage.request:{stage}",
            attrs={"request_kind": stage},
            meter=MeterPayload(
                meter_type="request",
                unit="request",
                quantity=1.0,
            ),
        )

    async def _emit_safe(self, **kwargs: Any) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(**kwargs)

    def _push_request_context(self, request_id: str) -> object:
        telemetry = self.telemetry
        if telemetry is None:
            return None
        try:
            return telemetry.push_request_context(request_id=request_id)
        except Exception:
            return None

    def _pop_request_context(self, token: object) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            telemetry.pop_request_context(token)


__all__ = ["Engine", "Overrides"]
