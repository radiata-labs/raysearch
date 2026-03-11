from __future__ import annotations

import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from serpsage.app.tokens import (
    ANSWER_RUNNER,
    ANSWER_STEPS,
    CHILD_FETCH_RUNNER,
    CHILD_FETCH_STEPS,
    FETCH_RUNNER,
    FETCH_STEPS,
    RESEARCH_ROUND_RUNNER,
    RESEARCH_ROUND_STEPS,
    RESEARCH_RUNNER,
    RESEARCH_STEPS,
    RESEARCH_SUBREPORT_STEP,
    SEARCH_RUNNER,
    SEARCH_STEPS,
)
from serpsage.components.base import (
    CACHE_FAMILY,
    CRAWL_FAMILY,
    EXTRACT_FAMILY,
    HTTP_FAMILY,
    LLM_FAMILY,
    PROVIDER_FAMILY,
    RANK_FAMILY,
    RATE_LIMIT_FAMILY,
    TELEMETRY_FAMILY,
)
from serpsage.components.cache.base import CacheBase
from serpsage.components.crawl.base import CrawlerBase
from serpsage.components.extract.base import ExtractorBase
from serpsage.components.http.base import HttpClientBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.core.runtime import ClockBase, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.dependencies import (
    BindingScope,
    Inject,
    InjectToken,
    ServiceCollection,
    ServiceProvider,
    format_service_key,
)
from serpsage.load.components import ComponentCatalog, ComponentRegistry
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
from serpsage.steps.base import RunnerBase
from serpsage.steps.fetch import (
    FetchAbstractBuildStep,
    FetchAbstractRankStep,
    FetchExtractStep,
    FetchFinalizeStep,
    FetchLoadStep,
    FetchOverviewStep,
    FetchParallelEnrichStep,
    FetchPrepareStep,
    FetchSubpageStep,
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
    from serpsage.models.app.request import (
        AnswerRequest,
        FetchRequest,
        ResearchRequest,
        SearchRequest,
    )


class EngineRuntime(Runtime):
    pass


class EnginePluginRegistry(ComponentRegistry):
    pass


class EngineComponentCatalog(ComponentCatalog):
    pass


class EngineServiceProvider(ServiceProvider):
    pass


class EngineServiceContainer(ServiceCollection):
    @override
    def build_provider(self, *, validate: bool = True) -> EngineServiceProvider:
        provider = EngineServiceProvider(
            single_bindings=dict(self._single_bindings),
            multi_bindings={
                key: tuple(sorted(items, key=lambda item: item.order))
                for key, items in self._multi_bindings.items()
            },
        )
        if validate:
            provider.validate()
        return provider


class SystemClock(ClockBase):
    @override
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _EngineAssembler:
    def build(
        self,
        *,
        settings: AppSettings | dict[str, Any],
    ) -> Engine:
        registry = EnginePluginRegistry.discover()
        normalized_settings = registry.materialize_settings(settings)
        runtime = EngineRuntime(
            settings=normalized_settings,
            clock=SystemClock(),
            env=dict(normalized_settings.runtime_env or {}),
        )
        catalog = EngineComponentCatalog(
            settings=normalized_settings,
            registry=registry,
            env=runtime.env,
        )
        catalog.attach_runtime(runtime)
        runtime.components = catalog
        services = self._build_services(rt=runtime, catalog=catalog)
        runtime.services = services
        runtime.telemetry = cast(
            "TelemetryEmitterBase[Any]",
            services.require(TelemetryEmitterBase),
        )
        return cast("Engine", services.require(Engine))

    def _build_services(
        self,
        *,
        rt: EngineRuntime,
        catalog: EngineComponentCatalog,
    ) -> EngineServiceProvider:
        services = EngineServiceContainer()
        services.bind_instance(AppSettings, rt.settings)
        services.bind_instance(ClockBase, rt.clock)
        services.bind_instance(Runtime, rt)
        services.bind_instance(EngineRuntime, rt)
        services.bind_instance(ComponentCatalog, catalog)
        services.bind_instance(EngineComponentCatalog, catalog)

        self._register_component_services(services=services, catalog=catalog)
        self._register_pipeline_services(services)
        services.bind_class(Engine, Engine, scope=BindingScope.SINGLETON)

        provider = services.build_provider(validate=False)
        provider.bind_instance(ServiceProvider, provider)
        provider.bind_instance(EngineServiceProvider, provider)
        provider.validate()
        return provider

    def _register_component_services(
        self,
        *,
        services: EngineServiceContainer,
        catalog: EngineComponentCatalog,
    ) -> None:
        family_defaults = _family_contracts()
        default_specs = {
            family.name: catalog.resolve_default_spec(family)
            for family in catalog.component_families()
        }
        collection_orders: dict[str, int] = {}
        for spec in catalog.iter_enabled_specs():
            instance_key = catalog.instance_key(spec.family, spec.component_name)
            instance_binding_id = _single_binding_id(instance_key)
            services.bind_class(
                instance_key,
                spec.descriptor.cls,
                scope=BindingScope.SINGLETON,
                overrides={"config": spec.config},
            )
            services.bind_alias(spec.descriptor.cls, instance_key)
            contracts = _component_contracts(spec.descriptor.cls)
            default_contract = family_defaults.get(spec.family.name)
            default_spec = default_specs.get(spec.family.name)
            is_default = default_spec is not None and (
                spec.component_name == default_spec.component_name
            )
            for contract in contracts:
                if default_contract is not None and contract is default_contract:
                    if is_default:
                        services.bind_alias(contract, instance_key)
                services.bind_many(
                    contract,
                    instance_key,
                    order=_next_collection_order(collection_orders, contract),
                    linked_binding_id=instance_binding_id,
                )

    def _register_pipeline_services(self, services: EngineServiceContainer) -> None:
        services.bind_class(
            FetchOverviewStep,
            FetchOverviewStep,
            scope=BindingScope.SINGLETON,
        )
        services.bind_class(
            FetchSubpageStep,
            FetchSubpageStep,
            scope=BindingScope.SINGLETON,
        )
        _bind_steps(
            services,
            CHILD_FETCH_STEPS,
            (
                (10, FetchPrepareStep, {}),
                (20, FetchLoadStep, {}),
                (30, FetchExtractStep, {}),
                (40, FetchAbstractBuildStep, {}),
                (50, FetchAbstractRankStep, {}),
                (60, FetchOverviewStep, {}),
                (70, FetchFinalizeStep, {}),
            ),
        )
        _bind_steps(
            services,
            FETCH_STEPS,
            (
                (10, FetchPrepareStep, {}),
                (20, FetchLoadStep, {}),
                (30, FetchExtractStep, {}),
                (40, FetchAbstractBuildStep, {}),
                (50, FetchAbstractRankStep, {}),
                (60, FetchParallelEnrichStep, {}),
                (70, FetchFinalizeStep, {}),
            ),
        )
        _bind_steps(
            services,
            SEARCH_STEPS,
            (
                (10, SearchPrepareStep, {}),
                (20, SearchOptimizeStep, {}),
                (30, SearchExpandStep, {}),
                (40, SearchStep, {}),
                (50, SearchFetchStep, {}),
                (60, SearchRankStep, {}),
                (70, SearchFinalizeStep, {}),
            ),
        )
        _bind_steps(
            services,
            ANSWER_STEPS,
            (
                (10, AnswerPlanStep, {}),
                (20, AnswerSearchStep, {}),
                (30, AnswerGenerateStep, {}),
            ),
        )
        _bind_steps(
            services,
            RESEARCH_ROUND_STEPS,
            (
                (10, ResearchPlanStep, {}),
                (20, ResearchFetchStep, {"phase": "pre"}),
                (30, ResearchSearchStep, {}),
                (40, ResearchFetchStep, {"phase": "post"}),
                (50, ResearchOverviewStep, {}),
                (60, ResearchContentStep, {}),
                (70, ResearchDecideStep, {}),
            ),
        )
        _bind_steps(
            services,
            RESEARCH_STEPS,
            (
                (10, ResearchPrepareStep, {}),
                (20, ResearchThemeStep, {}),
                (30, ResearchLoopStep, {}),
                (40, ResearchRenderStep, {}),
                (50, ResearchFinalizeStep, {}),
            ),
        )

        services.bind_class(
            CHILD_FETCH_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(CHILD_FETCH_STEPS),
                "kind": "child_fetch",
            },
        )
        services.bind_class(
            FETCH_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(FETCH_STEPS),
                "kind": "fetch",
            },
        )
        services.bind_class(
            SEARCH_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(SEARCH_STEPS),
                "kind": "search",
            },
        )
        services.bind_class(
            ANSWER_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(ANSWER_STEPS),
                "kind": "search",
            },
        )
        services.bind_class(
            RESEARCH_ROUND_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(RESEARCH_ROUND_STEPS),
                "kind": "search",
            },
        )
        services.bind_class(
            RESEARCH_SUBREPORT_STEP,
            ResearchSubreportStep,
            scope=BindingScope.SINGLETON,
            overrides={"rt": Inject(Runtime)},
        )
        services.bind_class(
            RESEARCH_RUNNER,
            RunnerBase,
            scope=BindingScope.SINGLETON,
            overrides={
                "rt": Inject(Runtime),
                "steps": Inject(RESEARCH_STEPS),
                "kind": "search",
            },
        )


class Engine(WorkUnit):
    """Async-only engine with search/fetch/answer/research paths."""

    def __init__(self) -> None:
        self.bind_deps(
            self.telemetry,
            self._search_runner(),
            self._fetch_runner(),
            self._answer_runner(),
            self._research_runner(),
        )

    @classmethod
    def from_settings(
        cls,
        setting_file: str | None = None,
        *,
        settings: AppSettings | dict[str, Any] | None = None,
    ) -> Engine:
        """Build an engine through the Engine-owned plugin registry and DI graph."""
        if settings is None:
            from serpsage.settings.load import load_settings  # noqa: PLC0415

            settings = load_settings(path=setting_file)
        return _EngineAssembler().build(settings=settings)

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
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=SearchResponse(
                request_id=request_id,
                search_mode=req.mode,
                results=[],
            ),
        )
        try:
            ctx = await self._search_runner().run(ctx)
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
                    settings=self.settings,
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
                contexts = await self._fetch_runner().run_batch(contexts)
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
                    "success_count": success_count,
                    "error_count": error_count,
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
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=AnswerResponse(
                request_id=request_id,
                answer="",
                citations=[],
            ),
        )
        try:
            answer_runner = self._answer_runner()
            ctx = await answer_runner.run(ctx)
            ctx.response.answer = ctx.output.answers
            ctx.response.citations = list(ctx.output.citations or [])
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
                    "content_chars": len(str(response.answer or "")),
                    "citation_count": len(response.citations),
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
            settings=self.settings,
            request=req,
            request_id=request_id,
            response=ResearchResponse(
                request_id=request_id,
                content="",
                structured=None,
            ),
        )
        try:
            research_runner = self._research_runner()
            ctx = await research_runner.run(ctx)
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

    def _search_runner(self) -> RunnerBase[SearchStepContext]:
        return cast(
            "RunnerBase[SearchStepContext]", self.services.require(SEARCH_RUNNER)
        )

    def _fetch_runner(self) -> RunnerBase[FetchStepContext]:
        return cast("RunnerBase[FetchStepContext]", self.services.require(FETCH_RUNNER))

    def _answer_runner(self) -> RunnerBase[AnswerStepContext]:
        return cast(
            "RunnerBase[AnswerStepContext]", self.services.require(ANSWER_RUNNER)
        )

    def _research_runner(self) -> RunnerBase[ResearchStepContext]:
        return cast(
            "RunnerBase[ResearchStepContext]",
            self.services.require(RESEARCH_RUNNER),
        )


def _bind_steps(
    services: EngineServiceContainer,
    key: InjectToken[Any],
    steps: tuple[tuple[int, type[object], dict[str, object]], ...],
) -> None:
    for order, cls, overrides in steps:
        services.bind_many(
            key,
            cls,
            order=order,
            scope=BindingScope.SINGLETON,
            overrides=overrides,
        )


def _family_contracts() -> dict[str, type[object]]:
    return {
        HTTP_FAMILY.name: HttpClientBase,
        PROVIDER_FAMILY.name: SearchProviderBase,
        CRAWL_FAMILY.name: CrawlerBase,
        EXTRACT_FAMILY.name: ExtractorBase,
        RANK_FAMILY.name: RankerBase,
        LLM_FAMILY.name: LLMClientBase,
        CACHE_FAMILY.name: CacheBase,
        TELEMETRY_FAMILY.name: TelemetryEmitterBase,
        RATE_LIMIT_FAMILY.name: RateLimiterBase,
    }


def _component_contracts(cls: type[object]) -> tuple[type[object], ...]:
    contracts: list[type[object]] = []
    seen: set[type[object]] = set()
    for base in cls.__mro__[1:]:
        if base in {object, WorkUnit}:
            continue
        if not bool(getattr(base, "__di_contract__", False)):
            continue
        if base in seen:
            continue
        seen.add(base)
        contracts.append(base)
    return tuple(contracts)


def _single_binding_id(key: type[object] | InjectToken[Any]) -> str:
    return f"single:{format_service_key(key)}"


def _next_collection_order(
    orders: dict[str, int],
    key: type[object] | InjectToken[Any],
) -> int:
    key_name = format_service_key(key)
    order = orders.get(key_name, 0) + 10
    orders[key_name] = order
    return order


__all__ = ["Engine"]
