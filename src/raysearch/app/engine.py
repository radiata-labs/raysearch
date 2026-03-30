from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Self, cast
from typing_extensions import override

import anyio

from raysearch.components.cache.base import CacheBase
from raysearch.components.crawl.base import CrawlerBase
from raysearch.components.extract.base import ExtractorBase
from raysearch.components.http.base import HttpClientBase
from raysearch.components.llm.base import LLMClientBase
from raysearch.components.loads import (
    ComponentRegistry,
    load_components,
    materialize_settings,
)
from raysearch.components.metering.base import MeteringSinkBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.rank.base import RankerBase
from raysearch.components.rate_limit.base import RateLimiterBase
from raysearch.components.tracking.base import TrackingSinkBase
from raysearch.core.overrides import Overrides
from raysearch.core.workunit import ClockBase, WorkUnit
from raysearch.dependencies import (
    ANSWER_RUNNER,
    CACHE_TOKEN,
    CHILD_FETCH_RUNNER,
    FETCH_RUNNER,
    RESEARCH_ROUND_RUNNER,
    RESEARCH_RUNNER,
    RESEARCH_SUBREPORT_STEP,
    SEARCH_RUNNER,
    Depends,
    solve_dependencies,
)
from raysearch.models.app.response import (
    AnswerResponse,
    FetchResponse,
    FetchStatusError,
    FetchStatusItem,
    ResearchResponse,
    SearchResponse,
)
from raysearch.models.steps.answer import AnswerStepContext
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.models.steps.research import ResearchStepContext
from raysearch.models.steps.search import SearchStepContext
from raysearch.settings.models import AppSettings
from raysearch.steps.answer import AnswerGenerateStep, AnswerPlanStep, AnswerSearchStep
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.steps.fetch import (
    FetchAbstractBuildStep,
    FetchAbstractRankStep,
    FetchExtractStep,
    FetchFinalizeStep,
    FetchLoadStep,
    FetchOverviewStep,
    FetchParallelEnrichStep,
    FetchPrepareStep,
)
from raysearch.steps.research import (
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
from raysearch.steps.search import (
    SearchFetchStep,
    SearchFinalizeStep,
    SearchPrepareStep,
    SearchQueryPlanStep,
    SearchRerankStep,
    SearchStep,
)
from raysearch.telemetry import (
    AsyncMeteringEmitter,
    AsyncTrackingEmitter,
    MeteringEmitterBase,
    TrackingEmitterBase,
)

if TYPE_CHECKING:
    from raysearch.models.app.request import (
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
    """Async-only engine with safe runtime swapping for config and component reloads."""

    _search_runner: RunnerBase[SearchStepContext] = Depends(SEARCH_RUNNER)
    _fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER)
    _answer_runner: RunnerBase[AnswerStepContext] = Depends(ANSWER_RUNNER)
    _research_runner: RunnerBase[ResearchStepContext] = Depends(RESEARCH_RUNNER)

    def __new__(
        cls,
        setting_file: str | None = None,
        settings: AppSettings | dict[str, Any] | None = None,
    ) -> Self:
        if cls is Engine and (setting_file is not None or settings is not None):
            return cast(
                "Self",
                cls.from_settings(setting_file=setting_file, settings=settings),
            )
        return super().__new__(cls)

    def __init__(
        self,
        setting_file: str | None = None,
        settings: AppSettings | dict[str, Any] | None = None,
    ) -> None:
        if bool(getattr(self, "_engine_is_host", False)):
            return
        if not bool(getattr(self, "_wu_bootstrapped", False)):
            raise TypeError(
                "Engine must be constructed with a `setting_file` or `settings`, "
                "or via `Engine.from_settings(...)`"
            )
        self._engine_is_host = False
        self._engine_runtime_lock = anyio.Lock()
        self._engine_reload_lock = anyio.Lock()
        self._engine_current_runtime: Engine | None = None
        self._engine_retired_runtimes: dict[int, Engine] = {}
        self._engine_runtime_refs: dict[int, int] = {}
        self._engine_accept_requests = True
        self._engine_generation = 0
        self._engine_setting_file = setting_file
        self._engine_settings_source = (
            None if settings is None else self._clone_settings_source(settings)
        )
        self._engine_overrides: Overrides | None = None
        self._engine_component_packages: tuple[str, ...] = ("raysearch.components",)
        self.bind_deps(
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
        component_packages: tuple[str, ...] = ("raysearch.components",),
    ) -> Engine:
        stored_settings = (
            None if settings is None else cls._clone_settings_source(settings)
        )
        normalized_packages = cls._normalize_component_packages(component_packages)
        overrides = overrides or Overrides()
        cls._validate_overrides(overrides)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(
                cls._build_host_engine,
                setting_file,
                stored_settings,
                overrides,
                normalized_packages,
            )

        payload: Any = None
        failure: BaseException | None = None

        def _worker() -> None:
            nonlocal payload, failure
            try:
                payload = anyio.run(
                    cls._build_host_engine,
                    setting_file,
                    stored_settings,
                    overrides,
                    normalized_packages,
                )
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
    def _validate_overrides(cls, overrides: Overrides) -> None:
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

    @classmethod
    def _clone_settings_source(
        cls,
        settings: AppSettings | dict[str, Any],
    ) -> AppSettings | dict[str, Any]:
        if isinstance(settings, AppSettings):
            return settings.model_copy(deep=True)
        return copy.deepcopy(dict(settings))

    @classmethod
    def _normalize_component_packages(
        cls,
        component_packages: tuple[str, ...],
    ) -> tuple[str, ...]:
        packages = tuple(
            dict.fromkeys(item.strip() for item in component_packages if item.strip())
        )
        if not packages:
            return ("raysearch.components",)
        if "raysearch.components" in packages:
            return packages
        return ("raysearch.components", *packages)

    @classmethod
    def _resolve_settings_input(
        cls,
        *,
        setting_file: str | None,
        settings_source: AppSettings | dict[str, Any] | None,
    ) -> AppSettings | dict[str, Any]:
        if settings_source is not None:
            return cls._clone_settings_source(settings_source)
        from raysearch.settings.load import load_settings  # noqa: PLC0415

        return load_settings(path=setting_file)

    @classmethod
    async def _build_host_engine(
        cls,
        setting_file: str | None,
        settings_source: AppSettings | dict[str, Any] | None,
        overrides: Overrides,
        component_packages: tuple[str, ...],
    ) -> Engine:
        runtime = await cls._build_engine(
            cls._resolve_settings_input(
                setting_file=setting_file,
                settings_source=settings_source,
            ),
            overrides,
            component_packages=component_packages,
            rescan_components=True,
        )
        host = cls.__new__(cls)
        host._wu_bootstrap(
            settings=runtime.settings,
            clock=runtime.clock,
            components=runtime.components,
        )
        host._engine_is_host = True
        host._engine_runtime_lock = anyio.Lock()
        host._engine_reload_lock = anyio.Lock()
        host._engine_current_runtime = runtime
        host._engine_retired_runtimes = {}
        host._engine_runtime_refs = {id(runtime): 0}
        host._engine_accept_requests = True
        host._engine_generation = 1
        host._engine_setting_file = setting_file
        host._engine_settings_source = (
            None
            if settings_source is None
            else cls._clone_settings_source(settings_source)
        )
        host._engine_overrides = overrides
        host._engine_component_packages = component_packages
        host._adopt_runtime(runtime)
        return host

    def _adopt_runtime(self, runtime: Engine) -> None:
        self.settings = runtime.settings
        self.clock = runtime.clock
        self.components = runtime.components
        self._tracker = getattr(runtime, "_tracker", None)
        self._meter = getattr(runtime, "_meter", None)
        self._search_runner = runtime._search_runner
        self._fetch_runner = runtime._fetch_runner
        self._answer_runner = runtime._answer_runner
        self._research_runner = runtime._research_runner

    @classmethod
    async def _build_engine(
        cls,
        settings: AppSettings | dict[str, Any],
        overrides: Overrides,
        *,
        component_packages: tuple[str, ...] = ("raysearch.components",),
        rescan_components: bool = False,
    ) -> Engine:
        components = load_components(
            package_names=component_packages,
            rescan=rescan_components,
        )
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
        cache[CACHE_TOKEN] = cache

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

        if registry.enabled_specs("tracking"):
            cache[TrackingSinkBase] = await solve(registry.default_spec("tracking").cls)
        if registry.enabled_specs("metering"):
            cache[MeteringSinkBase] = await solve(registry.default_spec("metering").cls)

        if overrides.tracker is not None:
            cache[TrackingEmitterBase] = overrides.tracker
            cache[type(overrides.tracker)] = overrides.tracker
        elif TrackingSinkBase in cache:
            cache[TrackingEmitterBase] = await solve(AsyncTrackingEmitter)
        else:
            raise RuntimeError("no tracking sink configured")

        if overrides.meter is not None:
            cache[MeteringEmitterBase] = overrides.meter
            cache[type(overrides.meter)] = overrides.meter
        elif MeteringSinkBase in cache:
            cache[MeteringEmitterBase] = await solve(AsyncMeteringEmitter)
        else:
            raise RuntimeError("no metering sink configured")

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

        if overrides.provider is not None:
            cache[SearchProviderBase] = overrides.provider
            cache[type(overrides.provider)] = overrides.provider
        elif registry.enabled_specs("provider"):
            cache[SearchProviderBase] = await solve(
                registry.default_spec("provider").cls
            )

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
                await solve(ResearchFetchStep),
                await solve(ResearchSearchStep),
                await solve(ResearchOverviewStep),
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

    @override
    async def on_init(self) -> None:
        if not self._engine_is_host:
            return
        runtime = self._engine_current_runtime
        if runtime is None:
            raise RuntimeError("engine host has no active runtime")
        await runtime.ainit()
        self._adopt_runtime(runtime)
        async with self._engine_runtime_lock:
            self._engine_accept_requests = True

    @override
    async def on_close(self) -> None:
        if not self._engine_is_host:
            return
        async with self._engine_runtime_lock:
            self._engine_accept_requests = False
            runtime = self._engine_current_runtime
            self._engine_current_runtime = None
            if runtime is not None:
                runtime_id = id(runtime)
                self._engine_retired_runtimes[runtime_id] = runtime
                self._engine_runtime_refs.setdefault(runtime_id, 0)
        await self._drain_retired_runtimes(wait_for_all=True)

    async def _lease_runtime(self) -> Engine:
        if not self._engine_is_host:
            return self
        if not bool(getattr(self, "_wu_initialized", False)):
            await self.ainit()
        async with self._engine_runtime_lock:
            if not self._engine_accept_requests:
                raise RuntimeError("engine is closing and cannot accept new requests")
            runtime = self._engine_current_runtime
            if runtime is None:
                raise RuntimeError("engine host has no active runtime")
            runtime_id = id(runtime)
            self._engine_runtime_refs[runtime_id] = (
                self._engine_runtime_refs.get(runtime_id, 0) + 1
            )
            return runtime

    async def _release_runtime(self, runtime: Engine) -> None:
        if not self._engine_is_host:
            return
        runtime_id = id(runtime)
        should_drain = False
        async with self._engine_runtime_lock:
            refs = self._engine_runtime_refs.get(runtime_id, 0)
            if refs <= 1:
                self._engine_runtime_refs[runtime_id] = 0
                should_drain = runtime_id in self._engine_retired_runtimes
            else:
                self._engine_runtime_refs[runtime_id] = refs - 1
        if should_drain:
            await self._drain_retired_runtimes(wait_for_all=False)

    async def _drain_retired_runtimes(self, *, wait_for_all: bool) -> None:
        close_errors: list[Exception] = []
        while True:
            ready_to_close: list[Engine] = []
            async with self._engine_runtime_lock:
                for runtime_id, runtime in list(self._engine_retired_runtimes.items()):
                    if self._engine_runtime_refs.get(runtime_id, 0) > 0:
                        continue
                    ready_to_close.append(runtime)
                    self._engine_retired_runtimes.pop(runtime_id, None)
                    self._engine_runtime_refs.pop(runtime_id, None)
                pending = wait_for_all and bool(self._engine_retired_runtimes)
            for runtime in ready_to_close:
                try:
                    await runtime.aclose()
                except Exception as exc:  # noqa: BLE001
                    close_errors.append(
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
            if not pending:
                break
            await anyio.sleep(0.05)
        if close_errors:
            raise ExceptionGroup("engine runtime close failed", close_errors)

    async def reload(
        self,
        setting_file: str | None = None,
        *,
        settings: AppSettings | dict[str, Any] | None = None,
        overrides: Overrides | None = None,
        component_packages: tuple[str, ...] | None = None,
        rescan_components: bool = True,
    ) -> None:
        if not self._engine_is_host:
            raise RuntimeError("reload is only available on the top-level Engine host")
        if bool(getattr(self, "_wu_closed", False)):
            raise RuntimeError("engine is already closed")
        async with self._engine_reload_lock:
            async with self._engine_runtime_lock:
                next_setting_file = (
                    self._engine_setting_file if setting_file is None else setting_file
                )
                next_settings_source = (
                    self._engine_settings_source
                    if settings is None
                    else self._clone_settings_source(settings)
                )
                next_overrides = (
                    self._engine_overrides if overrides is None else overrides
                )
                next_component_packages = (
                    self._engine_component_packages
                    if component_packages is None
                    else self._normalize_component_packages(component_packages)
                )
                host_started = bool(getattr(self, "_wu_initialized", False))
            if next_overrides is None:
                next_overrides = Overrides()
            self._validate_overrides(next_overrides)
            new_runtime = await self._build_engine(
                self._resolve_settings_input(
                    setting_file=next_setting_file,
                    settings_source=next_settings_source,
                ),
                next_overrides,
                component_packages=next_component_packages,
                rescan_components=rescan_components,
            )
            try:
                if host_started:
                    await new_runtime.ainit()
            except Exception:
                with suppress(Exception):
                    await new_runtime.aclose()
                raise
            async with self._engine_runtime_lock:
                old_runtime = self._engine_current_runtime
                new_runtime_id = id(new_runtime)
                self._engine_current_runtime = new_runtime
                self._engine_runtime_refs.setdefault(new_runtime_id, 0)
                self._engine_generation += 1
                self._engine_setting_file = next_setting_file
                self._engine_settings_source = (
                    None
                    if next_settings_source is None
                    else self._clone_settings_source(next_settings_source)
                )
                self._engine_overrides = next_overrides
                self._engine_component_packages = next_component_packages
                self._adopt_runtime(new_runtime)
                if old_runtime is not None:
                    old_runtime_id = id(old_runtime)
                    self._engine_retired_runtimes[old_runtime_id] = old_runtime
                    self._engine_runtime_refs.setdefault(old_runtime_id, 0)
            await self._drain_retired_runtimes(wait_for_all=False)

    async def search(self, req: SearchRequest) -> SearchResponse:
        if self._engine_is_host:
            runtime = await self._lease_runtime()
            try:
                return await runtime.search(req)
            finally:
                await self._release_runtime(runtime)
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
        if self._engine_is_host:
            runtime = await self._lease_runtime()
            try:
                return await runtime.fetch(req)
            finally:
                await self._release_runtime(runtime)
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
        if self._engine_is_host:
            runtime = await self._lease_runtime()
            try:
                return await runtime.answer(req)
            finally:
                await self._release_runtime(runtime)
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
        if self._engine_is_host:
            runtime = await self._lease_runtime()
            try:
                return await runtime.research(req)
            finally:
                await self._release_runtime(runtime)
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
