from __future__ import annotations

import time
from typing import Any, cast
from typing_extensions import override

from serpsage.app.engine import Engine
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
from serpsage.components import (
    BuiltinComponentDiscovery,
    ComponentCatalog,
    ComponentContainer,
    get_component_registry,
)
from serpsage.components.base import (
    CACHE_FAMILY,
    EXTRACT_FAMILY,
    FETCH_FAMILY,
    HTTP_FAMILY,
    LLM_FAMILY,
    PROVIDER_FAMILY,
    RANK_FAMILY,
    RATE_LIMIT_FAMILY,
    TELEMETRY_FAMILY,
    ComponentFamily,
)
from serpsage.components.cache.base import CacheBase
from serpsage.components.extract.base import ExtractorBase
from serpsage.components.fetch.base import FetcherBase
from serpsage.components.http.base import HttpClientBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.core.runtime import ClockBase, Overrides, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.dependencies import (
    BindingScope,
    Inject,
    InjectToken,
    ServiceCollection,
    ServiceProvider,
    format_service_key,
)
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


class SystemClock(ClockBase):
    @override
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def build_runtime(
    *,
    settings: AppSettings,
    overrides: Overrides | None = None,
) -> Runtime:
    ov = overrides or Overrides()
    clock = ov.clock or SystemClock()
    env = dict(settings.runtime_env or {})
    return Runtime(settings=settings, clock=clock, env=env)


def build_component_container(
    *,
    settings: AppSettings,
    overrides: Overrides | None = None,
) -> tuple[Runtime, ComponentContainer]:
    ov = overrides or Overrides()
    BuiltinComponentDiscovery.discover()
    rt = build_runtime(settings=settings, overrides=ov)
    catalog = ComponentCatalog(
        settings=settings,
        registry=get_component_registry(),
        overrides=ov,
        env=rt.env,
    )
    rt.components = catalog
    provider = _build_services(rt=rt, catalog=catalog, overrides=ov)
    rt.services = provider
    rt.telemetry = cast("TelemetryEmitterBase", provider.require(TelemetryEmitterBase))
    return rt, catalog


def build_engine(
    *,
    settings: AppSettings,
    overrides: Overrides | None = None,
) -> Engine:
    ov = overrides or Overrides()
    _validate_override_workunits(ov)
    rt, _catalog = build_component_container(settings=settings, overrides=ov)
    services = rt.services
    if services is None:
        raise RuntimeError("runtime service provider is not attached")
    return cast("Engine", services.require(Engine))


def _build_services(
    *,
    rt: Runtime,
    catalog: ComponentCatalog,
    overrides: Overrides,
) -> ServiceProvider:
    services = ServiceCollection()
    services.bind_instance(AppSettings, rt.settings)
    services.bind_instance(ClockBase, rt.clock)
    services.bind_instance(Runtime, rt)
    services.bind_instance(ComponentCatalog, catalog)

    _register_component_services(
        services=services, catalog=catalog, overrides=overrides
    )
    _register_pipeline_services(services)
    services.bind_class(Engine, Engine, scope=BindingScope.SINGLETON)

    provider = services.build_provider()
    provider.bind_instance(ServiceProvider, provider)
    return provider


def _register_component_services(
    *,
    services: ServiceCollection,
    catalog: ComponentCatalog,
    overrides: Overrides,
) -> None:
    family_defaults = _family_contracts()
    family_overrides = _workunit_overrides(overrides)
    collection_orders: dict[str, int] = {}
    for family, override in family_overrides.items():
        contract = family_defaults.get(family.name)
        if contract is not None:
            services.bind_instance(contract, override)
        services.bind_instance(type(override), override)
    for spec in catalog.iter_enabled_specs():
        if spec.family in family_overrides:
            continue
        instance_key = _component_instance_key(spec.family, spec.instance_id)
        services.bind_class(
            instance_key,
            spec.descriptor.cls,
            scope=BindingScope.SINGLETON,
            init_kwargs={"rt": rt, "config": spec.config},
        )
        services.bind_alias(spec.descriptor.cls, instance_key)
        contracts = list(spec.descriptor.meta.contracts)
        default_contract = family_defaults.get(spec.family.name)
        if (
            default_contract is not None
            and issubclass(spec.descriptor.cls, default_contract)
            and default_contract not in contracts
        ):
            contracts.insert(0, default_contract)
        family_settings = catalog.family_settings(spec.family)
        is_default = str(spec.instance_id) == str(family_settings.default)
        for contract in contracts:
            key_name = format_key_name(contract)
            order = collection_orders.get(key_name, 0) + 10
            collection_orders[key_name] = order
            services.bind_many(contract, instance_key, order=order)
            if is_default:
                services.bind_alias(contract, instance_key)


def _register_pipeline_services(services: ServiceCollection) -> None:
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
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(CHILD_FETCH_STEPS),
            "kind": "child_fetch",
        },
    )
    services.bind_class(
        FETCH_RUNNER,
        RunnerBase,
        scope=BindingScope.SINGLETON,
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(FETCH_STEPS),
            "kind": "fetch",
        },
    )
    services.bind_class(
        SEARCH_RUNNER,
        RunnerBase,
        scope=BindingScope.SINGLETON,
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(SEARCH_STEPS),
            "kind": "search",
        },
    )
    services.bind_class(
        ANSWER_RUNNER,
        RunnerBase,
        scope=BindingScope.SINGLETON,
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(ANSWER_STEPS),
            "kind": "search",
        },
    )
    services.bind_class(
        RESEARCH_ROUND_RUNNER,
        RunnerBase,
        scope=BindingScope.SINGLETON,
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(RESEARCH_ROUND_STEPS),
            "kind": "search",
        },
    )
    services.bind_class(
        RESEARCH_SUBREPORT_STEP,
        ResearchSubreportStep,
        scope=BindingScope.SINGLETON,
        init_kwargs={"rt": Inject(Runtime)},
    )
    services.bind_class(
        RESEARCH_RUNNER,
        RunnerBase,
        scope=BindingScope.SINGLETON,
        init_kwargs={
            "rt": Inject(Runtime),
            "steps": Inject(RESEARCH_STEPS),
            "kind": "search",
        },
    )


def _bind_steps(
    services: ServiceCollection,
    key: InjectToken[Any],
    steps: tuple[tuple[int, type[object], dict[str, object]], ...],
) -> None:
    for order, cls, init_kwargs in steps:
        services.bind_many(
            key,
            cast("type[object]", cls),
            order=order,
            scope=BindingScope.SINGLETON,
            init_kwargs=init_kwargs,
        )


def _component_instance_key(
    family: ComponentFamily[object],
    instance_id: str,
) -> InjectToken[object]:
    return InjectToken(f"component.{family.name}.{instance_id}")


def _family_contracts() -> dict[str, type[object]]:
    return {
        HTTP_FAMILY.name: HttpClientBase,
        PROVIDER_FAMILY.name: SearchProviderBase,
        FETCH_FAMILY.name: FetcherBase,
        EXTRACT_FAMILY.name: ExtractorBase,
        RANK_FAMILY.name: RankerBase,
        LLM_FAMILY.name: LLMClientBase,
        CACHE_FAMILY.name: CacheBase,
        TELEMETRY_FAMILY.name: TelemetryEmitterBase,
        RATE_LIMIT_FAMILY.name: RateLimiterBase,
    }


def _workunit_overrides(
    overrides: Overrides,
) -> dict[ComponentFamily[object], WorkUnit]:
    mapping = {
        PROVIDER_FAMILY: overrides.provider,
        FETCH_FAMILY: overrides.fetcher,
        EXTRACT_FAMILY: overrides.extractor,
        RANK_FAMILY: overrides.ranker,
        LLM_FAMILY: overrides.llm,
        CACHE_FAMILY: overrides.cache,
        TELEMETRY_FAMILY: overrides.telemetry,
        RATE_LIMIT_FAMILY: overrides.rate_limiter,
    }
    return {
        family: candidate
        for family, candidate in mapping.items()
        if isinstance(candidate, WorkUnit)
    }


def format_key_name(contract: type[object]) -> str:
    return format_service_key(contract)


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
