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
    ComponentFamily,
    family_collection_token,
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
from serpsage.load import ComponentCatalog
from serpsage.load.components import load_component_descriptors, materialize_settings
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
    settings: AppSettings | dict[str, Any],
    overrides: Overrides | None = None,
) -> Runtime:
    ov = overrides or Overrides()
    descriptors = load_component_descriptors()
    normalized_settings = materialize_settings(
        settings=settings,
        descriptors=descriptors,
    )
    rt = build_runtime(settings=normalized_settings, overrides=ov)
    catalog = ComponentCatalog(
        settings=normalized_settings,
        descriptors=descriptors,
        overrides=ov,
        env=rt.env,
    )
    rt.components = catalog
    provider = _build_services(rt=rt, catalog=catalog, overrides=ov)
    rt.services = provider
    rt.telemetry = cast(
        "TelemetryEmitterBase[Any]",
        provider.require(TelemetryEmitterBase),
    )
    return rt


def build_engine(
    *,
    settings: AppSettings | dict[str, Any],
    overrides: Overrides | None = None,
) -> Engine:
    ov = overrides or Overrides()
    _validate_override_workunits(ov)
    rt = build_component_container(
        settings=settings,
        overrides=ov,
    )
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

    provider = services.build_provider(validate=False)
    provider.bind_instance(ServiceProvider, provider)
    provider.validate()
    return provider


def _register_component_services(
    *,
    services: ServiceCollection,
    catalog: ComponentCatalog,
    overrides: Overrides,
) -> None:
    family_defaults = _family_contracts()
    family_overrides = _workunit_overrides(overrides)
    default_specs = {
        family.name: catalog.resolve_default_spec(family)
        for family in catalog.component_families()
    }
    collection_orders: dict[str, int] = {}
    for family, override_instance in family_overrides.items():
        contract = family_defaults.get(family.name)
        if contract is not None:
            services.bind_instance(contract, override_instance)
            services.bind_many(
                contract,
                override_instance,
                order=_next_collection_order(collection_orders, contract),
            )
        services.bind_instance(type(override_instance), override_instance)
        services.bind_many(
            family_collection_token(family),
            override_instance,
            order=_next_collection_order(
                collection_orders,
                family_collection_token(family),
            ),
        )
    for spec in catalog.iter_enabled_specs():
        if spec.family in family_overrides:
            continue
        instance_key = _component_instance_key(spec.family, spec.component_name)
        instance_binding_id = _single_binding_id(instance_key)
        services.bind_class(
            instance_key,
            spec.descriptor.cls,
            scope=BindingScope.SINGLETON,
            overrides=_component_binding_overrides(config=spec.config),
        )
        services.bind_alias(spec.descriptor.cls, instance_key)
        services.bind_many(
            family_collection_token(spec.family),
            instance_key,
            order=_next_collection_order(
                collection_orders,
                family_collection_token(spec.family),
            ),
            linked_binding_id=instance_binding_id,
        )
        contracts = _component_contracts(spec.descriptor.cls)
        default_contract = family_defaults.get(spec.family.name)
        default_spec = default_specs.get(spec.family.name)
        is_default = default_spec is not None and str(spec.component_name) == str(
            default_spec.component_name
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


def _register_pipeline_services(services: ServiceCollection) -> None:
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


def _bind_steps(
    services: ServiceCollection,
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


def _component_instance_key(
    family: ComponentFamily[Any],
    instance_id: str,
) -> InjectToken[object]:
    return InjectToken(f"component.{family.name}.{instance_id}")


def _component_binding_overrides(
    *,
    config: object,
) -> dict[str, object]:
    return {"config": config}


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


def _workunit_overrides(
    overrides: Overrides,
) -> dict[ComponentFamily[Any], WorkUnit]:
    mapping = {
        PROVIDER_FAMILY: overrides.provider,
        CRAWL_FAMILY: overrides.crawler,
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


def _validate_override_workunits(ov: Overrides) -> None:
    _ensure_workunit_override("cache", ov.cache)
    _ensure_workunit_override("rate_limiter", ov.rate_limiter)
    _ensure_workunit_override("provider", ov.provider)
    _ensure_workunit_override("crawler", ov.crawler)
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
        raise TypeError(f"override `{name}` must have a bootstrapped WorkUnit runtime")


__all__ = ["Overrides", "build_component_container", "build_engine", "build_runtime"]
