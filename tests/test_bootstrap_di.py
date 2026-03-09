from __future__ import annotations

from serpsage.app.bootstrap import build_component_container, build_engine
from serpsage.app.engine import Engine
from serpsage.app.tokens import (
    CHILD_FETCH_RUNNER,
    FETCH_RUNNER,
    RESEARCH_ROUND_RUNNER,
    RESEARCH_SUBREPORT_STEP,
    SEARCH_RUNNER,
)
from serpsage.components.fetch.base import FetcherBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings


def test_bootstrap_builds_runtime_services_and_defaults() -> None:
    runtime, catalog = build_component_container(settings=AppSettings())
    services = runtime.services

    assert isinstance(runtime, Runtime)
    assert services is not None
    assert runtime.components is catalog
    assert isinstance(runtime.telemetry, TelemetryEmitterBase)
    assert catalog.family_name("provider") == "searxng"
    assert services.require(Runtime) is runtime
    assert isinstance(services.require(SearchProviderBase), SearchProviderBase)
    assert isinstance(services.require(FetcherBase), FetcherBase)
    assert isinstance(services.require(RankerBase), RankerBase)


def test_bootstrap_builds_pipeline_runners_and_engine() -> None:
    runtime, _catalog = build_component_container(settings=AppSettings())
    services = runtime.services
    assert services is not None

    child_fetch_runner = services.require(CHILD_FETCH_RUNNER)
    fetch_runner = services.require(FETCH_RUNNER)
    search_runner = services.require(SEARCH_RUNNER)
    research_round_runner = services.require(RESEARCH_ROUND_RUNNER)
    subreport_step = services.require(RESEARCH_SUBREPORT_STEP)
    engine = build_engine(settings=AppSettings())

    assert child_fetch_runner is services.require(CHILD_FETCH_RUNNER)
    assert fetch_runner is services.require(FETCH_RUNNER)
    assert search_runner is services.require(SEARCH_RUNNER)
    assert research_round_runner is services.require(RESEARCH_ROUND_RUNNER)
    assert subreport_step is services.require(RESEARCH_SUBREPORT_STEP)
    assert isinstance(engine, Engine)
