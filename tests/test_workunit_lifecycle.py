from __future__ import annotations

import httpx
import pytest

from serpsage import Engine
from serpsage.components.fetch.http_client_unit import HttpClientUnit
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import SearchProviderBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


class _TraceUnit(WorkUnit):
    def __init__(
        self,
        *,
        rt: Runtime,
        name: str,
        events: list[str],
        fail_init: bool = False,
        fail_close: bool = False,
    ) -> None:
        super().__init__(rt=rt)
        self._name = name
        self._events = events
        self._fail_init = fail_init
        self._fail_close = fail_close

    async def on_init(self) -> None:
        self._events.append(f"init:{self._name}")
        if self._fail_init:
            raise RuntimeError(f"init failed: {self._name}")

    async def on_close(self) -> None:
        self._events.append(f"close:{self._name}")
        if self._fail_close:
            raise RuntimeError(f"close failed: {self._name}")


class _BadProvider(SearchProviderBase):
    # Intentionally missing super().__init__(rt=...).
    def __init__(self) -> None:
        self._items: list[dict[str, str]] = []

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


def _rt(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_FakeClock())


@pytest.mark.anyio
async def test_shared_dependency_inits_once_and_closes_in_reverse_order():
    settings = AppSettings.model_validate({})
    rt = _rt(settings)
    events: list[str] = []

    shared = _TraceUnit(rt=rt, name="shared", events=events)
    left = _TraceUnit(rt=rt, name="left", events=events)
    right = _TraceUnit(rt=rt, name="right", events=events)
    root = _TraceUnit(rt=rt, name="root", events=events)
    left.bind_dep(shared)
    right.bind_dep(shared)
    root.bind_deps(left, right)

    await root.ainit()
    await root.aclose()

    assert events == [
        "init:shared",
        "init:left",
        "init:right",
        "init:root",
        "close:root",
        "close:right",
        "close:left",
        "close:shared",
    ]
    assert events.count("init:shared") == 1
    assert events.count("close:shared") == 1


@pytest.mark.anyio
async def test_lifecycle_is_idempotent_and_closed_state_is_final():
    settings = AppSettings.model_validate({})
    rt = _rt(settings)
    events: list[str] = []
    root = _TraceUnit(rt=rt, name="root", events=events)

    await root.ainit()
    await root.ainit()
    await root.aclose()
    await root.aclose()

    assert events == ["init:root", "close:root"]
    with pytest.raises(RuntimeError, match="already closed"):
        await root.ainit()


@pytest.mark.anyio
async def test_init_failure_rolls_back_initialized_nodes():
    settings = AppSettings.model_validate({})
    rt = _rt(settings)
    events: list[str] = []

    good = _TraceUnit(rt=rt, name="good", events=events)
    bad = _TraceUnit(rt=rt, name="bad", events=events, fail_init=True)
    root = _TraceUnit(rt=rt, name="root", events=events)
    root.bind_deps(good, bad)

    with pytest.raises(RuntimeError, match="init failed: bad"):
        await root.ainit()

    assert events == ["init:good", "init:bad", "close:good"]


@pytest.mark.anyio
async def test_close_aggregates_errors_and_keeps_closing():
    settings = AppSettings.model_validate({})
    rt = _rt(settings)
    events: list[str] = []

    a = _TraceUnit(rt=rt, name="a", events=events, fail_close=True)
    b = _TraceUnit(rt=rt, name="b", events=events, fail_close=True)
    root = _TraceUnit(rt=rt, name="root", events=events)
    root.bind_deps(a, b)
    await root.ainit()

    with pytest.raises(ExceptionGroup) as exc:
        await root.aclose()
    assert len(exc.value.exceptions) == 2
    assert "close:root" in events
    assert "close:a" in events
    assert "close:b" in events


def test_build_engine_rejects_non_bootstrapped_override_workunit():
    settings = AppSettings.model_validate({"enrich": {"enabled": False}})
    with pytest.raises(TypeError, match="WorkUnit.__init__"):
        Engine.from_settings(
            settings,
            overrides=Overrides(provider=_BadProvider()),
        )


@pytest.mark.anyio
async def test_http_client_unit_ownership_controls_close_behavior():
    settings = AppSettings.model_validate({})
    rt = _rt(settings)

    owned_client = httpx.AsyncClient()
    owned = HttpClientUnit(rt=rt, client=owned_client, owns_client=True)
    await owned.aclose()
    assert owned_client.is_closed is True

    external_client = httpx.AsyncClient()
    external = HttpClientUnit(rt=rt, client=external_client, owns_client=False)
    await external.aclose()
    assert external_client.is_closed is False
    await external_client.aclose()
