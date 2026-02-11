from __future__ import annotations

import pytest

import serpsage.components.cache as factory
from serpsage.components.cache.null import NullCache
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


def _rt(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=factory_rt_clock)


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return 0


factory_rt_clock = _Clock()


def test_cache_disabled_uses_null_cache() -> None:
    settings = AppSettings.model_validate({"cache": {"enabled": False}})
    cache = factory.build_cache(rt=_rt(settings))
    assert isinstance(cache, NullCache)


def test_memory_backend_uses_memory_cache() -> None:
    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": "memory"}}
    )
    cache = factory.build_cache(rt=_rt(settings))
    assert cache.__class__.__name__ == "MemoryCache"


@pytest.mark.parametrize("backend", ["sqlite", "redis", "sqlalchemy"])
def test_missing_dependency_fail_fast(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(
        factory,
        "_check_dep",
        lambda module_name, package_name: "missing optional dependency",  # noqa: ARG005
    )
    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": backend}}
    )
    with pytest.raises(RuntimeError, match=f"cache backend `{backend}`"):
        factory.build_cache(rt=_rt(settings))


def test_mysql_missing_driver_fail_fast(monkeypatch) -> None:
    monkeypatch.setattr(
        factory,
        "_pick_mysql_driver",
        lambda *, pref: (None, "no mysql driver"),  # noqa: ARG005
    )
    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": "mysql"}}
    )
    with pytest.raises(RuntimeError, match="cache backend `mysql`"):
        factory.build_cache(rt=_rt(settings))
