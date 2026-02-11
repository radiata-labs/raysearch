from __future__ import annotations

import pytest

import serpsage.components.cache as factory
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return 0


def _rt(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


class _StubCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    async def aget(self, *, namespace: str, key: str) -> bytes | None:  # noqa: ARG002
        return None

    async def aset(  # noqa: ARG002
        self, *, namespace: str, key: str, value: bytes, ttl_s: int
    ) -> None:
        return


class _StubMySQLCache(_StubCache):
    def __init__(self, *, rt: Runtime, driver: str | None = None) -> None:
        super().__init__(rt=rt)
        self.driver = driver


def test_selects_sqlite_backend_when_dependency_ready(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_check_dep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(factory, "SqliteCache", _StubCache)

    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": "sqlite"}}
    )
    cache = factory.build_cache(rt=_rt(settings))
    assert isinstance(cache, _StubCache)


def test_selects_redis_backend_when_dependency_ready(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_check_dep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(factory, "RedisCache", _StubCache)

    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": "redis"}}
    )
    cache = factory.build_cache(rt=_rt(settings))
    assert isinstance(cache, _StubCache)


def test_selects_sqlalchemy_backend_when_dependency_ready(monkeypatch) -> None:
    monkeypatch.setattr(factory, "_check_dep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(factory, "SQLAlchemyCache", _StubCache)

    settings = AppSettings.model_validate(
        {
            "cache": {
                "enabled": True,
                "backend": "sqlalchemy",
                "sqlalchemy": {"url": "sqlite+aiosqlite:///./cache.db"},
            }
        }
    )
    cache = factory.build_cache(rt=_rt(settings))
    assert isinstance(cache, _StubCache)


def test_pick_mysql_driver_auto_prefers_asyncmy(monkeypatch) -> None:
    def _check(module_name: str, package_name: str) -> str | None:  # noqa: ARG001
        if package_name == "asyncmy":
            return None
        return "not used"

    monkeypatch.setattr(factory, "_check_dep", _check)
    driver, err = factory._pick_mysql_driver(pref="auto")
    assert err is None
    assert driver == "asyncmy"


def test_pick_mysql_driver_auto_falls_back_to_aiomysql(monkeypatch) -> None:
    def _check(module_name: str, package_name: str) -> str | None:  # noqa: ARG001
        if package_name == "asyncmy":
            return "missing asyncmy"
        return None

    monkeypatch.setattr(factory, "_check_dep", _check)
    driver, err = factory._pick_mysql_driver(pref="auto")
    assert err is None
    assert driver == "aiomysql"


def test_build_cache_passes_selected_mysql_driver(monkeypatch) -> None:
    monkeypatch.setattr(
        factory, "_pick_mysql_driver", lambda *, pref: ("asyncmy", None)
    )
    monkeypatch.setattr(factory, "MySQLCache", _StubMySQLCache)

    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "backend": "mysql"}}
    )
    cache = factory.build_cache(rt=_rt(settings))
    assert isinstance(cache, _StubMySQLCache)
    assert cache.driver == "asyncmy"


def test_pick_mysql_driver_auto_returns_error_when_none_available(monkeypatch) -> None:
    monkeypatch.setattr(
        factory,
        "_check_dep",
        lambda module_name, package_name: f"missing {package_name}",  # noqa: ARG005
    )
    driver, err = factory._pick_mysql_driver(pref="auto")
    assert driver is None
    assert err is not None


@pytest.mark.parametrize("pref", ["asyncmy", "aiomysql"])
def test_pick_mysql_driver_respects_explicit_driver(monkeypatch, pref: str) -> None:
    monkeypatch.setattr(factory, "_check_dep", lambda *_args, **_kwargs: None)
    driver, err = factory._pick_mysql_driver(pref=pref)
    assert err is None
    assert driver == pref
