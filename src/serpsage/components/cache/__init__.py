from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from serpsage.components.cache.base import CacheBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def _check_dep(module_name: str, package_name: str) -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return f"missing optional dependency `{package_name}`"
    return None


def _extract_sqlalchemy_driver(url: str) -> str | None:
    if "://" not in url:
        return None
    scheme = url.split("://", 1)[0]
    if "+" not in scheme:
        return None
    return scheme.split("+", 1)[1].lower()


def _pick_mysql_driver(*, pref: str) -> tuple[str | None, str | None]:
    if pref == "auto":
        err = _check_dep("asyncmy", "asyncmy")
        if err is None:
            return "asyncmy", None
        err2 = _check_dep("aiomysql", "aiomysql")
        if err2 is None:
            return "aiomysql", None
        return None, f"{err}; {err2}"
    if pref == "asyncmy":
        err = _check_dep("asyncmy", "asyncmy")
        return ("asyncmy", None) if err is None else (None, err)
    if pref == "aiomysql":
        err = _check_dep("aiomysql", "aiomysql")
        return ("aiomysql", None) if err is None else (None, err)
    return None, f"unsupported mysql driver `{pref}`"


def build_cache(*, rt: Runtime) -> CacheBase:
    cfg = rt.settings.cache
    if not bool(cfg.enabled):
        from serpsage.components.cache.null import NullCache

        return NullCache(rt=rt)
    backend = (cfg.backend or "sqlite").lower()
    if backend == "memory":
        from serpsage.components.cache.memory import MemoryCache

        return MemoryCache(rt=rt)
    if backend == "sqlite":
        err = _check_dep("aiosqlite", "aiosqlite")
        if err is not None:
            raise RuntimeError(f"cache backend `sqlite` is unavailable: {err}")
        from serpsage.components.cache.sqlite import SqliteCache

        return SqliteCache(rt=rt)
    if backend == "redis":
        err = _check_dep("aioredis", "aioredis")
        if err is not None:
            raise RuntimeError(f"cache backend `redis` is unavailable: {err}")
        from serpsage.components.cache.redis import RedisCache

        return RedisCache(rt=rt)
    if backend == "mysql":
        driver, err = _pick_mysql_driver(pref=str(cfg.mysql.driver or "auto").lower())
        if err is not None or driver is None:
            raise RuntimeError(
                "cache backend `mysql` is unavailable: "
                f"{err or 'unable to select mysql driver'}"
            )
        from serpsage.components.cache.mysql import MySQLCache

        return MySQLCache(rt=rt, driver=driver)
    if backend == "sqlalchemy":
        err = _check_dep("sqlalchemy", "sqlalchemy")
        if err is not None:
            raise RuntimeError(f"cache backend `sqlalchemy` is unavailable: {err}")
        url = str(cfg.sqlalchemy.url)
        driver = _extract_sqlalchemy_driver(url)
        if driver is None:
            raise RuntimeError(
                "cache backend `sqlalchemy` is unavailable: "
                "sqlalchemy cache URL must use async driver"
            )
        if driver == "aiosqlite":
            err = _check_dep("aiosqlite", "aiosqlite")
            if err is not None:
                raise RuntimeError(f"cache backend `sqlalchemy` is unavailable: {err}")
        elif driver == "aiomysql":
            err = _check_dep("aiomysql", "aiomysql")
            if err is not None:
                raise RuntimeError(f"cache backend `sqlalchemy` is unavailable: {err}")
        elif driver == "asyncmy":
            err = _check_dep("asyncmy", "asyncmy")
            if err is not None:
                raise RuntimeError(f"cache backend `sqlalchemy` is unavailable: {err}")
        elif "async" not in driver and not driver.startswith("aio"):
            raise RuntimeError(
                "cache backend `sqlalchemy` is unavailable: "
                f"sqlalchemy driver `{driver}` does not look async"
            )
        from serpsage.components.cache.sqlalchemy import SQLAlchemyCache

        return SQLAlchemyCache(rt=rt)
    raise ValueError(
        f"unsupported cache backend `{backend}`; expected "
        "sqlite|memory|redis|mysql|sqlalchemy"
    )


__all__ = [
    "CacheBase",
    "build_cache",
]
