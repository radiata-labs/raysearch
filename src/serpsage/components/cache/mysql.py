from __future__ import annotations

import importlib
import re
import zlib
from typing import Any, Final
from typing_extensions import override

from pydantic import field_validator

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase
from serpsage.load import register_component

_SAFE_SQL_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_identifier(value: str) -> str:
    name = str(value or "")
    if not _SAFE_SQL_IDENT_RE.fullmatch(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


def _quote_ident(ident: str) -> str:
    return f"`{_ensure_identifier(ident)}`"


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


class _SQL:
    __slots__ = (
        "create_table",
        "select_one",
        "delete_one",
        "upsert_one",
        "cleanup_expired",
    )

    def __init__(
        self,
        *,
        create_table: str,
        select_one: str,
        delete_one: str,
        upsert_one: str,
        cleanup_expired: str,
    ) -> None:
        self.create_table = create_table
        self.select_one = select_one
        self.delete_one = delete_one
        self.upsert_one = upsert_one
        self.cleanup_expired = cleanup_expired


class MySQLCacheConfig(CacheConfigBase):
    driver: str = "auto"
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "serpsage"
    table: str = "cache"
    minsize: int = 1
    maxsize: int = 10
    connect_timeout: float = 10.0
    charset: str = "utf8mb4"

    @field_validator("driver", "host", "user", "database", "table", "charset")
    @classmethod
    def _validate_strings(cls, value: str) -> str:
        return str(value or "").strip()


_MYSQL_CACHE_META = ComponentMeta(
    family="cache",
    name="mysql",
    version="1.0.0",
    summary="MySQL-backed cache.",
    provides=("cache.store",),
    config_model=MySQLCacheConfig,
)


@register_component(meta=_MYSQL_CACHE_META)
class MySQLCache(CacheBase[MySQLCacheConfig]):
    meta = _MYSQL_CACHE_META

    def __init__(self) -> None:
        self._driver_pref: Final[str] = str(self.config.driver or "auto").lower()
        self._pool: Any | None = None
        self._table_ident: Final[str] = _ensure_identifier(str(self.config.table))
        self._table_quoted: Final[str] = _quote_ident(self._table_ident)
        self._sql: _SQL | None = None

    def _resolve_driver(self) -> Any:
        pref = self._driver_pref
        if pref == "auto":
            if _module_available("asyncmy"):
                return importlib.import_module("asyncmy")
            if _module_available("aiomysql"):
                return importlib.import_module("aiomysql")
            raise RuntimeError("neither asyncmy nor aiomysql is available")
        if pref not in {"asyncmy", "aiomysql"}:
            raise ValueError(f"unsupported mysql cache driver: {pref}")
        if not _module_available(pref):
            raise RuntimeError(f"{pref} is not available")
        return importlib.import_module(pref)

    def _build_sql(self) -> _SQL:
        t = self._table_quoted
        return _SQL(
            create_table=(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    `namespace` VARCHAR(255) NOT NULL,
                    `key` VARCHAR(512) NOT NULL,
                    `expires_at_ms` BIGINT NOT NULL,
                    `value` LONGBLOB NOT NULL,
                    PRIMARY KEY (`namespace`, `key`),
                    INDEX `idx_cache_exp` (`expires_at_ms`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """.strip()
            ),
            select_one=f"SELECT expires_at_ms, value FROM {t} WHERE namespace=%s AND `key`=%s",
            delete_one=f"DELETE FROM {t} WHERE namespace=%s AND `key`=%s",
            upsert_one=(
                f"INSERT INTO {t} (`namespace`, `key`, `expires_at_ms`, `value`) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "`expires_at_ms` = VALUES(`expires_at_ms`), "
                "`value` = VALUES(`value`)"
            ),
            cleanup_expired=f"DELETE FROM {t} WHERE expires_at_ms <= %s",
        )

    @override
    async def on_init(self) -> None:
        if self._pool is not None:
            return
        cfg = self.config
        mod = self._resolve_driver()
        pool: Any = await mod.create_pool(
            host=str(cfg.host),
            port=int(cfg.port),
            user=str(cfg.user),
            password=str(cfg.password),
            db=str(cfg.database),
            minsize=int(cfg.minsize),
            maxsize=int(cfg.maxsize),
            connect_timeout=float(cfg.connect_timeout),
            charset=str(cfg.charset),
            autocommit=False,
        )
        self._pool = pool
        self._sql = self._build_sql()
        ready_pool, ready_sql = self._require_ready()
        async with ready_pool.acquire() as con:
            async with con.cursor() as cur:
                await cur.execute(ready_sql.create_table)
            await con.commit()

    def _require_ready(self) -> tuple[Any, _SQL]:
        if self._pool is None or self._sql is None:
            raise RuntimeError("mysql cache is not initialized")
        return self._pool, self._sql

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        pool, sql = self._require_ready()
        now = int(self.clock.now_ms())
        async with pool.acquire() as con, con.cursor() as cur:
            await cur.execute(sql.select_one, (namespace, key))
            row = await cur.fetchone()
            if not row:
                return None
            exp_ms, blob = int(row[0]), row[1]
            if exp_ms <= now:
                await cur.execute(sql.delete_one, (namespace, key))
                await con.commit()
                return None
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        try:
            return zlib.decompress(bytes(blob))
        except zlib.error:
            return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        pool, sql = self._require_ready()
        now = int(self.clock.now_ms())
        exp_ms = now + int(ttl_s) * 1000
        compressed = zlib.compress(value, level=6)
        async with pool.acquire() as con:
            async with con.cursor() as cur:
                await cur.execute(sql.upsert_one, (namespace, key, exp_ms, compressed))
                await cur.execute(sql.cleanup_expired, (now,))
            await con.commit()

    @override
    async def on_close(self) -> None:
        pool = self._pool
        if pool is None:
            return
        try:
            pool.close()
            await pool.wait_closed()
        finally:
            self._pool = None
            self._sql = None


__all__ = ["MySQLCache", "MySQLCacheConfig"]
