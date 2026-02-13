from __future__ import annotations

import importlib
import re
import zlib
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
)
from typing_extensions import override

from serpsage.contracts.services import CacheBase

if TYPE_CHECKING:
    import aiomysql

    from serpsage.core.runtime import Runtime


_SAFE_SQL_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_identifier(value: str) -> str:
    name = str(value or "")
    if not _SAFE_SQL_IDENT_RE.fullmatch(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


def _quote_ident(ident: str) -> str:
    """
    Quote a SQL identifier for MySQL with backticks.
    Because _ensure_identifier() enforces a strict pattern (no backticks, spaces, dots, etc),
    this is safe and prevents injection via identifier interpolation.
    """
    ident = _ensure_identifier(ident)
    return f"`{ident}`"


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:  # noqa: BLE001
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


class MySQLCache(CacheBase):
    def __init__(self, *, rt: Runtime, driver: str | None = None) -> None:
        super().__init__(rt=rt)
        cfg = self.settings.cache.mysql

        self._driver_pref: Final[str] = str(driver or cfg.driver or "auto").lower()

        self._pool: aiomysql.Pool | None = None

        # identifier sanitization + quoting (prevents injection via table name)
        self._table_ident: Final[str] = _ensure_identifier(str(cfg.table))
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
        # All statements constructed once; afterwards no string-based query construction at call sites.
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
            select_one=f"SELECT expires_at_ms, value FROM {t} WHERE namespace=%s AND `key`=%s",  # noqa: S608
            delete_one=f"DELETE FROM {t} WHERE namespace=%s AND `key`=%s",  # noqa: S608
            upsert_one=(
                f"INSERT INTO {t} (`namespace`, `key`, `expires_at_ms`, `value`) "  # noqa: S608
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "`expires_at_ms` = VALUES(`expires_at_ms`), "
                "`value` = VALUES(`value`)"
            ),
            cleanup_expired=f"DELETE FROM {t} WHERE expires_at_ms <= %s",  # noqa: S608
        )

    @override
    async def on_init(self) -> None:
        if self._pool is not None:
            return

        cfg = self.settings.cache.mysql
        mod = self._resolve_driver()

        pool: aiomysql.Pool = await mod.create_pool(
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
        # Narrow Any -> _Pool via structural typing; runtime behavior still depends on driver.
        self._pool = cast("aiomysql.Pool", pool)
        self._sql = self._build_sql()

        async with self._pool.acquire() as con:
            async with con.cursor() as cur:
                await cur.execute(self._sql.create_table)
            await con.commit()

    def _require_ready(self) -> tuple[aiomysql.Pool, _SQL]:
        if self._pool is None or self._sql is None:
            raise RuntimeError("mysql cache is not initialized")
        return self._pool, self._sql

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        if self._pool is None:
            await self.on_init()
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

        # normalize blob
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        data = bytes(blob)

        try:
            return zlib.decompress(data)
        except zlib.error:
            # Corrupt data: treat as miss (or raise, depending on your preference)
            # Here we choose miss to avoid crashing callers.
            return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        if self._pool is None:
            await self.on_init()
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
            # both asyncmy/aiomysql expose wait_closed()
            await pool.wait_closed()
        finally:
            self._pool = None
            self._sql = None


__all__ = ["MySQLCache"]
