from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, cast
from typing_extensions import override

from pydantic import field_validator

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase

AioSqliteModule: Any | None = None
try:
    import aiosqlite as _aiosqlite

    AioSqliteModule = _aiosqlite
except Exception:  # noqa: BLE001
    AioSqliteModule = None


class SqliteCacheConfig(CacheConfigBase):
    __setting_family__ = "cache"
    __setting_name__ = "sqlite"

    db_path: str = ".serpsage_cache.sqlite3"
    table: str = "cache"

    @field_validator("db_path", "table")
    @classmethod
    def _validate_strings(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("sqlite cache settings must be non-empty")
        return token


_SQLITE_CACHE_META = ComponentMeta(
    version="1.0.0",
    summary="SQLite-backed cache.",
)


class SqliteCache(CacheBase[SqliteCacheConfig]):
    meta = _SQLITE_CACHE_META

    def __init__(self) -> None:
        if AioSqliteModule is None:
            raise RuntimeError("aiosqlite is required for SqliteCache")
        self._path = Path(self.config.db_path)
        self._table = str(self.config.table)
        self._con: Any | None = None

    @override
    async def on_init(self) -> None:
        if self._con is not None:
            return
        if AioSqliteModule is None:
            raise RuntimeError("aiosqlite is required for SqliteCache")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = await cast("Any", AioSqliteModule).connect(str(self._path))
        con = cast("Any", self._con)
        await con.execute("PRAGMA journal_mode=WAL;")
        await con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                value BLOB NOT NULL,
                PRIMARY KEY (namespace, key)
            );
            """
        )
        await con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self._table}_exp ON {self._table}(expires_at_ms);"
        )
        await con.commit()

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        con = self._con
        if con is None:
            raise RuntimeError("sqlite cache connection is not initialized")
        con = cast("Any", con)
        now = int(self.clock.now_ms())
        cur = await con.execute(
            f"SELECT expires_at_ms, value FROM {self._table} WHERE namespace=? AND key=?",
            (namespace, key),
        )
        try:
            row = await cur.fetchone()
        finally:
            await cur.close()
        if not row:
            return None
        exp_ms, blob = int(row[0]), row[1]
        if exp_ms <= now:
            await con.execute(
                f"DELETE FROM {self._table} WHERE namespace=? AND key=?",
                (namespace, key),
            )
            await con.commit()
            return None
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        return zlib.decompress(bytes(blob))

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        con = self._con
        if con is None:
            raise RuntimeError("sqlite cache connection is not initialized")
        con = cast("Any", con)
        exp_ms = int(self.clock.now_ms()) + int(ttl_s) * 1000
        compressed = zlib.compress(value, level=6)
        await con.execute(
            f"INSERT OR REPLACE INTO {self._table}(namespace,key,expires_at_ms,value) VALUES (?,?,?,?)",
            (namespace, key, exp_ms, compressed),
        )
        await con.execute(
            f"DELETE FROM {self._table} WHERE expires_at_ms <= ?",
            (int(self.clock.now_ms()),),
        )
        await con.commit()

    @override
    async def on_close(self) -> None:
        if self._con is None:
            return
        con = cast("Any", self._con)
        try:
            await con.close()
        finally:
            self._con = None


__all__ = ["SqliteCache", "SqliteCacheConfig"]
