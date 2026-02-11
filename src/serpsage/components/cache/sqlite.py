from __future__ import annotations

import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from serpsage.contracts.services import CacheBase

AioSqliteModule: Any | None = None
try:
    import aiosqlite as _aiosqlite

    AioSqliteModule = _aiosqlite
except Exception:  # noqa: BLE001
    AioSqliteModule = None

if TYPE_CHECKING:
    import aiosqlite

    from serpsage.core.runtime import Runtime


class SqliteCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if AioSqliteModule is None:
            raise RuntimeError("aiosqlite is required for SqliteCache")
        self._path = Path(self.settings.cache.sqlite.db_path)
        self._con: aiosqlite.Connection | None = None

    @override
    async def on_init(self) -> None:
        if self._con is not None:
            return
        if AioSqliteModule is None:
            raise RuntimeError("aiosqlite is required for SqliteCache")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._con = await cast("aiosqlite", AioSqliteModule).connect(str(self._path))
        assert self._con is not None
        await self._con.execute("PRAGMA journal_mode=WAL;")
        await self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                expires_at_ms INTEGER NOT NULL,
                value BLOB NOT NULL,
                PRIMARY KEY (namespace, key)
            );
            """
        )
        await self._con.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(expires_at_ms);"
        )
        await self._con.commit()

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        if self._con is None:
            await self.on_init()
        if self._con is None:
            raise RuntimeError("sqlite cache connection is not initialized")

        now = int(self.clock.now_ms())
        cur = await self._con.execute(
            "SELECT expires_at_ms, value FROM cache WHERE namespace=? AND key=?",
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
            await self._con.execute(
                "DELETE FROM cache WHERE namespace=? AND key=?",
                (namespace, key),
            )
            await self._con.commit()
            return None
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        data = bytes(blob)
        return zlib.decompress(data)

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        if self._con is None:
            await self.on_init()
        if self._con is None:
            raise RuntimeError("sqlite cache connection is not initialized")

        exp_ms = int(self.clock.now_ms()) + int(ttl_s) * 1000
        compressed = zlib.compress(value, level=6)
        await self._con.execute(
            "INSERT OR REPLACE INTO cache(namespace,key,expires_at_ms,value) VALUES (?,?,?,?)",
            (namespace, key, exp_ms, compressed),
        )
        await self._con.execute(
            "DELETE FROM cache WHERE expires_at_ms <= ?",
            (int(self.clock.now_ms()),),
        )
        await self._con.commit()

    @override
    async def on_close(self) -> None:
        if self._con is None:
            return
        try:
            await self._con.close()
        finally:
            self._con = None


__all__ = ["SqliteCache"]
