from __future__ import annotations

import sqlite3
import zlib
from pathlib import Path
from typing_extensions import override

from anyio import to_thread

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import Cache


class NullCache(WorkUnit, Cache):
    def __init__(self, *, rt) -> None:  # noqa: ANN001
        super().__init__(rt=rt)

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return None

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        return


class SqliteCache(WorkUnit, Cache):
    def __init__(self, *, rt) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._path = Path(self.settings.cache.db_path)
        self._inited = False

    async def _init(self) -> None:
        if self._inited:
            return

        def _do_init() -> None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(self._path))
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute(
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
                con.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(expires_at_ms);"
                )
                con.commit()
            finally:
                con.close()

        await to_thread.run_sync(_do_init)
        self._inited = True

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        await self._init()
        now = int(self.clock.now_ms())

        def _do_get() -> bytes | None:
            con = sqlite3.connect(str(self._path))
            try:
                row = con.execute(
                    "SELECT expires_at_ms, value FROM cache WHERE namespace=? AND key=?",
                    (namespace, key),
                ).fetchone()
                if not row:
                    return None
                exp_ms, blob = int(row[0]), row[1]
                if exp_ms <= now:
                    con.execute(
                        "DELETE FROM cache WHERE namespace=? AND key=?",
                        (namespace, key),
                    )
                    con.commit()
                    return None
                if isinstance(blob, memoryview):
                    blob = blob.tobytes()
                data = bytes(blob)
                return zlib.decompress(data)
            finally:
                con.close()

        return await to_thread.run_sync(_do_get)

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        await self._init()
        exp_ms = int(self.clock.now_ms()) + int(ttl_s) * 1000
        compressed = zlib.compress(value, level=6)

        def _do_set() -> None:
            con = sqlite3.connect(str(self._path))
            try:
                con.execute(
                    "INSERT OR REPLACE INTO cache(namespace,key,expires_at_ms,value) VALUES (?,?,?,?)",
                    (namespace, key, exp_ms, compressed),
                )
                # Opportunistic cleanup.
                con.execute(
                    "DELETE FROM cache WHERE expires_at_ms <= ?",
                    (int(self.clock.now_ms()),),
                )
                con.commit()
            finally:
                con.close()

        await to_thread.run_sync(_do_set)


__all__ = ["NullCache", "SqliteCache"]
