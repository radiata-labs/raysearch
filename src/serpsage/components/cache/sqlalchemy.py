from __future__ import annotations

import re
import zlib
from typing import Any
from typing_extensions import override

from pydantic import field_validator

from serpsage.components.base import ComponentMeta
from serpsage.components.cache.base import CacheBase, CacheConfigBase
from serpsage.load import register_component

_SAFE_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_identifier(value: str) -> str:
    name = str(value or "")
    if not _SAFE_SQL_IDENT_RE.fullmatch(name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


class SQLAlchemyCacheConfig(CacheConfigBase):
    url: str = "sqlite+aiosqlite:///./.serpsage_cache.sqlite3"
    table: str = "cache"

    @field_validator("url", "table")
    @classmethod
    def _validate_strings(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("sqlalchemy cache settings must be non-empty")
        return token


_SQLALCHEMY_CACHE_META = ComponentMeta(
    family="cache",
    name="sqlalchemy",
    version="1.0.0",
    summary="SQLAlchemy async cache backend.",
    provides=("cache.store",),
    config_model=SQLAlchemyCacheConfig,
)


@register_component(meta=_SQLALCHEMY_CACHE_META)
class SQLAlchemyCache(CacheBase[SQLAlchemyCacheConfig]):
    meta = _SQLALCHEMY_CACHE_META

    def __init__(self) -> None:
        try:
            import sqlalchemy as sa  # noqa: PLC0415
            from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError("sqlalchemy is required for SQLAlchemyCache") from exc
        self._url = str(self.config.url)
        self._table_name = _ensure_identifier(str(self.config.table))
        self._validate_async_url(self._url)
        self._sa = sa
        self._create_async_engine = create_async_engine
        self._engine: Any | None = None
        self._metadata = sa.MetaData()
        self._table = sa.Table(
            self._table_name,
            self._metadata,
            sa.Column("namespace", sa.String(255), primary_key=True),
            sa.Column("key", sa.String(512), primary_key=True),
            sa.Column("expires_at_ms", sa.BigInteger(), nullable=False),
            sa.Column("value", sa.LargeBinary(), nullable=False),
            sa.Index("idx_cache_exp", "expires_at_ms"),
        )

    @staticmethod
    def _validate_async_url(url: str) -> None:
        if "://" not in url:
            raise ValueError(f"invalid sqlalchemy cache url: {url!r}")
        scheme = url.split("://", 1)[0]
        if "+" not in scheme:
            raise ValueError(
                "sqlalchemy cache requires async URL with async driver, "
                "e.g. sqlite+aiosqlite:// or mysql+aiomysql://"
            )
        driver = scheme.split("+", 1)[1].lower()
        if "async" not in driver and not driver.startswith("aio"):
            raise ValueError(
                "sqlalchemy cache requires async driver in URL, "
                "e.g. sqlite+aiosqlite:// or mysql+asyncmy://"
            )

    @override
    async def on_init(self) -> None:
        if self._engine is not None:
            return
        self._engine = self._create_async_engine(self._url)
        async with self._engine.begin() as con:
            await con.run_sync(self._metadata.create_all)

    @override
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        if self._engine is None:
            raise RuntimeError("sqlalchemy engine is not initialized")
        now = int(self.clock.now_ms())
        stmt = self._sa.select(self._table.c.expires_at_ms, self._table.c.value).where(
            self._table.c.namespace == namespace,
            self._table.c.key == key,
        )
        delete_stmt = self._sa.delete(self._table).where(
            self._table.c.namespace == namespace,
            self._table.c.key == key,
        )
        async with self._engine.begin() as con:
            row = (await con.execute(stmt)).first()
            if not row:
                return None
            exp_ms = int(row[0])
            blob = row[1]
            if exp_ms <= now:
                await con.execute(delete_stmt)
                return None
        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        return zlib.decompress(bytes(blob))

    @override
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        if ttl_s <= 0:
            return
        if self._engine is None:
            raise RuntimeError("sqlalchemy engine is not initialized")
        now = int(self.clock.now_ms())
        exp_ms = now + int(ttl_s) * 1000
        compressed = zlib.compress(value, level=6)
        delete_one_stmt = self._sa.delete(self._table).where(
            self._table.c.namespace == namespace,
            self._table.c.key == key,
        )
        insert_stmt = self._sa.insert(self._table).values(
            namespace=namespace,
            key=key,
            expires_at_ms=exp_ms,
            value=compressed,
        )
        cleanup_stmt = self._sa.delete(self._table).where(
            self._table.c.expires_at_ms <= now
        )
        async with self._engine.begin() as con:
            await con.execute(delete_one_stmt)
            await con.execute(insert_stmt)
            await con.execute(cleanup_stmt)

    @override
    async def on_close(self) -> None:
        if self._engine is None:
            return
        try:
            await self._engine.dispose()
        finally:
            self._engine = None


__all__ = ["SQLAlchemyCache", "SQLAlchemyCacheConfig"]
