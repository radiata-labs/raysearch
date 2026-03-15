from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast
from typing_extensions import override

import anyio
from pydantic import field_validator

from serpsage.components.base import ComponentConfigBase
from serpsage.components.metering.base import MeteringSinkBase
from serpsage.models.components.metering import MeterRecord

AioSqliteModule: Any | None = None
try:
    AioSqliteModule = importlib.import_module("aiosqlite")
except Exception:  # noqa: BLE001
    AioSqliteModule = None


class SqliteMeteringSinkConfig(ComponentConfigBase):
    __setting_family__ = "metering"
    __setting_name__ = "sqlite"

    sqlite_db_path: str = ".serpsage_metering.sqlite3"

    @field_validator("sqlite_db_path")
    @classmethod
    def _validate_sqlite_db_path(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("metering sqlite_db_path must be non-empty")
        return token


class SqliteMeterLedgerSink(MeteringSinkBase[SqliteMeteringSinkConfig]):
    """Metering sink that writes records to a SQLite database."""

    def __init__(self) -> None:
        self._db_path = Path(str(self.config.sqlite_db_path).strip())
        self._con: Any | None = None
        self._lock = anyio.Lock()

    @override
    async def on_init(self) -> None:
        if self._con is not None:
            return
        if AioSqliteModule is None:
            raise RuntimeError("aiosqlite is required for SqliteMeterLedgerSink")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        factory = cast("Any", AioSqliteModule)
        self._con = await factory.connect(str(self._db_path))
        con = self._con
        if con is None:
            raise RuntimeError("sqlite metering connection is not initialized")
        await con.execute("PRAGMA journal_mode=WAL;")
        await con.execute(
            """
            CREATE TABLE IF NOT EXISTS metering_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT NOT NULL UNIQUE,
                record_key TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                occurred_at_ms INTEGER NOT NULL
            );
            """
        )
        await con.commit()

    @override
    async def emit(self, *, record: MeterRecord) -> None:
        if self._con is None:
            return
        async with self._lock:
            await self._con.execute(
                """
                INSERT OR IGNORE INTO metering_ledger(
                    record_id,
                    record_key,
                    request_id,
                    name,
                    unit,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    occurred_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.id),
                    str(record.key or record.id),
                    str(record.request_id or ""),
                    str(record.name),
                    str(record.unit),
                    str(record.provider or ""),
                    str(record.model or ""),
                    int(
                        record.tokens.prompt_tokens if record.tokens is not None else 0
                    ),
                    int(
                        record.tokens.completion_tokens
                        if record.tokens is not None
                        else 0
                    ),
                    int(record.tokens.total_tokens if record.tokens is not None else 0),
                    int(record.ts_ms),
                ),
            )
            await self._con.commit()

    @override
    async def on_close(self) -> None:
        if self._con is None:
            return
        await self._con.close()
        self._con = None


__all__ = ["SqliteMeterLedgerSink", "SqliteMeteringSinkConfig"]
