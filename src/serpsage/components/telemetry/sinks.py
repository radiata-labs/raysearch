from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.telemetry.base import EventSinkBase
from serpsage.models.components.telemetry import EventEnvelope

AioSqliteModule: Any | None = None
try:
    AioSqliteModule = importlib.import_module("aiosqlite")
except Exception:  # noqa: BLE001
    AioSqliteModule = None
if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class NullObsSink(EventSinkBase):
    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        _ = event


class JsonlObsSink(EventSinkBase):
    def __init__(self, *, rt: Runtime, file_path: str) -> None:
        super().__init__(rt=rt)
        self._file_path = str(file_path).strip()
        self._file: Any | None = None
        self._lock = anyio.Lock()

    @override
    async def on_init(self) -> None:
        if self._file is not None:
            return
        if not self._file_path:
            raise ValueError("telemetry.obs.jsonl_path must be non-empty")
        path = Path(self._file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = await anyio.open_file(path, mode="a", encoding="utf-8")

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        if self._file is None:
            return
        payload = event.model_dump(mode="json")
        line = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            await self._file.write(line + "\n")
            await self._file.flush()

    @override
    async def on_close(self) -> None:
        if self._file is None:
            return
        await self._file.aclose()
        self._file = None


class SqliteMeterLedgerSink(EventSinkBase):
    def __init__(self, *, rt: Runtime, db_path: str) -> None:
        super().__init__(rt=rt)
        self._db_path = Path(str(db_path).strip())
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
                event_id TEXT NOT NULL UNIQUE,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL,
                event_name TEXT NOT NULL,
                meter_type TEXT NOT NULL,
                unit TEXT NOT NULL,
                quantity REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                occurred_at_ms INTEGER NOT NULL,
                attrs_json TEXT NOT NULL
            );
            """
        )
        await con.commit()

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        if self._con is None:
            return
        meter = event.meter
        if meter is None:
            return
        con = self._con
        if con is None:
            return
        idempotency_key = str(event.idempotency_key or event.event_id)
        attrs_json = json.dumps(
            event.attrs.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            await con.execute(
                """
                INSERT OR IGNORE INTO metering_ledger(
                    event_id,
                    idempotency_key,
                    request_id,
                    event_name,
                    meter_type,
                    unit,
                    quantity,
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    occurred_at_ms,
                    attrs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.event_id),
                    idempotency_key,
                    str(event.request_id or ""),
                    str(event.event_name),
                    str(meter.meter_type),
                    str(meter.unit),
                    float(meter.quantity),
                    str(meter.provider or ""),
                    str(meter.model or ""),
                    int(meter.prompt_tokens),
                    int(meter.completion_tokens),
                    int(meter.total_tokens),
                    int(event.ts_ms),
                    attrs_json,
                ),
            )
            await con.commit()

    @override
    async def on_close(self) -> None:
        if self._con is None:
            return
        await self._con.close()
        self._con = None


__all__ = ["JsonlObsSink", "NullObsSink", "SqliteMeterLedgerSink"]
