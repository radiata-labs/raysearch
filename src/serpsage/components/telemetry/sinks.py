from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.base import ComponentConfigBase, ComponentMeta
from serpsage.components.registry import register_component
from serpsage.components.telemetry.base import (
    EventSinkBase,
    JsonlObsConfig,
    SqliteMeteringConfig,
)
from serpsage.models.components.telemetry import EventEnvelope

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

AioSqliteModule: Any | None = None
try:
    AioSqliteModule = importlib.import_module("aiosqlite")
except Exception:  # noqa: BLE001
    AioSqliteModule = None

_NULL_SINK_META = ComponentMeta(
    family="telemetry",
    name="null_sink",
    version="1.0.0",
    summary="No-op telemetry sink.",
    provides=("telemetry.sink",),
    contracts=(EventSinkBase,),
)
_JSONL_SINK_META = ComponentMeta(
    family="telemetry",
    name="jsonl_sink",
    version="1.0.0",
    summary="Append telemetry events to a JSONL file.",
    provides=("telemetry.sink",),
    contracts=(EventSinkBase,),
    config_model=JsonlObsConfig,
)
_SQLITE_SINK_META = ComponentMeta(
    family="telemetry",
    name="sqlite_metering_sink",
    version="1.0.0",
    summary="Write metering events into sqlite.",
    provides=("telemetry.sink",),
    contracts=(EventSinkBase,),
    config_model=SqliteMeteringConfig,
)


@register_component(meta=_NULL_SINK_META)
class NullObsSink(EventSinkBase):
    meta = _NULL_SINK_META

    def __init__(
        self,
        *,
        rt: Runtime,
        config: ComponentConfigBase,
    ) -> None:
        super().__init__(rt=rt, config=config)

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        _ = event


@register_component(meta=_JSONL_SINK_META)
class JsonlObsSink(EventSinkBase):
    meta = _JSONL_SINK_META

    def __init__(
        self,
        *,
        rt: Runtime,
        config: JsonlObsConfig,
    ) -> None:
        super().__init__(rt=rt, config=config)
        self._file_path = str(config.jsonl_path).strip()
        self._file: Any | None = None
        self._lock = anyio.Lock()

    @override
    async def on_init(self) -> None:
        if self._file is not None:
            return
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


@register_component(meta=_SQLITE_SINK_META)
class SqliteMeterLedgerSink(EventSinkBase):
    meta = _SQLITE_SINK_META

    def __init__(
        self,
        *,
        rt: Runtime,
        config: SqliteMeteringConfig,
    ) -> None:
        super().__init__(rt=rt, config=config)
        self._db_path = Path(str(config.sqlite_db_path).strip())
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
        if self._con is None or event.meter is None:
            return
        meter = event.meter
        attrs_json = json.dumps(
            event.attrs.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            await self._con.execute(
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
                    str(event.idempotency_key or event.event_id),
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
            await self._con.commit()

    @override
    async def on_close(self) -> None:
        if self._con is None:
            return
        await self._con.close()
        self._con = None


__all__ = ["JsonlObsSink", "NullObsSink", "SqliteMeterLedgerSink"]
