from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.telemetry.base import EventSinkBase, TelemetryEmitterBase
from serpsage.components.telemetry.emitter import (
    AsyncEventEmitter,
    NullTelemetryEmitter,
)
from serpsage.components.telemetry.sinks import (
    JsonlObsSink,
    NullObsSink,
    SqliteMeterLedgerSink,
)

if TYPE_CHECKING:
    from serpsage.core.runtime import Overrides, Runtime


def build_telemetry(
    *,
    rt: Runtime,
    overrides: Overrides | None = None,
) -> TelemetryEmitterBase:
    ov = overrides
    if ov is not None and ov.telemetry is not None:
        return ov.telemetry
    cfg = rt.settings.telemetry
    if not bool(cfg.enabled):
        return NullTelemetryEmitter(rt=rt)
    sinks: list[EventSinkBase] = []
    obs_backend = str(cfg.obs.backend or "null").lower()
    if obs_backend == "jsonl":
        sinks.append(JsonlObsSink(rt=rt, file_path=str(cfg.obs.jsonl_path)))
    else:
        sinks.append(NullObsSink(rt=rt))
    meter_backend = str(cfg.metering.backend or "null").lower()
    if meter_backend == "sqlite":
        sinks.append(
            SqliteMeterLedgerSink(
                rt=rt,
                db_path=str(cfg.metering.sqlite_db_path),
            )
        )
    if not sinks:
        return NullTelemetryEmitter(rt=rt)
    return AsyncEventEmitter(
        rt=rt,
        sinks=sinks,
        queue_size=int(cfg.queue_size),
        drop_noncritical_when_full=bool(cfg.drop_noncritical_when_full),
    )


__all__ = [
    "AsyncEventEmitter",
    "EventSinkBase",
    "JsonlObsSink",
    "NullObsSink",
    "NullTelemetryEmitter",
    "SqliteMeterLedgerSink",
    "TelemetryEmitterBase",
    "build_telemetry",
]
