from __future__ import annotations

from typing import Any, cast

from serpsage.components.telemetry.base import EventSinkBase, TelemetryEmitterBase


def build_telemetry(
    *,
    rt: Any,
    overrides: Any | None = None,
) -> TelemetryEmitterBase:
    _ = overrides
    return cast("TelemetryEmitterBase", rt.services.require(TelemetryEmitterBase))


__all__ = ["EventSinkBase", "TelemetryEmitterBase", "build_telemetry"]
