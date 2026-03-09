from __future__ import annotations

from typing import Any

from serpsage.components.telemetry.base import EventSinkBase, TelemetryEmitterBase


def build_telemetry(
    *,
    rt: Any,
    overrides: Any | None = None,
) -> TelemetryEmitterBase:
    _ = overrides
    return rt.components.resolve_default(
        "telemetry", expected_type=TelemetryEmitterBase
    )


__all__ = ["EventSinkBase", "TelemetryEmitterBase", "build_telemetry"]
