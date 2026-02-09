from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.contracts.protocols import Span, Telemetry

if TYPE_CHECKING:
    from serpsage.settings.models import TelemetrySettings


class NoopSpan(Span):
    @override
    def add_event(self, name: str, **fields: Any) -> None:
        return

    @override
    def set_attr(self, name: str, value: Any) -> None:
        return

    @override
    def end(self) -> None:
        return


class NoopTelemetry(Telemetry):
    @override
    def start_span(self, name: str, **attrs: Any) -> Span:
        return NoopSpan()


@dataclass
class SpanRecord:
    name: str
    start_ms: int
    end_ms: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


class TraceSpan(Span):
    def __init__(self, telemetry: TraceTelemetry, record: SpanRecord) -> None:
        self._telemetry = telemetry
        self._record = record

    @override
    def add_event(self, name: str, **fields: Any) -> None:
        self._record.events.append({"name": name, "t_ms": _now_ms(), "fields": fields})

    @override
    def set_attr(self, name: str, value: Any) -> None:
        self._record.attrs[name] = value

    @override
    def end(self) -> None:
        if self._record.end_ms is None:
            self._record.end_ms = _now_ms()


class TraceTelemetry(Telemetry):
    def __init__(self, settings: TelemetrySettings) -> None:
        self._settings = settings
        self._spans: list[SpanRecord] = []

    @override
    def start_span(self, name: str, **attrs: Any) -> Span:
        rec = SpanRecord(name=name, start_ms=_now_ms(), attrs=dict(attrs))
        self._spans.append(rec)
        return TraceSpan(self, rec)

    def summary(self) -> dict[str, Any]:
        spans = []
        for s in self._spans:
            end_ms = s.end_ms if s.end_ms is not None else _now_ms()
            spans.append(
                {
                    "name": s.name,
                    "duration_ms": max(0, int(end_ms - s.start_ms)),
                    "attrs": s.attrs,
                    "events": s.events if self._settings.include_events else [],
                }
            )
        return {"spans": spans}


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = ["NoopTelemetry", "TraceTelemetry"]
