from __future__ import annotations

import contextvars
import uuid
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pydantic import Field

from serpsage.contracts.lifecycle import ClockBase, SpanBase, TelemetryBase
from serpsage.core.model_base import MutableModel

if TYPE_CHECKING:
    from serpsage.settings.models import TelemetrySettings


class NoopSpan(SpanBase):
    @override
    def add_event(self, name: str, **fields: Any) -> None:
        return

    @override
    def set_attr(self, name: str, value: Any) -> None:
        return

    @override
    def end(self) -> None:
        return


class NoopTelemetry(TelemetryBase):
    @override
    def start_span(self, name: str, **attrs: Any) -> SpanBase:
        _ = name, attrs
        return NoopSpan()

    @override
    def summary(self) -> dict[str, Any]:
        return {"enabled": False, "trace_id": "noop", "spans": []}


class SpanRecord(MutableModel):
    span_id: str
    parent_id: str | None
    name: str
    start_ms: int
    end_ms: int | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class TraceSpan(SpanBase):
    def __init__(
        self,
        telemetry: TraceTelemetry,
        record: SpanRecord,
        *,
        stack_token: contextvars.Token[tuple[str, ...]] | None,
    ) -> None:
        self._telemetry = telemetry
        self._record = record
        self._stack_token = stack_token

    @override
    def add_event(self, name: str, **fields: Any) -> None:
        self._record.events.append(
            {"name": name, "t_ms": self._telemetry._now_ms(), "fields": fields}
        )

    @override
    def set_attr(self, name: str, value: Any) -> None:
        self._record.attrs[name] = value

    @override
    def end(self) -> None:
        if self._record.end_ms is None:
            self._record.end_ms = self._telemetry._now_ms()
        self._telemetry._pop_stack(self._record.span_id, self._stack_token)


class TraceTelemetry(TelemetryBase):
    def __init__(self, settings: TelemetrySettings, *, clock: ClockBase) -> None:
        self._settings = settings
        self._clock = clock
        self._trace_id = uuid.uuid4().hex
        self._spans: list[SpanRecord] = []
        self._stack: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
            "serpsage_trace_stack", default=()
        )

    @override
    def start_span(self, name: str, **attrs: Any) -> SpanBase:
        stack = self._stack.get()
        parent_id = stack[-1] if stack else None
        span_id = uuid.uuid4().hex

        rec = SpanRecord(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_ms=self._now_ms(),
            attrs=dict(attrs),
        )
        self._spans.append(rec)

        token: contextvars.Token[tuple[str, ...]] | None = None
        try:
            token = self._stack.set(stack + (span_id,))
        except Exception:
            token = None
        return TraceSpan(self, rec, stack_token=token)

    @override
    def summary(self) -> dict[str, Any]:
        spans = []
        for s in sorted(self._spans, key=lambda r: int(r.start_ms)):
            end_ms = s.end_ms if s.end_ms is not None else self._now_ms()
            spans.append(
                {
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "name": s.name,
                    "duration_ms": max(0, int(end_ms - s.start_ms)),
                    "attrs": s.attrs,
                    "events": s.events if self._settings.include_events else [],
                }
            )
        return {"enabled": True, "trace_id": self._trace_id, "spans": spans}

    def _now_ms(self) -> int:
        return int(self._clock.now_ms())

    def _pop_stack(
        self,
        span_id: str,
        token: contextvars.Token[tuple[str, ...]] | None,
    ) -> None:
        if token is not None:
            try:
                self._stack.reset(token)
                return
            except Exception:
                pass

        stack = self._stack.get()
        if not stack:
            return
        if stack and stack[-1] == span_id:
            try:
                self._stack.set(stack[:-1])
            except Exception:
                return
            return
        if span_id in stack:
            new_stack = tuple(x for x in stack if x != span_id)
            try:
                self._stack.set(new_stack)
            except Exception:
                return


__all__ = ["NoopTelemetry", "TraceTelemetry"]
