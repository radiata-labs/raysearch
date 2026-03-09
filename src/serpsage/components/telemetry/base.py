from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.telemetry import (
    EventAttributes,
    EventEnvelope,
    EventStatus,
    MeterPayload,
    sanitize_attr_map,
)

EventSinkConfigT = TypeVar("EventSinkConfigT", bound=ComponentConfigBase)
TelemetryEmitterConfigT = TypeVar("TelemetryEmitterConfigT", bound=ComponentConfigBase)


class EventSinkBase(ComponentBase[EventSinkConfigT], ABC, Generic[EventSinkConfigT]):
    @abstractmethod
    async def emit(self, *, event: EventEnvelope) -> None:
        raise NotImplementedError


class TelemetryEmitterBase(
    ComponentBase[TelemetryEmitterConfigT], ABC, Generic[TelemetryEmitterConfigT]
):
    @abstractmethod
    async def emit_event(self, *, event: EventEnvelope) -> None:
        raise NotImplementedError

    @abstractmethod
    def push_request_context(self, *, request_id: str) -> object:
        raise NotImplementedError

    @abstractmethod
    def pop_request_context(self, token: object) -> None:
        raise NotImplementedError

    async def emit(
        self,
        *,
        event_name: str,
        status: EventStatus = "ok",
        request_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        component: str = "",
        stage: str = "",
        duration_ms: int | None = None,
        error_code: str = "",
        error_type: str = "",
        attrs: dict[str, Any] | None = None,
        meter: MeterPayload | None = None,
        idempotency_key: str = "",
    ) -> None:
        event = EventEnvelope(
            event_id=uuid.uuid4().hex,
            idempotency_key=str(idempotency_key or ""),
            event_name=str(event_name),
            status=status,
            ts_ms=int(self.clock.now_ms()),
            request_id=str(request_id or ""),
            trace_id=str(trace_id or ""),
            span_id=str(span_id or ""),
            component=str(component or ""),
            stage=str(stage or ""),
            duration_ms=duration_ms,
            error_code=str(error_code or ""),
            error_type=str(error_type or ""),
            attrs=EventAttributes(values=sanitize_attr_map(attrs)),
            meter=meter,
        )
        await self.emit_event(event=event)


__all__ = [
    "EventSinkBase",
    "TelemetryEmitterBase",
]
