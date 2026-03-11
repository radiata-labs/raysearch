from __future__ import annotations

import uuid
from contextlib import suppress
from contextvars import ContextVar, Token
from typing import Any
from typing_extensions import override

import anyio
from anyio import WouldBlock
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from serpsage.components.base import ComponentConfigBase
from serpsage.components.telemetry.base import (
    EventSinkBase,
    TelemetryEmitterBase,
)
from serpsage.dependencies import Depends
from serpsage.models.components.telemetry import (
    EventAttributes,
    EventEnvelope,
    is_critical_event,
)

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("telemetry_request_id", default="")
TELEMETRY_SINKS_TOKEN = "component.telemetry_sinks"  # noqa: S105


class TelemetryEmitterConfig(ComponentConfigBase):
    __setting_family__ = "telemetry"
    __setting_name__ = "async_emitter"

    queue_size: int = 2048
    drop_noncritical_when_full: bool = True


class NullTelemetryEmitterConfig(ComponentConfigBase):
    __setting_family__ = "telemetry"
    __setting_name__ = "null_emitter"


class NullTelemetryEmitter(TelemetryEmitterBase[NullTelemetryEmitterConfig]):
    @override
    async def emit_event(self, *, event: EventEnvelope) -> None:
        _ = event

    @override
    def push_request_context(self, *, request_id: str) -> object:
        _ = request_id
        return None

    @override
    def pop_request_context(self, token: object) -> None:
        _ = token


class AsyncEventEmitter(TelemetryEmitterBase[TelemetryEmitterConfig]):
    def __init__(
        self,
        *,
        config: TelemetryEmitterConfig,
        sinks: tuple[EventSinkBase[Any], ...] = Depends(TELEMETRY_SINKS_TOKEN),
    ) -> None:
        self._sinks = list(sinks)
        self._queue_size = max(1, int(config.queue_size))
        self._drop_noncritical_when_full = bool(config.drop_noncritical_when_full)
        self._send: MemoryObjectSendStream[EventEnvelope] | None = None
        self._recv: MemoryObjectReceiveStream[EventEnvelope] | None = None
        self._tg_cm: TaskGroup | None = None
        self._tg: TaskGroup | None = None
        self._dropped_noncritical = 0

    @override
    async def on_init(self) -> None:
        if self._send is not None:
            return
        send, recv = anyio.create_memory_object_stream[EventEnvelope](
            max_buffer_size=self._queue_size
        )
        tg_cm = anyio.create_task_group()
        tg = await tg_cm.__aenter__()
        tg.start_soon(self._worker_loop, recv.clone())
        self._send = send
        self._recv = recv
        self._tg_cm = tg_cm
        self._tg = tg

    @override
    async def on_close(self) -> None:
        send = self._send
        recv = self._recv
        tg_cm = self._tg_cm
        self._send = None
        self._recv = None
        self._tg = None
        self._tg_cm = None
        if send is not None:
            await send.aclose()
        if tg_cm is not None:
            await tg_cm.__aexit__(None, None, None)
        if recv is not None:
            await recv.aclose()

    @override
    def push_request_context(self, *, request_id: str) -> object:
        return _REQUEST_ID_CTX.set(str(request_id or ""))

    @override
    def pop_request_context(self, token: object) -> None:
        if isinstance(token, Token):
            _REQUEST_ID_CTX.reset(token)

    @override
    async def emit_event(self, *, event: EventEnvelope) -> None:
        send = self._send
        if send is None:
            return
        out = self._fill_request_context(event)
        critical = is_critical_event(event_name=str(out.event_name), status=out.status)
        await self._flush_drop_notice_if_any(
            send=send, for_event=out, force=not critical
        )
        if critical:
            try:
                send.send_nowait(out)
                return
            except WouldBlock:
                if self._start_direct_emit(event=out):
                    return
            await send.send(out)
            return
        if not self._drop_noncritical_when_full:
            await send.send(out)
            return
        try:
            send.send_nowait(out)
        except WouldBlock:
            self._dropped_noncritical += 1
            await self._flush_drop_notice_if_any(send=send, for_event=out, force=False)

    async def _worker_loop(
        self, recv: MemoryObjectReceiveStream[EventEnvelope]
    ) -> None:
        async with recv:
            async for event in recv:
                await self._emit_direct(event)

    async def _emit_direct(self, event: EventEnvelope) -> None:
        for sink in self._sinks:
            with suppress(Exception):
                await sink.emit(event=event)

    def _start_direct_emit(self, *, event: EventEnvelope) -> bool:
        tg = self._tg
        if tg is None:
            return False
        try:
            tg.start_soon(self._emit_direct, event)
            return True
        except Exception:
            return False

    def _fill_request_context(self, event: EventEnvelope) -> EventEnvelope:
        if str(event.request_id):
            return event
        request_id = str(_REQUEST_ID_CTX.get() or "")
        if not request_id:
            return event
        return event.model_copy(update={"request_id": request_id})

    async def _flush_drop_notice_if_any(
        self,
        *,
        send: MemoryObjectSendStream[EventEnvelope],
        for_event: EventEnvelope,
        force: bool = True,
    ) -> None:
        dropped = int(self._dropped_noncritical)
        if dropped <= 0:
            return
        if str(for_event.event_name) == "telemetry.dropped":
            return
        notice = EventEnvelope(
            event_id=uuid.uuid4().hex,
            idempotency_key="",
            event_name="telemetry.dropped",
            status="error",
            ts_ms=int(self.clock.now_ms()),
            request_id=str(for_event.request_id or ""),
            component="telemetry",
            stage="emitter",
            attrs=EventAttributes(values={"dropped_noncritical_events": dropped}),
        )
        if force:
            await send.send(notice)
            self._dropped_noncritical = 0
            return
        try:
            send.send_nowait(notice)
        except WouldBlock:
            return
        self._dropped_noncritical = 0


__all__ = [
    "AsyncEventEmitter",
    "NullTelemetryEmitter",
    "NullTelemetryEmitterConfig",
    "TELEMETRY_SINKS_TOKEN",
    "TelemetryEmitterConfig",
]
