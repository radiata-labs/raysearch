from __future__ import annotations

from contextlib import suppress
from contextvars import ContextVar, Token
from typing_extensions import override

import anyio
from anyio import WouldBlock
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from serpsage.components.base import ComponentConfigBase
from serpsage.components.metering.base import MeteringEmitterBase, MeteringSinkBase
from serpsage.dependencies import Depends
from serpsage.models.components.metering import MeterRecord

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("metering_request_id", default="")
METERING_SINKS_TOKEN = "component.metering_sinks"  # noqa: S105


class MeteringEmitterConfig(ComponentConfigBase):
    __setting_family__ = "metering"
    __setting_name__ = "async_emitter"

    queue_size: int = 2048


class NullMeteringEmitterConfig(ComponentConfigBase):
    __setting_family__ = "metering"
    __setting_name__ = "null_emitter"


class NullMeteringEmitter(MeteringEmitterBase[NullMeteringEmitterConfig]):
    @override
    async def emit_record(self, *, record: MeterRecord) -> None:
        _ = record

    @override
    def push_request_context(self, *, request_id: str) -> object:
        _ = request_id
        return None

    @override
    def pop_request_context(self, token: object) -> None:
        _ = token


class AsyncMeteringEmitter(MeteringEmitterBase[MeteringEmitterConfig]):
    def __init__(
        self,
        *,
        config: MeteringEmitterConfig,
        sinks: tuple[object, ...] = Depends(METERING_SINKS_TOKEN),
    ) -> None:
        self._sinks = [sink for sink in sinks if isinstance(sink, MeteringSinkBase)]
        self._queue_size = max(1, int(config.queue_size))
        self._send: MemoryObjectSendStream[MeterRecord] | None = None
        self._recv: MemoryObjectReceiveStream[MeterRecord] | None = None
        self._tg_cm: TaskGroup | None = None
        self._tg: TaskGroup | None = None

    @override
    async def on_init(self) -> None:
        if self._send is not None:
            return
        send, recv = anyio.create_memory_object_stream[MeterRecord](
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
    async def emit_record(self, *, record: MeterRecord) -> None:
        send = self._send
        if send is None:
            return
        out = self._fill_request_context(record)
        try:
            send.send_nowait(out)
        except WouldBlock:
            if self._start_direct_emit(record=out):
                return
            await send.send(out)

    async def _worker_loop(
        self,
        recv: MemoryObjectReceiveStream[MeterRecord],
    ) -> None:
        async with recv:
            async for record in recv:
                await self._emit_direct(record)

    async def _emit_direct(self, record: MeterRecord) -> None:
        for sink in self._sinks:
            with suppress(Exception):
                await sink.emit(record=record)

    def _start_direct_emit(self, *, record: MeterRecord) -> bool:
        tg = self._tg
        if tg is None:
            return False
        try:
            tg.start_soon(self._emit_direct, record)
            return True
        except Exception:
            return False

    def _fill_request_context(self, record: MeterRecord) -> MeterRecord:
        if str(record.request_id):
            return record
        request_id = str(_REQUEST_ID_CTX.get() or "")
        if not request_id:
            return record
        return record.model_copy(update={"request_id": request_id})


__all__ = [
    "AsyncMeteringEmitter",
    "MeteringEmitterConfig",
    "METERING_SINKS_TOKEN",
    "NullMeteringEmitter",
    "NullMeteringEmitterConfig",
]
