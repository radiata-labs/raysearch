from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar
from typing_extensions import override

import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream, TaskGroup

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.models.steps.base import BaseStepContext

    TContext = TypeVar("TContext", bound="BaseStepContext[Any, Any]")
else:
    TContext = TypeVar("TContext")


class StepBase(WorkUnit, ABC, Generic[TContext]):
    async def run(self, ctx: TContext) -> TContext:
        request_id = str(getattr(ctx, "request_id", "") or "anonymous")
        step_name = type(self).__name__
        started_ms = int(self.clock.now_ms())
        await self._emit_step_event(
            event_name="step.start",
            request_id=request_id,
            step_name=step_name,
            status="start",
            attrs={"step": step_name},
        )
        try:
            out = await self.run_inner(ctx)
            if out is None:
                raise RuntimeError(f"{step_name} returned no context")
            await self._emit_step_event(
                event_name="step.end",
                request_id=request_id,
                step_name=step_name,
                status="ok",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "step": step_name,
                },
            )
            return out
        except Exception as exc:  # noqa: BLE001
            await self._emit_step_event(
                event_name="step.error",
                request_id=request_id,
                step_name=step_name,
                status="error",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_code="step_failed",
                error_type=type(exc).__name__,
                attrs={
                    "step": step_name,
                    "fatal_step": True,
                    "error_message": str(exc),
                },
            )
            raise

    async def _emit_step_event(
        self,
        *,
        event_name: str,
        request_id: str,
        step_name: str,
        status: Literal["start", "ok", "error"],
        duration_ms: int | None = None,
        error_code: str = "",
        error_type: str = "",
        attrs: dict[str, object] | None = None,
    ) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(
                event_name=event_name,
                status=status,
                request_id=request_id,
                component="step",
                stage=step_name,
                duration_ms=duration_ms,
                error_code=error_code,
                error_type=error_type,
                attrs=attrs,
            )

    async def emit_tracking_event(
        self,
        *,
        event_name: str,
        request_id: str,
        stage: str,
        status: Literal["ok", "error"] = "ok",
        error_code: str = "",
        error_type: str = "",
        attrs: dict[str, object] | None = None,
    ) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(
                event_name=event_name,
                status="error" if status == "error" else "ok",
                request_id=request_id,
                component="step",
                stage=stage,
                error_code=error_code,
                error_type=error_type,
                attrs=attrs,
            )

    @abstractmethod
    async def run_inner(self, ctx: TContext) -> TContext:
        raise NotImplementedError


RunnerKind = Literal["search", "fetch", "child_fetch"]
RunnerLifecycleState = Literal["created", "running", "draining", "closed"]


@dataclass(slots=True)
class _RunnerTask(Generic[TContext]):
    task_id: str
    request_id: str
    ctx: TContext


class RunnerBase(WorkUnit, Generic[TContext]):
    def __init__(
        self,
        *,
        steps: list[StepBase[TContext]],
        kind: RunnerKind = "fetch",
    ) -> None:
        self._steps = list(steps)
        self._kind = kind
        self._seq = 0
        self._state: RunnerLifecycleState = "created"
        self._accepting = False
        self._send: ObjectSendStream[_RunnerTask[TContext]] | None = None
        self._recv: ObjectReceiveStream[_RunnerTask[TContext]] | None = None
        self._tg_cm: TaskGroup | None = None
        self._tg: TaskGroup | None = None
        self._results: dict[str, TContext | Exception] = {}
        self._result_events: dict[str, anyio.Event] = {}
        self._orphan_task_ids: set[str] = set()
        self._infra_error: Exception | None = None
        self._seq_lock = anyio.Lock()
        self._state_lock = anyio.Lock()
        self._orphan_lock = anyio.Lock()
        self._result_lock = self._orphan_lock
        self.bind_deps(*steps)

    @override
    async def on_init(self) -> None:
        async with self._state_lock:
            if self._state == "running":
                return
            if self._state == "closed":
                raise RuntimeError("runner is closed and cannot be initialized again")
            if self._state == "draining":
                raise RuntimeError("runner is draining and cannot be initialized")
            send, recv = anyio.create_memory_object_stream[_RunnerTask[TContext]](
                max_buffer_size=self._queue_size()
            )
            tg_cm = anyio.create_task_group()
            try:
                tg = await tg_cm.__aenter__()
            except BaseException:
                await send.aclose()
                await recv.aclose()
                raise
            try:
                for _ in range(self._max_concurrency()):
                    tg.start_soon(self._worker_loop, recv.clone())
            except BaseException:
                tg.cancel_scope.cancel()
                await tg_cm.__aexit__(None, None, None)
                await send.aclose()
                await recv.aclose()
                raise
            self._send = send
            self._recv = recv
            self._tg_cm = tg_cm
            self._tg = tg
            self._accepting = True
            self._infra_error = None
            self._results.clear()
            self._result_events.clear()
            self._orphan_task_ids.clear()
            self._state = "running"

    @override
    async def on_close(self) -> None:
        send: ObjectSendStream[_RunnerTask[TContext]] | None = None
        recv: ObjectReceiveStream[_RunnerTask[TContext]] | None = None
        tg_cm: TaskGroup | None = None
        async with self._state_lock:
            if self._state == "closed":
                return
            if self._state == "created":
                self._accepting = False
                self._infra_error = None
                self._results.clear()
                self._result_events.clear()
                self._orphan_task_ids.clear()
                self._state = "closed"
                return
            self._state = "draining"
            self._accepting = False
            send = self._send
            recv = self._recv
            tg_cm = self._tg_cm
            self._send = None
            self._recv = None
            self._tg = None
            self._tg_cm = None
        try:
            if send is not None:
                await send.aclose()
            if tg_cm is not None:
                await tg_cm.__aexit__(None, None, None)
            if recv is not None:
                await recv.aclose()
        finally:
            async with self._state_lock:
                self._accepting = False
                self._infra_error = None
                self._orphan_task_ids.clear()
                self._state = "closed"

    async def run(self, ctx: TContext) -> TContext:
        await self._ensure_running()
        request_id = str(getattr(ctx, "request_id", "") or "anonymous")
        task_id: str | None = None
        pending: set[str] = set()
        try:
            task_id = await self._push(ctx=ctx)
            pending.add(task_id)
            item = await self._wait_and_get(task_id=task_id)
            if item is None:
                raise RuntimeError(f"runner result missing for task_id={task_id}")
            pending.clear()
        except BaseException as exc:
            if pending:
                with anyio.CancelScope(shield=True):
                    await self._mark_orphans(task_ids=pending)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RuntimeError):
                raise
            task_hint = task_id or f"{request_id}#unknown"
            raise RuntimeError(
                f"runner execution failed for task_id={task_hint} request_id={request_id}"
            ) from exc
        if isinstance(item, Exception):
            request_id = task_id.split("#", 1)[0]
            raise RuntimeError(  # noqa: TRY004
                f"runner worker failed for task_id={task_id} request_id={request_id}"
            ) from item
        return item

    async def run_batch(self, contexts: list[TContext]) -> list[TContext]:
        await self._ensure_running()
        if not contexts:
            return []
        task_ids: list[str] = []
        pending: set[str] = set()
        try:
            for ctx in contexts:
                task_id = await self._push(ctx=ctx)
                task_ids.append(task_id)
                pending.add(task_id)
            done: dict[str, TContext | Exception] = {}
            for task_id in task_ids:
                item = await self._wait_and_get(task_id=task_id)
                if item is None:
                    raise RuntimeError(f"runner result missing for task_id={task_id}")
                done[task_id] = item
                pending.discard(task_id)
            return [self._into_result(tid, done[tid]) for tid in task_ids]
        except BaseException as exc:
            if pending:
                with anyio.CancelScope(shield=True):
                    await self._mark_orphans(task_ids=pending)
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RuntimeError):
                raise
            if task_ids:
                task_hint = task_ids[0]
                request_id = task_hint.split("#", 1)[0]
            elif contexts:
                request_id = str(getattr(contexts[0], "request_id", "") or "anonymous")
                task_hint = f"{request_id}#unknown"
            else:
                request_id = "anonymous"
                task_hint = "anonymous#unknown"
            raise RuntimeError(
                f"runner batch execution failed for task_id={task_hint} request_id={request_id}"
            ) from exc

    def _queue_size(self) -> int:
        return max(1, int(self.settings.runner.queue_size))

    def _max_concurrency(self) -> int:
        cfg = self.settings.runner
        if self._kind == "search":
            return max(1, int(cfg.search_limit))
        if self._kind == "child_fetch":
            return max(1, int(cfg.child_fetch_limit))
        return max(1, int(cfg.fetch_limit))

    async def _ensure_running(self) -> None:
        async with self._state_lock:
            state = self._state
            accepting = self._accepting
            infra_error = self._infra_error
        if state != "running" or not accepting:
            raise RuntimeError(f"runner is not running (state={state})")
        if infra_error is not None:
            raise RuntimeError("runner infrastructure failed") from infra_error

    async def _record_infra_error(self, exc: Exception) -> None:
        async with self._state_lock:
            if self._infra_error is None:
                self._infra_error = exc
            self._accepting = False
        events_to_notify: list[anyio.Event] = []
        async with self._orphan_lock:
            for task_id, event in list(self._result_events.items()):
                self._results.setdefault(
                    task_id,
                    RuntimeError("runner infrastructure failed"),
                )
                events_to_notify.append(event)
        for event in events_to_notify:
            with suppress(Exception):
                event.set()

    async def _mark_orphans(self, *, task_ids: set[str]) -> None:
        if not task_ids:
            return
        events_to_notify: list[anyio.Event] = []
        async with self._orphan_lock:
            self._orphan_task_ids.update(task_ids)
            for task_id in task_ids:
                if task_id not in self._results:
                    event = self._result_events.pop(task_id, None)
                    if event is not None:
                        events_to_notify.append(event)
        for event in events_to_notify:
            with suppress(Exception):
                event.set()

    async def _store_result(self, *, task_id: str, item: TContext | Exception) -> None:
        event: anyio.Event | None = None
        async with self._orphan_lock:
            if task_id in self._orphan_task_ids:
                self._orphan_task_ids.remove(task_id)
            else:
                self._results[task_id] = item
            event = self._result_events.get(task_id)
        if event is not None:
            event.set()

    async def _worker_loop(
        self,
        recv: ObjectReceiveStream[_RunnerTask[TContext]],
    ) -> None:
        try:
            async with recv:
                while True:
                    try:
                        task = await recv.receive()
                    except anyio.EndOfStream:
                        return
                    try:
                        out = task.ctx
                        for step in self._steps:
                            out = await step.run(out)
                        result: TContext | Exception = out
                    except Exception as exc:  # noqa: BLE001
                        result = (
                            exc if isinstance(exc, Exception) else Exception(str(exc))
                        )
                    await self._store_result(task_id=task.task_id, item=result)
        except Exception as exc:  # noqa: BLE001
            await self._record_infra_error(exc)
            raise

    async def _push(self, *, ctx: TContext) -> str:
        request_id = str(getattr(ctx, "request_id", "") or "anonymous")
        async with self._seq_lock:
            task_id = f"{request_id}#{self._seq}"
            self._seq += 1
        async with self._state_lock:
            state = self._state
            accepting = self._accepting
            send = self._send
            infra_error = self._infra_error
        if state != "running" or not accepting or send is None:
            raise RuntimeError(
                f"runner enqueue rejected for task_id={task_id} request_id={request_id} state={state}"
            )
        if infra_error is not None:
            raise RuntimeError(
                f"runner enqueue rejected for task_id={task_id} request_id={request_id}"
            ) from infra_error
        async with self._orphan_lock:
            self._result_events[task_id] = anyio.Event()
        try:
            await send.send(
                _RunnerTask[TContext](task_id=task_id, request_id=request_id, ctx=ctx)
            )
        except Exception as exc:  # noqa: BLE001
            async with self._orphan_lock:
                self._result_events.pop(task_id, None)
            raise RuntimeError(
                f"runner enqueue failed for task_id={task_id} request_id={request_id}"
            ) from exc
        return task_id

    async def _wait_and_get(self, *, task_id: str) -> TContext | Exception | None:
        """Wait for result event and atomically retrieve and cleanup."""
        event: anyio.Event | None = None
        async with self._orphan_lock:
            item = self._results.pop(task_id, None)
            if item is not None:
                self._result_events.pop(task_id, None)
                return item
            if task_id in self._orphan_task_ids:
                self._orphan_task_ids.remove(task_id)
                self._result_events.pop(task_id, None)
                return None
            event = self._result_events.get(task_id)
        if event is None:
            return None
        await event.wait()
        async with self._state_lock:
            infra_error = self._infra_error
        if infra_error is not None:
            raise RuntimeError("runner infrastructure failed") from infra_error
        async with self._orphan_lock:
            item = self._results.pop(task_id, None)
            self._result_events.pop(task_id, None)
            if task_id in self._orphan_task_ids:
                self._orphan_task_ids.remove(task_id)
            return item

    def _into_result(self, task_id: str, item: TContext | Exception) -> TContext:
        if isinstance(item, Exception):
            request_id = task_id.split("#", 1)[0]
            raise RuntimeError(  # noqa: TRY004
                f"runner worker failed for task_id={task_id} request_id={request_id}"
            ) from item
        return item


__all__ = ["RunnerBase", "StepBase"]
