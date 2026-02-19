from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import anyio
import pytest
from pydantic import Field

from serpsage.core.runtime import Runtime
from serpsage.models.pipeline import BaseStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.telemetry.base import ClockBase, SpanBase
from serpsage.telemetry.trace import NoopTelemetry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_runtime(
    *,
    search_limit: int = 8,
    fetch_limit: int = 24,
    child_fetch_limit: int = 24,
    queue_size: int = 256,
) -> Runtime:
    settings = AppSettings()
    settings.runner.search_limit = search_limit
    settings.runner.fetch_limit = fetch_limit
    settings.runner.child_fetch_limit = child_fetch_limit
    settings.runner.queue_size = queue_size
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_TestClock())


class _TestContext(BaseStepContext):
    value: int = 0
    sleep_s: float = 0.0
    errors: list[object] = Field(default_factory=list)


class _AddStep(StepBase[_TestContext]):
    async def run_inner(
        self, ctx: _TestContext, *, span: SpanBase
    ) -> _TestContext:
        _ = span
        if float(ctx.sleep_s) > 0:
            await anyio.sleep(float(ctx.sleep_s))
        ctx.value += 1
        return ctx


@dataclass(slots=True)
class _Tracker:
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    current: int = 0
    peak: int = 0


class _TrackedSleepStep(StepBase[_TestContext]):
    def __init__(self, *, rt: Runtime, tracker: _Tracker, delay_s: float = 0.03) -> None:
        super().__init__(rt=rt)
        self._tracker = tracker
        self._delay_s = float(delay_s)

    async def run_inner(
        self, ctx: _TestContext, *, span: SpanBase
    ) -> _TestContext:
        _ = span
        async with self._tracker.lock:
            self._tracker.current += 1
            self._tracker.peak = max(self._tracker.peak, self._tracker.current)
        try:
            delay_s = float(ctx.sleep_s) if float(ctx.sleep_s) > 0 else self._delay_s
            await anyio.sleep(delay_s)
        finally:
            async with self._tracker.lock:
                self._tracker.current -= 1
        ctx.value += 1
        return ctx


class _GateStep(StepBase[_TestContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        entered: anyio.Event,
        release: anyio.Event,
    ) -> None:
        super().__init__(rt=rt)
        self._entered = entered
        self._release = release

    async def run_inner(
        self, ctx: _TestContext, *, span: SpanBase
    ) -> _TestContext:
        _ = span
        self._entered.set()
        await self._release.wait()
        ctx.value += 1
        return ctx


@pytest.mark.anyio
async def test_run_requires_initialized_runner() -> None:
    rt = _build_runtime()
    step = _AddStep(rt=rt)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")

    with pytest.raises(RuntimeError, match="not running"):
        await runner.run(_TestContext(request_id="req-1", value=1))


@pytest.mark.anyio
async def test_run_batch_requires_initialized_runner() -> None:
    rt = _build_runtime()
    step = _AddStep(rt=rt)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")

    with pytest.raises(RuntimeError, match="not running"):
        await runner.run_batch([_TestContext(request_id="req-1", value=1)])


@pytest.mark.anyio
async def test_run_single_task_uses_queue_path() -> None:
    rt = _build_runtime()
    step = _AddStep(rt=rt)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")

    await runner.ainit()
    try:
        out = await runner.run(_TestContext(request_id="single", value=41))
    finally:
        await runner.aclose()
    assert out.value == 42


@pytest.mark.anyio
async def test_run_batch_same_request_id_and_stable_order() -> None:
    rt = _build_runtime(fetch_limit=4)
    step = _AddStep(rt=rt)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")
    contexts = [
        _TestContext(
            request_id="same-request",
            value=index,
            sleep_s=float(0.06 - index * 0.005),
        )
        for index in range(8)
    ]

    await runner.ainit()
    try:
        out = await runner.run_batch(contexts)
    finally:
        await runner.aclose()

    assert [item.value for item in out] == [index + 1 for index in range(8)]


@pytest.mark.anyio
async def test_run_batch_respects_kind_concurrency_limit() -> None:
    rt = _build_runtime(fetch_limit=2)
    tracker = _Tracker()
    step = _TrackedSleepStep(rt=rt, tracker=tracker, delay_s=0.04)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")
    contexts = [_TestContext(request_id="conc", value=index) for index in range(10)]

    await runner.ainit()
    try:
        out = await runner.run_batch(contexts)
    finally:
        await runner.aclose()

    assert [item.value for item in out] == [index + 1 for index in range(10)]
    assert tracker.peak <= 2


@pytest.mark.anyio
async def test_run_batch_small_queue_size_backpressure_no_drop() -> None:
    rt = _build_runtime(fetch_limit=1, queue_size=1)
    tracker = _Tracker()
    step = _TrackedSleepStep(rt=rt, tracker=tracker, delay_s=0.03)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")
    contexts = [_TestContext(request_id="q", value=index) for index in range(5)]

    await runner.ainit()
    try:
        out = await runner.run_batch(contexts)
    finally:
        await runner.aclose()

    assert [item.value for item in out] == [index + 1 for index in range(5)]
    assert tracker.peak <= 1


@pytest.mark.anyio
async def test_on_close_drains_inflight_and_rejects_new_tasks() -> None:
    rt = _build_runtime(fetch_limit=1)
    entered = anyio.Event()
    release = anyio.Event()
    step = _GateStep(rt=rt, entered=entered, release=release)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")
    result: dict[str, _TestContext] = {}

    await runner.ainit()

    async def run_one() -> None:
        result["ctx"] = await runner.run(_TestContext(request_id="drain", value=1))

    async def delayed_release() -> None:
        await anyio.sleep(0.02)
        release.set()

    run_task = asyncio.create_task(run_one())
    release_task = asyncio.create_task(delayed_release())
    await entered.wait()
    await runner.aclose()
    await release_task
    await run_task

    assert result["ctx"].value == 2
    with pytest.raises(RuntimeError, match="not running"):
        await runner.run(_TestContext(request_id="late", value=1))


@pytest.mark.anyio
async def test_cancellation_does_not_leak_results() -> None:
    rt = _build_runtime(fetch_limit=1)
    entered = anyio.Event()
    release = anyio.Event()
    step = _GateStep(rt=rt, entered=entered, release=release)
    runner = RunnerBase[_TestContext](rt=rt, steps=[step], kind="fetch")

    await runner.ainit()
    cancelled = False

    async def run_with_timeout() -> None:
        nonlocal cancelled
        with anyio.move_on_after(0.05) as scope:
            await runner.run(_TestContext(request_id="cancel", value=1))
        cancelled = bool(scope.cancel_called)

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_with_timeout)
        await entered.wait()
        await anyio.sleep(0.08)
        release.set()

    await anyio.sleep(0.05)
    async with runner._result_lock:
        assert not runner._results
    async with runner._orphan_lock:
        assert not runner._orphan_task_ids
    assert cancelled
    await runner.aclose()
