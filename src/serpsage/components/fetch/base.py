from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.components.fetch.models import FetchResult
    from serpsage.core.runtime import Runtime
_NESTED_INFLIGHT_BYPASS: ContextVar[bool] = ContextVar(
    "fetcher_inflight_nested_bypass", default=False
)


@dataclass(slots=True)
class _InFlightEntry:
    done: anyio.Event
    result: FetchResult | None = None
    error: BaseException | None = None


class FetcherBase(WorkUnit, ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._wrap_lifecycle_hook("on_init", base_first=True)
        cls._wrap_lifecycle_hook("on_close", base_first=False)

    @classmethod
    def _wrap_lifecycle_hook(cls, name: str, *, base_first: bool) -> None:
        original = cls.__dict__.get(name)
        if original is None:
            return
        if bool(getattr(original, "__fetcherbase_wrapped__", False)):
            return
        if name == "on_init":

            @wraps(original)
            async def wrapped(self: FetcherBase, *args: Any, **kwargs: Any) -> None:
                if base_first:
                    await FetcherBase.on_init(self)
                    await original(self, *args, **kwargs)
                else:
                    await original(self, *args, **kwargs)
                    await FetcherBase.on_init(self)
        else:

            @wraps(original)
            async def wrapped(self: FetcherBase, *args: Any, **kwargs: Any) -> None:
                if base_first:
                    try:
                        await FetcherBase.on_close(self)
                    finally:
                        await original(self, *args, **kwargs)
                else:
                    try:
                        await original(self, *args, **kwargs)
                    finally:
                        await FetcherBase.on_close(self)

        # Cast to Any to allow setting custom attribute
        wrapped_with_attr = cast("Any", wrapped)
        wrapped_with_attr.__fetcherbase_wrapped__ = True
        setattr(cls, name, wrapped_with_attr)

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._inflight_lock = anyio.Lock()
        self._inflight_pool: dict[str, _InFlightEntry] = {}
        self._inflight_tg_cm: Any | None = None
        self._inflight_tg: Any | None = None

    @override
    async def on_init(self) -> None:
        if self._inflight_tg is not None:
            return
        tg_cm = anyio.create_task_group()
        tg = await tg_cm.__aenter__()
        self._inflight_tg_cm = tg_cm
        self._inflight_tg = tg

    @override
    async def on_close(self) -> None:
        async with self._inflight_lock:
            entries = list(self._inflight_pool.values())
            self._inflight_pool.clear()
        for entry in entries:
            if entry.result is None and entry.error is None:
                entry.error = RuntimeError("fetcher closed while waiting for result")
            entry.done.set()
        tg = self._inflight_tg
        tg_cm = self._inflight_tg_cm
        self._inflight_tg = None
        self._inflight_tg_cm = None
        if tg is not None:
            tg.cancel_scope.cancel()
        if tg_cm is not None:
            await tg_cm.__aexit__(None, None, None)

    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        if (
            not bool(self.settings.fetch.inflight_enabled)
            or _NESTED_INFLIGHT_BYPASS.get()
        ):
            return await self._run_inner_once(url=url, timeout_s=timeout_s)
        entry, created = await self._get_inflight_entry(url=url)
        if created:
            tg = self._inflight_tg
            if tg is None:
                raise RuntimeError("fetcher task group is not initialized")
            tg.start_soon(self._run_inflight_worker, url, entry)
        return await self._await_inflight_entry(entry=entry, timeout_s=timeout_s)

    async def _get_inflight_entry(self, *, url: str) -> tuple[_InFlightEntry, bool]:
        async with self._inflight_lock:
            entry = self._inflight_pool.get(url)
            if entry is not None:
                return entry, False
            created = _InFlightEntry(done=anyio.Event())
            self._inflight_pool[url] = created
            return created, True

    async def _await_inflight_entry(
        self,
        *,
        entry: _InFlightEntry,
        timeout_s: float | None,
    ) -> FetchResult:
        if timeout_s is None:
            await entry.done.wait()
        else:
            with anyio.fail_after(float(timeout_s)):
                await entry.done.wait()
        if entry.result is not None:
            return entry.result
        if entry.error is None:
            raise RuntimeError("in-flight fetch completed without result or error")
        raise entry.error

    async def _run_inflight_worker(self, url: str, entry: _InFlightEntry) -> None:
        try:
            try:
                entry.result = await self._run_inflight_with_retry(url=url)
            except BaseException as exc:  # noqa: BLE001
                if entry.result is None and entry.error is None:
                    entry.error = exc
            finally:
                entry.done.set()
        finally:
            async with self._inflight_lock:
                current = self._inflight_pool.get(url)
                if current is entry:
                    self._inflight_pool.pop(url, None)

    async def _run_inflight_with_retry(self, *, url: str) -> FetchResult:
        try:
            return await self._run_inflight_once(url=url)
        except TimeoutError:
            raise
        except BaseException:  # noqa: BLE001
            return await self._run_inflight_once(url=url)

    async def _run_inflight_once(self, *, url: str) -> FetchResult:
        timeout_s = float(self.settings.fetch.inflight_timeout_s)
        token = _NESTED_INFLIGHT_BYPASS.set(True)
        try:
            with anyio.fail_after(timeout_s):
                return await self._afetch_inner(url=url, timeout_s=timeout_s)
        finally:
            _NESTED_INFLIGHT_BYPASS.reset(token)

    async def _run_inner_once(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> FetchResult:
        if timeout_s is None:
            return await self._afetch_inner(url=url, timeout_s=timeout_s)
        with anyio.fail_after(float(timeout_s)):
            return await self._afetch_inner(url=url, timeout_s=timeout_s)

    @abstractmethod
    async def _afetch_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        raise NotImplementedError
