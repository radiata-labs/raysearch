from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic
from typing_extensions import TypeVar, override

import anyio
from pydantic import field_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.crawl import CrawlResult

_NESTED_INFLIGHT_BYPASS: ContextVar[bool] = ContextVar(
    "crawler_inflight_nested_bypass", default=False
)


class CrawlerConfigBase(ComponentConfigBase):
    timeout_s: float = 2.0
    inflight_enabled: bool = True
    inflight_timeout_s: float = 60.0

    @field_validator("timeout_s", "inflight_timeout_s")
    @classmethod
    def _validate_positive_timeout(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError("crawler timeouts must be > 0")
        return float(value)


@dataclass(slots=True)
class _InFlightEntry:
    done: anyio.Event
    result: CrawlResult | None = None
    error: BaseException | None = None


CrawlerConfigT = TypeVar(
    "CrawlerConfigT",
    bound=CrawlerConfigBase,
    default=CrawlerConfigBase,
)


class CrawlerBase(ComponentBase[CrawlerConfigT], ABC, Generic[CrawlerConfigT]):
    def __init__(self) -> None:
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
                entry.error = RuntimeError("crawler closed while waiting for result")
            entry.done.set()
        tg = self._inflight_tg
        tg_cm = self._inflight_tg_cm
        self._inflight_tg = None
        self._inflight_tg_cm = None
        if tg is not None:
            tg.cancel_scope.cancel()
        if tg_cm is not None:
            await tg_cm.__aexit__(None, None, None)

    async def acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        if not bool(self.config.inflight_enabled) or _NESTED_INFLIGHT_BYPASS.get():
            return await self._run_inner_once(url=url, timeout_s=timeout_s)
        entry, created = await self._get_inflight_entry(url=url)
        if created:
            tg = self._inflight_tg
            if tg is None:
                raise RuntimeError("crawler task group is not initialized")
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
    ) -> CrawlResult:
        if timeout_s is None:
            await entry.done.wait()
        else:
            with anyio.fail_after(float(timeout_s)):
                await entry.done.wait()
        if entry.result is not None:
            return entry.result
        if entry.error is None:
            raise RuntimeError("in-flight crawl completed without result or error")
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

    async def _run_inflight_with_retry(self, *, url: str) -> CrawlResult:
        try:
            return await self._run_inflight_once(url=url)
        except TimeoutError:
            raise
        except BaseException:  # noqa: BLE001
            return await self._run_inflight_once(url=url)

    async def _run_inflight_once(self, *, url: str) -> CrawlResult:
        timeout_s = float(self.config.inflight_timeout_s)
        token = _NESTED_INFLIGHT_BYPASS.set(True)
        try:
            with anyio.fail_after(timeout_s):
                return await self._acrawl(url=url, timeout_s=timeout_s)
        finally:
            _NESTED_INFLIGHT_BYPASS.reset(token)

    async def _run_inner_once(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> CrawlResult:
        if timeout_s is None:
            return await self._acrawl(url=url, timeout_s=timeout_s)
        with anyio.fail_after(float(timeout_s)):
            return await self._acrawl(url=url, timeout_s=timeout_s)

    @abstractmethod
    async def _acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        raise NotImplementedError


class SpecializedCrawlerBase(CrawlerBase[CrawlerConfigT], ABC, Generic[CrawlerConfigT]):
    """Base class for specialized crawlers that handle specific URL patterns.

    Specialized crawlers implement `can_handle` to declare what URLs they can process.
    The AutoCrawler will try each specialized crawler in order and use the
    first one that returns True. Falls back to curl_cffi/playwright for unmatched URLs.

    This is analogous to SpecializedExtractorBase in the extract module.
    """

    @classmethod
    @abstractmethod
    def can_handle(cls, *, url: str) -> bool:
        """Return True if this crawler should handle the given URL.

        Args:
            url: The URL to check

        Returns:
            True if this crawler should handle the URL
        """
        raise NotImplementedError


__all__ = [
    "CrawlerBase",
    "CrawlerConfigBase",
    "CrawlerConfigT",
    "SpecializedCrawlerBase",
]
