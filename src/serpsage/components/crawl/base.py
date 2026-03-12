from __future__ import annotations

from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic
from typing_extensions import TypeVar, override

import anyio
from pydantic import Field, field_validator

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.crawl import CrawlResult

_NESTED_INFLIGHT_BYPASS: ContextVar[bool] = ContextVar(
    "crawler_inflight_nested_bypass", default=False
)


class CrawlRetrySettings(ComponentConfigBase):
    max_attempts: int = 3
    delay_ms: int = 200


class CrawlRenderSettings(ComponentConfigBase):
    js_concurrency: int = 12
    nav_timeout_ms: int = 8_000
    block_resources: bool = True
    readiness_poll_ms: int = 150
    readiness_stable_rounds: int = 2
    post_ready_wait_ms: int = 120

    @field_validator("js_concurrency")
    @classmethod
    def _validate_js_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("js_concurrency must be >= 1")
        return value

    @field_validator(
        "readiness_poll_ms", "readiness_stable_rounds", "post_ready_wait_ms"
    )
    @classmethod
    def _validate_render_timing(cls, value: int) -> int:
        if value < 0:
            raise ValueError("render timing settings must be >= 0")
        return value


def _default_blocked_markers() -> list[str]:
    return [
        "cloudflare",
        "just a moment",
        "verify you are human",
        "access denied",
        "please enable javascript",
        "security check",
        "checking your browser",
    ]


class CrawlQualitySettings(ComponentConfigBase):
    min_text_chars: int = 100
    script_ratio_threshold: float = 0.35
    quality_score_threshold: float = 0.15
    blocked_markers: list[str] = Field(default_factory=_default_blocked_markers)


class CrawlAutoSettings(ComponentConfigBase):
    scout_bytes: int = 48_000
    route_memory_size: int = 4096
    direct_route_min_samples: int = 3
    direct_playwright_cost_ratio: float = 0.78
    direct_playwright_min_useful: float = 0.72
    learning_rate: float = 0.22


class CrawlConfigBase(ComponentConfigBase):
    inflight_enabled: bool = True
    inflight_timeout_s: float = 60.0
    timeout_s: float = 2.0
    follow_redirects: bool = True
    user_agent: str = "serpsage-bot/4.0"
    auto: CrawlAutoSettings = Field(default_factory=CrawlAutoSettings)
    render: CrawlRenderSettings = Field(default_factory=CrawlRenderSettings)
    quality: CrawlQualitySettings = Field(default_factory=CrawlQualitySettings)

    @field_validator("inflight_timeout_s")
    @classmethod
    def _validate_inflight_timeout_s(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError("inflight_timeout_s must be > 0")
        return float(value)


@dataclass(slots=True)
class _InFlightEntry:
    done: anyio.Event
    result: CrawlResult | None = None
    error: BaseException | None = None


CrawlerConfigT = TypeVar(
    "CrawlerConfigT",
    bound=CrawlConfigBase,
    default=CrawlConfigBase,
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
                return await self._acrawl_inner(url=url, timeout_s=timeout_s)
        finally:
            _NESTED_INFLIGHT_BYPASS.reset(token)

    async def _run_inner_once(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> CrawlResult:
        if timeout_s is None:
            return await self._acrawl_inner(url=url, timeout_s=timeout_s)
        with anyio.fail_after(float(timeout_s)):
            return await self._acrawl_inner(url=url, timeout_s=timeout_s)

    @abstractmethod
    async def _acrawl_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        raise NotImplementedError


__all__ = [
    "CrawlAutoSettings",
    "CrawlConfigBase",
    "CrawlerConfigT",
    "CrawlerBase",
    "CrawlQualitySettings",
    "CrawlRenderSettings",
    "CrawlRetrySettings",
]
