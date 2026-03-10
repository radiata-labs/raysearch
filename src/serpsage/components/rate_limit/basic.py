from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.components.base import ComponentConfigBase, ComponentMeta
from serpsage.components.rate_limit.base import RateLimiterBase


class RateLimiterConfig(ComponentConfigBase):
    global_limit: int = 24
    per_host: int = 4
    politeness_delay_ms: int = 0


_BASIC_RATE_LIMITER_META = ComponentMeta(
    family="rate_limit",
    name="basic",
    version="1.0.0",
    summary="Basic semaphore-based host-aware rate limiter.",
    provides=("rate_limit.control",),
    config_model=RateLimiterConfig,
    config_optional=True,
)


class BasicRateLimiter(RateLimiterBase[RateLimiterConfig]):
    meta = _BASIC_RATE_LIMITER_META

    def __init__(self) -> None:
        self._global = anyio.Semaphore(max(1, int(self.config.global_limit)))
        self._per_host_limit = max(1, int(self.config.per_host))
        self._politeness_delay_ms = max(0, int(self.config.politeness_delay_ms))
        self._host_sems: dict[str, anyio.Semaphore] = {}
        self._host_lock = anyio.Lock()
        self._last_host_ms: dict[str, int] = {}

    @override
    async def acquire(self, *, host: str) -> None:
        await self._global.acquire()
        sem = await self._get_host_sem(host)
        await sem.acquire()
        if self._politeness_delay_ms > 0 and host:
            async with self._host_lock:
                now = int(self.clock.now_ms())
                last = int(self._last_host_ms.get(host, 0))
                wait_ms = int(self._politeness_delay_ms - (now - last))
                if wait_ms > 0:
                    await anyio.sleep(wait_ms / 1000.0)
                self._last_host_ms[host] = int(self.clock.now_ms())

    @override
    async def release(self, *, host: str) -> None:
        sem = await self._get_host_sem(host)
        sem.release()
        self._global.release()

    async def _get_host_sem(self, host: str) -> anyio.Semaphore:
        if not host:
            host = "<unknown>"
        async with self._host_lock:
            sem = self._host_sems.get(host)
            if sem is None:
                sem = anyio.Semaphore(self._per_host_limit)
                self._host_sems[host] = sem
            return sem


__all__ = ["BasicRateLimiter"]
