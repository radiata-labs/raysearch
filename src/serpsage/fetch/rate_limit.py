from __future__ import annotations

import math
from dataclasses import dataclass

import anyio

from serpsage.contracts.base import Component
from serpsage.contracts.protocols import Clock, Telemetry
from serpsage.settings.models import AppSettings


class RateLimiter(Component[None]):
    def __init__(self, *, settings: AppSettings, telemetry: Telemetry, clock: Clock) -> None:
        super().__init__(settings=settings, telemetry=telemetry, clock=clock)
        fetch = settings.enrich.fetch
        self._global = anyio.Semaphore(max(1, int(fetch.global_concurrency)))
        self._per_host_limit = max(1, int(fetch.per_host_concurrency))
        self._politeness_delay_ms = max(0, int(fetch.politeness_delay_ms))
        self._host_sems: dict[str, anyio.Semaphore] = {}
        self._host_lock = anyio.Lock()
        self._last_host_ms: dict[str, int] = {}

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

    async def release(self, *, host: str) -> None:
        sem = await self._get_host_sem(host)
        sem.release()
        self._global.release()

    async def _get_host_sem(self, host: str) -> anyio.Semaphore:
        if not host:
            # Put empty host under a shared semaphore.
            host = "<unknown>"
        async with self._host_lock:
            sem = self._host_sems.get(host)
            if sem is None:
                sem = anyio.Semaphore(self._per_host_limit)
                self._host_sems[host] = sem
            return sem


__all__ = ["RateLimiter"]

