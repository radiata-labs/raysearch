from __future__ import annotations

import pytest

from serpsage.components.cache.memory import MemoryCache
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def __init__(self) -> None:
        self._ms = 1_000_000

    def now_ms(self) -> int:
        return self._ms

    def advance(self, ms: int) -> None:
        self._ms += int(ms)


@pytest.mark.anyio
async def test_memory_cache_ttl() -> None:
    clock = FakeClock()
    settings = AppSettings.model_validate({"cache": {"enabled": True}})
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=clock)
    async with MemoryCache(rt=rt) as cache:
        await cache.aset(namespace="x", key="k", value=b"v", ttl_s=1)
        assert await cache.aget(namespace="x", key="k") == b"v"
        clock.advance(1500)
        assert await cache.aget(namespace="x", key="k") is None


@pytest.mark.anyio
async def test_memory_cache_ignores_non_positive_ttl() -> None:
    settings = AppSettings.model_validate({"cache": {"enabled": True}})
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    async with MemoryCache(rt=rt) as cache:
        await cache.aset(namespace="x", key="k0", value=b"v", ttl_s=0)
        await cache.aset(namespace="x", key="k1", value=b"v", ttl_s=-1)
        assert await cache.aget(namespace="x", key="k0") is None
        assert await cache.aget(namespace="x", key="k1") is None
