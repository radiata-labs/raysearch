from __future__ import annotations

import pytest

from serpsage.app.runtime import CoreRuntime
from serpsage.cache.sqlite import SqliteCache
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock:
    def __init__(self):
        self._ms = 1_000_000

    def now_ms(self) -> int:
        return self._ms

    def advance(self, ms: int) -> None:
        self._ms += int(ms)


@pytest.mark.anyio
async def test_sqlite_cache_ttl(tmp_path):
    clock = FakeClock()
    settings = AppSettings.model_validate(
        {"cache": {"enabled": True, "db_path": str(tmp_path / "c.sqlite3")}}
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=clock)
    cache = SqliteCache(rt=rt)

    await cache.aset(namespace="x", key="k", value=b"v", ttl_s=1)
    assert await cache.aget(namespace="x", key="k") == b"v"

    clock.advance(1500)
    assert await cache.aget(namespace="x", key="k") is None
