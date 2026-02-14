from __future__ import annotations

import time

import anyio

from serpsage import Engine
from serpsage.app.request import FetchRequest
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase, FetcherBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.models.fetch import FetchResult
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry

_URL = "https://example.com/mode"
_HTML = (
    "<html><head><title>Mode</title></head><body>"
    "<main><article><h1>Mode</h1>"
    "<p>This mode page contains enough repeated textual content for extraction. "
    "This mode page contains enough repeated textual content for extraction. "
    "This mode page contains enough repeated textual content for extraction.</p>"
    "</article></main>"
    "</body></html>"
).encode("utf-8")


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _MemoryCache(CacheBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        store: dict[tuple[str, str], bytes],
    ) -> None:
        super().__init__(rt=rt)
        self._store = store

    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return self._store.get((namespace, key))

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = ttl_s
        self._store[(namespace, key)] = bytes(value)


class _SwitchFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime, state: _FetchState) -> None:
        super().__init__(rt=rt)
        self._state = state

    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        rank_index: int = 0,
    ) -> FetchResult:
        _ = timeout_s, allow_render, rank_index
        self._state.calls += 1
        if self._state.fail:
            raise RuntimeError("crawler down")
        return FetchResult(
            url=url,
            status_code=200,
            content_type="text/html; charset=utf-8",
            content=_HTML,
            fetch_mode="httpx",
            rendered=False,
            content_kind="html",
            headers={},
            attempt_chain=["httpx"],
            quality_score=1.0,
        )


def _build_settings() -> AppSettings:
    settings = AppSettings()
    settings.fetch.render.enabled = False
    settings.fetch.timeout_s = 2.0
    return settings


def _build_runtime(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


class _FetchState:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False


async def _fetch_with(
    settings: AppSettings,
    cache_store: dict[tuple[str, str], bytes],
    fetch_state: _FetchState,
    req: FetchRequest,
) -> object:
    rt = _build_runtime(settings)
    cache = _MemoryCache(rt=rt, store=cache_store)
    fetcher = _SwitchFetcher(rt=rt, state=fetch_state)
    overrides = Overrides(
        cache=cache,
        fetcher=fetcher,
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        return await engine.fetch(req)


def test_crawl_mode_never_cache_hit_and_miss() -> None:
    settings = _build_settings()
    cache_store: dict[tuple[str, str], bytes] = {}
    fetch_state = _FetchState()

    warm_req = FetchRequest(urls=[_URL], content=True, crawl_mode="fallback")
    warm_resp = anyio.run(_fetch_with, settings, cache_store, fetch_state, warm_req)
    assert len(warm_resp.results) == 1
    assert fetch_state.calls == 1

    fetch_state.fail = True
    cached_req = FetchRequest(urls=[_URL], content=True, crawl_mode="never")
    cached_resp = anyio.run(
        _fetch_with,
        settings,
        cache_store,
        fetch_state,
        cached_req,
    )
    assert len(cached_resp.results) == 1
    assert fetch_state.calls == 1

    miss_req = FetchRequest(
        urls=["https://example.com/miss"],
        content=True,
        crawl_mode="never",
    )
    miss_resp = anyio.run(_fetch_with, settings, cache_store, fetch_state, miss_req)
    assert miss_resp.results == []
    assert miss_resp.errors
    assert miss_resp.errors[0].code == "fetch_cache_miss"


def test_crawl_mode_preferred_and_always() -> None:
    settings = _build_settings()
    cache_store: dict[tuple[str, str], bytes] = {}
    fetch_state = _FetchState()

    warm_req = FetchRequest(urls=[_URL], content=True, crawl_mode="fallback")
    warm_resp = anyio.run(_fetch_with, settings, cache_store, fetch_state, warm_req)
    assert len(warm_resp.results) == 1
    assert fetch_state.calls == 1

    fetch_state.fail = True
    preferred_req = FetchRequest(urls=[_URL], content=True, crawl_mode="preferred")
    preferred_resp = anyio.run(
        _fetch_with,
        settings,
        cache_store,
        fetch_state,
        preferred_req,
    )
    assert len(preferred_resp.results) == 1
    assert fetch_state.calls == 2

    always_req = FetchRequest(urls=[_URL], content=True, crawl_mode="always")
    always_resp = anyio.run(
        _fetch_with,
        settings,
        cache_store,
        fetch_state,
        always_req,
    )
    assert always_resp.results == []
    assert always_resp.errors
    assert always_resp.errors[0].code == "fetch_crawl_failed"
