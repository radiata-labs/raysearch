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

_HTML_TMPL = (
    "<html><head><title>{title}</title></head><body>"
    "<main><article><h1>{title}</h1>"
    "<p>{body} {body} {body} {body}</p>"
    "</article></main>"
    "</body></html>"
)


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _NoopCache(CacheBase):
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        _ = namespace, key
        return None

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = namespace, key, value, ttl_s
        return


class _TrackingFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime, fail_urls: set[str] | None = None) -> None:
        super().__init__(rt=rt)
        self._fail_urls = set(fail_urls or set())
        self._lock = anyio.Lock()
        self.active = 0
        self.max_active = 0

    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        rank_index: int = 0,
    ) -> FetchResult:
        _ = timeout_s, allow_render, rank_index
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            await anyio.sleep(0.05)
            if url in self._fail_urls:
                raise RuntimeError("failed to fetch")
            body = (
                f"This page body for {url} contains sufficiently long extraction text "
                "for markdown conversion and downstream scoring."
            )
            html = _HTML_TMPL.format(title=url.rsplit("/", 1)[-1], body=body)
            return FetchResult(
                url=url,
                status_code=200,
                content_type="text/html; charset=utf-8",
                content=html.encode("utf-8"),
                fetch_mode="httpx",
                rendered=False,
                content_kind="html",
                headers={},
                attempt_chain=["httpx"],
                quality_score=0.8,
            )
        finally:
            async with self._lock:
                self.active -= 1


def _build_runtime(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


async def _run_fetch(
    settings: AppSettings,
    cache: CacheBase,
    fetcher: _TrackingFetcher,
    req: FetchRequest,
) -> object:
    overrides = Overrides(cache=cache, fetcher=fetcher)
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        return await engine.fetch(req)


def test_multi_url_results_only_success_and_stable_order() -> None:
    settings = AppSettings()
    settings.fetch.render.enabled = False
    settings.fetch.concurrency.global_limit = 4
    rt = _build_runtime(settings)
    cache = _NoopCache(rt=rt)

    u1 = "https://example.com/1"
    u2 = "https://example.com/2"
    u3 = "https://example.com/3"
    fetcher = _TrackingFetcher(rt=rt, fail_urls={u2})
    req = FetchRequest(urls=[u1, u2, u3], content=True)
    resp = anyio.run(_run_fetch, settings, cache, fetcher, req)

    assert [item.url for item in resp.results] == [u1, u3]
    assert resp.errors
    assert any(err.details.get("url") == u2 for err in resp.errors)


def test_multi_url_concurrency_limit_respected() -> None:
    settings = AppSettings()
    settings.fetch.render.enabled = False
    settings.fetch.concurrency.global_limit = 2
    rt = _build_runtime(settings)
    cache = _NoopCache(rt=rt)

    urls = [f"https://example.com/{i}" for i in range(6)]
    fetcher = _TrackingFetcher(rt=rt)
    req = FetchRequest(urls=urls, content=True)
    resp = anyio.run(_run_fetch, settings, cache, fetcher, req)

    assert len(resp.results) == len(urls)
    assert fetcher.max_active <= 2
    assert fetcher.max_active >= 2
