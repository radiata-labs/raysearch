from __future__ import annotations

import json

import httpx
import pytest

from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import CoreRuntime
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.fetch.auto import AutoFetcher
from serpsage.fetch.http import HttpxFetcher
from serpsage.models.fetch import FetchAttempt
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


class DummyRateLimiter:
    async def acquire(self, *, host: str) -> None:  # noqa: ARG002
        return

    async def release(self, *, host: str) -> None:  # noqa: ARG002
        return


class MemCache:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return self.store.get((namespace, key))

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:  # noqa: ARG002
        self.store[(namespace, key)] = value


class FakeCurl:
    def __init__(self) -> None:
        self.called = 0

    async def fetch_attempt(self, *, url: str, span):  # noqa: ANN001
        self.called += 1
        html = (
            "<html><body>"
            "<article><p>This is main content. " + ("hello " * 120) + "</p></article>"
            "</body></html>"
        ).encode("utf-8")
        return FetchAttempt(
            url=url,
            status_code=200,
            content_type="text/html; charset=utf-8",
            content=html,
            truncated=False,
            strategy_used="curl_cffi",
        )


@pytest.mark.anyio
async def test_auto_fetcher_switches_to_curl_on_challenge_page_and_caches_good_result():
    settings = AppSettings.model_validate(
        {
            "cache": {"enabled": True, "fetch_ttl_s": 600},
            "enrich": {
                "enabled": True,
                "fetch": {
                    "strategy": "auto",
                    "validate_extractable": True,
                    "min_blocks": 1,
                    "min_text_chars": 120,
                    "total_budget_s": 3.0,
                    "max_attempts_total": 3,
                    "max_attempts_per_strategy": 1,
                    "timeout_s": 1.0,
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())

    challenge = b"<html><title>Just a moment...</title><body>Just a moment...</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=challenge,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        httpx_fetcher = HttpxFetcher(rt=rt, http=client)
        curl = FakeCurl()
        cache = MemCache()
        extractor = BasicHtmlExtractor(rt=rt)
        af = AutoFetcher(
            rt=rt,
            cache=cache,
            rate_limiter=DummyRateLimiter(),
            httpx_fetcher=httpx_fetcher,
            curl_fetcher=curl,
            extractor=extractor,
        )

        res = await af.afetch(url="https://example.com/x")

    assert curl.called >= 1
    assert b"This is main content" in res.content

    # Verify cache stored the good (curl) response, not the challenge page.
    assert len(cache.store) == 1
    (_, _), blob = next(iter(cache.store.items()))
    payload = json.loads(blob.decode("utf-8"))
    assert payload["strategy_used"] == "curl_cffi"
    assert "This is main content" in bytes.fromhex(payload["content_hex"]).decode("utf-8")




