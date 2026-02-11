from __future__ import annotations

import httpx
import pytest

from serpsage.cache.null import NullCache
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.fetch.auto import AutoFetcher
from serpsage.fetch.http import HttpxFetcher
from serpsage.fetch.http_client_unit import HttpClientUnit
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


@pytest.mark.anyio
async def test_sniff_allows_mislabeled_html():
    html = b"<html><body><p>Hello " + (b"x" * 800) + b"</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=html
        )

    settings = AppSettings.model_validate(
        {
            "cache": {"enabled": False},
            "enrich": {
                "enabled": True,
                "fetch": {
                    "strategy": "httpx",
                    "validate_extractable": True,
                    "min_blocks": 1,
                    "min_text_chars": 20,
                    "timeout_s": 1.0,
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        af = AutoFetcher(
            rt=rt,
            cache=NullCache(rt=rt),
            rate_limiter=RateLimiter(rt=rt),
            httpx_fetcher=HttpxFetcher(
                rt=rt,
                http=HttpClientUnit(rt=rt, client=client, owns_client=False),
            ),
            curl_fetcher=None,
            extractor=BasicHtmlExtractor(rt=rt),
        )
        res = await af.afetch(url="https://example.com/x")

    assert b"<html" in res.content.lower()


@pytest.mark.anyio
async def test_sniff_rejects_non_html_and_raises_unusable():
    body = b"\x89PNG\r\n\x1a\n" + (b"x" * 5000)

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, headers={"content-type": "image/png"}, content=body)

    settings = AppSettings.model_validate(
        {
            "cache": {"enabled": False},
            "enrich": {
                "enabled": True,
                "fetch": {
                    "strategy": "httpx",
                    "validate_extractable": True,
                    "min_blocks": 1,
                    "min_text_chars": 10,
                    "timeout_s": 1.0,
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        af = AutoFetcher(
            rt=rt,
            cache=NullCache(rt=rt),
            rate_limiter=RateLimiter(rt=rt),
            httpx_fetcher=HttpxFetcher(
                rt=rt,
                http=HttpClientUnit(rt=rt, client=client, owns_client=False),
            ),
            curl_fetcher=None,
            extractor=BasicHtmlExtractor(rt=rt),
        )
        with pytest.raises(RuntimeError):
            await af.afetch(url="https://example.com/x")
