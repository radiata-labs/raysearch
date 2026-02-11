from __future__ import annotations

import httpx
import pytest

from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase
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


class MemCache(CacheBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self.keys: list[str] = []

    async def aget(self, *, namespace: str, key: str) -> bytes | None:  # noqa: ARG002
        return None

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:  # noqa: ARG002
        self.keys.append(key)


@pytest.mark.anyio
async def test_cache_key_varies_with_accept_language_and_strategy():
    html = b"<html><body><p>" + (b"x" * 2000) + b"</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, headers={"content-type": "text/html"}, content=html)

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        base = {
            "cache": {"enabled": True, "fetch_ttl_s": 600},
            "enrich": {
                "enabled": True,
                "fetch": {
                    "validate_extractable": True,
                    "min_blocks": 1,
                    "min_text_chars": 10,
                    "timeout_s": 1.0,
                },
            },
            "overview": {"enabled": False},
        }

        s1 = AppSettings.model_validate(
            {
                **base,
                "enrich": {
                    **base["enrich"],
                    "fetch": {
                        **base["enrich"]["fetch"],
                        "strategy": "auto",
                        "accept_language": "zh-CN",
                    },
                },
            }
        )
        s2 = AppSettings.model_validate(
            {
                **base,
                "enrich": {
                    **base["enrich"],
                    "fetch": {
                        **base["enrich"]["fetch"],
                        "strategy": "httpx",
                        "accept_language": "en-US",
                    },
                },
            }
        )

        rt1 = Runtime(settings=s1, telemetry=NoopTelemetry(), clock=FakeClock())
        rt2 = Runtime(settings=s2, telemetry=NoopTelemetry(), clock=FakeClock())

        cache1 = MemCache(rt=rt1)
        cache2 = MemCache(rt=rt2)

        af1 = AutoFetcher(
            rt=rt1,
            cache=cache1,
            rate_limiter=RateLimiter(rt=rt1),
            httpx_fetcher=HttpxFetcher(
                rt=rt1,
                http=HttpClientUnit(rt=rt1, client=client, owns_client=False),
            ),
            curl_fetcher=None,
            extractor=BasicHtmlExtractor(rt=rt1),
        )
        af2 = AutoFetcher(
            rt=rt2,
            cache=cache2,
            rate_limiter=RateLimiter(rt=rt2),
            httpx_fetcher=HttpxFetcher(
                rt=rt2,
                http=HttpClientUnit(rt=rt2, client=client, owns_client=False),
            ),
            curl_fetcher=None,
            extractor=BasicHtmlExtractor(rt=rt2),
        )

        await af1.afetch(url="https://example.com/x")
        await af2.afetch(url="https://example.com/x")

    assert cache1.keys and cache2.keys
    assert cache1.keys[0] != cache2.keys[0]
