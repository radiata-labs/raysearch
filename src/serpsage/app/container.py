from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from serpsage.cache.sqlite import NullCache, SqliteCache
from serpsage.contracts.protocols import (
    Cache,
    Clock,
    Extractor,
    Fetcher,
    LLMClient,
    Ranker,
    SearchProvider,
    Telemetry,
)
from serpsage.extract.html_basic import BasicHtmlExtractor
from serpsage.fetch.http import HttpFetcher
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.overview.openai_compat import NullLLMClient, OpenAICompatLLMClient
from serpsage.provider.searxng import SearxngProvider
from serpsage.rank.blend import BlendRanker
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry, TraceTelemetry


class SystemClock:
    def now_ms(self) -> int:
        import time

        return int(time.time() * 1000)


@dataclass
class Overrides:
    provider: SearchProvider | None = None
    fetcher: Fetcher | None = None
    extractor: Extractor | None = None
    ranker: Ranker | None = None
    cache: Cache | None = None
    llm: LLMClient | None = None
    telemetry: Telemetry | None = None
    clock: Clock | None = None


class Container:
    """Dependency injection container (async-only)."""

    def __init__(self, *, settings: AppSettings, overrides: Overrides | None = None) -> None:
        self.settings = settings
        self._overrides = overrides or Overrides()

        self.clock: Clock = self._overrides.clock or SystemClock()
        self.telemetry: Telemetry = (
            self._overrides.telemetry
            or (TraceTelemetry(settings.telemetry) if settings.telemetry.enabled else NoopTelemetry())
        )

        self._http: httpx.AsyncClient | None = None
        self._owns_http = True

        self.cache: Cache = self._overrides.cache or (
            SqliteCache(settings=settings, telemetry=self.telemetry, clock=self.clock)
            if settings.cache.enabled
            else NullCache(settings=settings, telemetry=self.telemetry, clock=self.clock)
        )

        self._rate_limiter = RateLimiter(
            settings=settings,
            telemetry=self.telemetry,
            clock=self.clock,
        )

        self.provider: SearchProvider = self._overrides.provider or SearxngProvider(
            settings=settings,
            telemetry=self.telemetry,
            clock=self.clock,
            http=self._get_http(),
            cache=self.cache,
        )

        self.fetcher: Fetcher = self._overrides.fetcher or HttpFetcher(
            settings=settings,
            telemetry=self.telemetry,
            clock=self.clock,
            http=self._get_http(),
            cache=self.cache,
            rate_limiter=self._rate_limiter,
        )

        self.extractor: Extractor = self._overrides.extractor or BasicHtmlExtractor(
            settings=settings,
            telemetry=self.telemetry,
            clock=self.clock,
        )

        self.ranker: Ranker = self._overrides.ranker or BlendRanker(
            settings=settings,
            telemetry=self.telemetry,
            clock=self.clock,
        )

        self.llm: LLMClient = self._overrides.llm or (
            OpenAICompatLLMClient(
                settings=settings,
                telemetry=self.telemetry,
                clock=self.clock,
                http=self._get_http(),
            )
            if settings.overview.enabled and settings.overview.llm.api_key
            else NullLLMClient(settings=settings, telemetry=self.telemetry, clock=self.clock)
        )

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            # Keep client generic; pass headers/timeouts per-request.
            self._http = httpx.AsyncClient()
        return self._http

    async def aclose(self) -> None:
        # Close owned resources.
        span = self.telemetry.start_span("container.aclose")
        try:
            span.add_event("closing")
        finally:
            span.end()

        # Cache might have background resources (sqlite).
        if hasattr(self.cache, "aclose"):
            await getattr(self.cache, "aclose")()

        if self._http is not None and self._owns_http:
            await self._http.aclose()
            self._http = None


__all__ = ["Container", "Overrides"]
