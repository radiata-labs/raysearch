from __future__ import annotations

from typing import Any

import httpx
from pydantic import ConfigDict

from serpsage.components.cache.base import CacheBase
from serpsage.components.crawl import CrawlerBase
from serpsage.components.extract import ExtractorBase
from serpsage.components.llm import LLMClientBase
from serpsage.components.provider import SearchProviderBase
from serpsage.components.rank import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.core.workunit import ClockBase
from serpsage.models.base import MutableModel


class Overrides(MutableModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    http: httpx.AsyncClient | None = None
    clock: ClockBase | None = None
    cache: CacheBase | None = None
    rate_limiter: RateLimiterBase[Any] | None = None
    provider: SearchProviderBase | None = None
    crawler: CrawlerBase | None = None
    extractor: ExtractorBase | None = None
    ranker: RankerBase | None = None
    llm: LLMClientBase | None = None
    telemetry: TelemetryEmitterBase[Any] | None = None


__all__ = ["Overrides"]
