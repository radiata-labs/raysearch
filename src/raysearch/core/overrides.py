from __future__ import annotations

from typing import Any

import httpx
from pydantic import ConfigDict

from raysearch.components.cache.base import CacheBase
from raysearch.components.crawl import CrawlerBase
from raysearch.components.extract import ExtractorBase
from raysearch.components.llm import LLMClientBase
from raysearch.components.provider import SearchProviderBase
from raysearch.components.rank import RankerBase
from raysearch.components.rate_limit.base import RateLimiterBase
from raysearch.core.workunit import ClockBase
from raysearch.models.base import MutableModel
from raysearch.telemetry import MeteringEmitterBase, TrackingEmitterBase


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
    tracker: TrackingEmitterBase | None = None
    meter: MeteringEmitterBase | None = None


__all__ = ["Overrides"]
