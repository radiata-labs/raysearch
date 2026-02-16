from __future__ import annotations

import httpx
from pydantic import ConfigDict

from serpsage.components.cache import CacheBase
from serpsage.components.extract import ExtractorBase
from serpsage.components.fetch import FetcherBase
from serpsage.components.overview import LLMClientBase
from serpsage.components.provider import SearchProviderBase
from serpsage.components.rank import RankerBase
from serpsage.components.rate_limit.basic import BasicRateLimiter
from serpsage.core.model_base import FrozenModel, MutableModel
from serpsage.settings.models import AppSettings
from serpsage.telemetry.base import ClockBase, TelemetryBase


class Runtime(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    settings: AppSettings
    telemetry: TelemetryBase
    clock: ClockBase


class Overrides(MutableModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    http: httpx.AsyncClient | None = None
    telemetry: TelemetryBase | None = None
    clock: ClockBase | None = None
    cache: CacheBase | None = None
    rate_limiter: BasicRateLimiter | None = None
    provider: SearchProviderBase | None = None
    fetcher: FetcherBase | None = None
    extractor: ExtractorBase | None = None
    ranker: RankerBase | None = None
    llm: LLMClientBase | None = None


__all__ = ["Overrides", "Runtime"]
