from __future__ import annotations

import httpx
from pydantic import ConfigDict

from serpsage.components.fetch.rate_limit import RateLimiter
from serpsage.contracts.lifecycle import ClockBase, TelemetryBase
from serpsage.contracts.services import (
    CacheBase,
    ExtractorBase,
    FetcherBase,
    LLMClientBase,
    RankerBase,
    SearchProviderBase,
)
from serpsage.core.model_base import FrozenModel, MutableModel
from serpsage.settings.models import AppSettings


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
    rate_limiter: RateLimiter | None = None
    provider: SearchProviderBase | None = None
    fetcher: FetcherBase | None = None
    extractor: ExtractorBase | None = None
    ranker: RankerBase | None = None
    llm: LLMClientBase | None = None


__all__ = ["Overrides", "Runtime"]
