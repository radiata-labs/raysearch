from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx
from pydantic import ConfigDict, Field

from serpsage.components.cache.base import CacheBase
from serpsage.components.extract import ExtractorBase
from serpsage.components.fetch import FetcherBase
from serpsage.components.llm import LLMClientBase
from serpsage.components.provider import SearchProviderBase
from serpsage.components.rank import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.models.base import FrozenModel, MutableModel
from serpsage.settings.models import AppSettings

if TYPE_CHECKING:
    from serpsage.components.container import ComponentContainer


class ClockBase(ABC):
    @abstractmethod
    def now_ms(self) -> int:
        raise NotImplementedError


class Runtime(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )
    settings: AppSettings
    clock: ClockBase
    telemetry: TelemetryEmitterBase | None = None
    components: ComponentContainer | None = None
    env: dict[str, str] = Field(default_factory=dict)


class Overrides(MutableModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    http: httpx.AsyncClient | None = None
    clock: ClockBase | None = None
    cache: CacheBase | None = None
    rate_limiter: RateLimiterBase | None = None
    provider: SearchProviderBase | None = None
    fetcher: FetcherBase | None = None
    extractor: ExtractorBase | None = None
    ranker: RankerBase | None = None
    llm: LLMClientBase | None = None
    telemetry: TelemetryEmitterBase | None = None


__all__ = ["ClockBase", "Overrides", "Runtime"]
