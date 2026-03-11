from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import ConfigDict, Field

from serpsage.components.cache.base import CacheBase
from serpsage.components.crawl import CrawlerBase
from serpsage.components.extract import ExtractorBase
from serpsage.components.llm import LLMClientBase
from serpsage.components.provider import SearchProviderBase
from serpsage.components.rank import RankerBase
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.components.telemetry import TelemetryEmitterBase
from serpsage.models.base import MutableModel
from serpsage.settings.models import AppSettings

if TYPE_CHECKING:
    from serpsage.components.loads import ComponentRegistry


class ClockBase(ABC):
    @abstractmethod
    def now_ms(self) -> int:
        raise NotImplementedError


class Runtime(MutableModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    settings: AppSettings
    clock: ClockBase
    telemetry: TelemetryEmitterBase[Any] | None = None
    components: ComponentRegistry | None = None
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
    rate_limiter: RateLimiterBase[Any] | None = None
    provider: SearchProviderBase | None = None
    crawler: CrawlerBase | None = None
    extractor: ExtractorBase | None = None
    ranker: RankerBase | None = None
    llm: LLMClientBase | None = None
    telemetry: TelemetryEmitterBase[Any] | None = None


def _rebuild_runtime_model() -> None:
    from serpsage.components.loads import ComponentRegistry

    Runtime.model_rebuild(
        _types_namespace={
            "ComponentRegistry": ComponentRegistry,
        }
    )


_rebuild_runtime_model()

__all__ = ["ClockBase", "Overrides", "Runtime"]
