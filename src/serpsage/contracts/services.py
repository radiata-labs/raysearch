from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from serpsage.models.extract import ExtractedText
    from serpsage.models.fetch import FetchResult
    from serpsage.models.llm import ChatJSONResult
    from serpsage.models.pipeline import SearchStepContext


class CacheBase(WorkUnit, ABC):
    @abstractmethod
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        raise NotImplementedError

    @abstractmethod
    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        raise NotImplementedError


class SearchProviderBase(WorkUnit, ABC):
    @abstractmethod
    async def asearch(
        self, *, query: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class FetcherBase(WorkUnit, ABC):
    @abstractmethod
    async def afetch(self, *, url: str) -> FetchResult:
        raise NotImplementedError


class RateLimiterBase(WorkUnit, ABC):
    @abstractmethod
    async def acquire(self, *, host: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def release(self, *, host: str) -> None:
        raise NotImplementedError


class ExtractorBase(WorkUnit, ABC):
    @abstractmethod
    def extract(
        self, *, url: str, content: bytes, content_type: str | None
    ) -> ExtractedText:
        raise NotImplementedError


class RankerBase(WorkUnit, ABC):
    @abstractmethod
    def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, *, scores: list[float]) -> list[float]:
        raise NotImplementedError


class LLMClientBase(WorkUnit, ABC):
    @abstractmethod
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatJSONResult:
        raise NotImplementedError


class PipelineStepBase(WorkUnit, ABC):
    @abstractmethod
    async def run(self, ctx: SearchStepContext) -> SearchStepContext:
        raise NotImplementedError


__all__ = [
    "CacheBase",
    "ExtractorBase",
    "FetcherBase",
    "LLMClientBase",
    "PipelineStepBase",
    "RankerBase",
    "SearchProviderBase",
]
