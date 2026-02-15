from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

    from serpsage.models.extract import ExtractContentOptions, ExtractedDocument
    from serpsage.models.fetch import FetchResult
    from serpsage.models.llm import ChatResult
    from serpsage.models.pipeline import BaseStepContext

    TContext = TypeVar("TContext", bound=BaseStepContext)
else:
    TContext = TypeVar("TContext")


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
    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        rank_index: int = 0,
    ) -> FetchResult:
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
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        raise NotImplementedError


class RankerBase(WorkUnit, ABC):
    @abstractmethod
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        raise NotImplementedError


class LLMClientBase(WorkUnit, ABC):
    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError


class HttpClientBase(WorkUnit, ABC):
    @property
    @abstractmethod
    def client(self) -> httpx.AsyncClient:
        raise NotImplementedError


class PipelineStepBase(WorkUnit, ABC, Generic[TContext]):
    @abstractmethod
    async def run(self, ctx: TContext) -> TContext:
        raise NotImplementedError


class PipelineRunnerBase(WorkUnit, ABC, Generic[TContext]):
    @abstractmethod
    async def run(self, ctx: TContext) -> TContext:
        raise NotImplementedError


__all__ = [
    "CacheBase",
    "ExtractorBase",
    "FetcherBase",
    "HttpClientBase",
    "LLMClientBase",
    "PipelineRunnerBase",
    "PipelineStepBase",
    "RankerBase",
    "SearchProviderBase",
    "TContext",
]
