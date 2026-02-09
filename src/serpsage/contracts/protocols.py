from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from serpsage.contracts.llm import ChatJSONResult


class AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


class Span(Protocol):
    def add_event(self, name: str, **fields: Any) -> None: ...
    def set_attr(self, name: str, value: Any) -> None: ...
    def end(self) -> None: ...


class Telemetry(Protocol):
    def start_span(self, name: str, **attrs: Any) -> Span: ...
    def summary(self) -> dict[str, Any]: ...


class Clock(Protocol):
    def now_ms(self) -> int: ...


@runtime_checkable
class Cache(AsyncCloseable, Protocol):
    async def aget(self, *, namespace: str, key: str) -> bytes | None: ...
    async def aset(
        self, *, namespace: str, key: str, value: bytes, ttl_s: int
    ) -> None: ...


class SearchProvider(Protocol):
    async def asearch(
        self, *, query: str, params: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]: ...


class FetchResult(Protocol):
    @property
    def url(self) -> str: ...

    @property
    def status_code(self) -> int: ...

    @property
    def content_type(self) -> str | None: ...

    @property
    def content(self) -> bytes: ...


class Fetcher(Protocol):
    async def afetch(self, *, url: str) -> FetchResult: ...


class ExtractedText(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def blocks(self) -> list[str]: ...


class Extractor(Protocol):
    def extract(
        self, *, url: str, content: bytes, content_type: str | None
    ) -> ExtractedText: ...


class ChunkDraft(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def position(self) -> int: ...


class Chunker(Protocol):
    def chunk(self, *, text: str) -> list[ChunkDraft]: ...


class Ranker(Protocol):
    def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[float]: ...

    def normalize(self, *, scores: list[float]) -> list[float]: ...


class LLMClient(Protocol):
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatJSONResult: ...


__all__ = [
    "AsyncCloseable",
    "Cache",
    "ChunkDraft",
    "Chunker",
    "Clock",
    "ExtractedText",
    "Extractor",
    "FetchResult",
    "Fetcher",
    "LLMClient",
    "Ranker",
    "SearchProvider",
    "Span",
    "Telemetry",
]
