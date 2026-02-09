from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable


class Span(Protocol):
    def add_event(self, name: str, **fields: Any) -> None: ...
    def set_attr(self, name: str, value: Any) -> None: ...
    def end(self) -> None: ...


class Telemetry(Protocol):
    def start_span(self, name: str, **attrs: Any) -> Span: ...


class Clock(Protocol):
    def now_ms(self) -> int: ...


@runtime_checkable
class Cache(Protocol):
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
    ) -> dict[str, Any]: ...


def stable_json(obj: Any) -> str:
    """Deterministic JSON representation used for cache keys."""
    import json

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def uniq_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


__all__ = [
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
    "stable_json",
    "uniq_preserve_order",
]

