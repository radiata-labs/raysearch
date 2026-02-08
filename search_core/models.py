from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PageChunk(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    text: str
    score: float = 0.0


class PageEnrichment(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    chunks: list[PageChunk] = Field(default_factory=list)
    error: str | None = None


class SearchResult(BaseModel):
    """Normalized search result."""

    model_config = ConfigDict(validate_assignment=True)

    url: str
    title: str
    snippet: str
    domain: str
    published_date: str
    engine: str
    raw: dict[str, Any]
    score: float = 0.0
    hit_keywords: list[str] = Field(default_factory=list)
    page: PageEnrichment = Field(default_factory=PageEnrichment)


class SearchContext(BaseModel):
    """Search context output."""

    model_config = ConfigDict(validate_assignment=True)

    query: str
    results: list[SearchResult]
    json_data: dict[str, Any]
    markdown: str


__all__ = ["PageChunk", "PageEnrichment", "SearchResult", "SearchContext"]
