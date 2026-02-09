"""Pydantic models used by SerpSage public APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PageChunk(BaseModel):
    """A single extracted chunk from a crawled page."""

    model_config = ConfigDict(validate_assignment=True)

    text: str
    score: float = 0.0


class PageEnrichment(BaseModel):
    """Page-level enrichment attached to a search result."""

    model_config = ConfigDict(validate_assignment=True)

    chunks: list[PageChunk] = Field(default_factory=list)
    error: str | None = None


class SearchContext(BaseModel):
    """Search context."""

    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: str
    number_of_results: int = 0
    results: list[SearchResult] = Field(default_factory=list)


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


__all__ = ["PageChunk", "PageEnrichment", "SearchResult", "SearchContext"]
