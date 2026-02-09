from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from serpsage.contracts.errors import AppError


class PageChunk(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    chunk_id: str | None = None
    text: str
    score: float = 0.0


class PageEnrichment(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    chunks: list[PageChunk] = Field(default_factory=list)
    error: str | None = None


class ResultItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    source_id: str | None = None
    url: str = ""
    title: str = ""
    snippet: str = ""
    domain: str = ""
    published_date: str = ""
    engine: str = ""
    score: float = 0.0
    hit_keywords: list[str] = Field(default_factory=list)
    page: PageEnrichment = Field(default_factory=PageEnrichment)
    raw: dict[str, Any] | None = None


class Citation(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    cite_id: str
    source_id: str
    url: str
    title: str | None = None
    chunk_id: str | None = None
    quote: str | None = None


class OverviewResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class SearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: str
    results: list[ResultItem] = Field(default_factory=list)
    overview: OverviewResult | None = None
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] | None = None


__all__ = [
    "Citation",
    "OverviewResult",
    "PageChunk",
    "PageEnrichment",
    "ResultItem",
    "SearchResponse",
]

