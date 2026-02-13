from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from serpsage.models.errors import AppError  # noqa: TC001
from serpsage.models.llm import LLMUsage  # noqa: TC001


class PageChunk(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    chunk_id: str | None = None
    text: str
    score: float = 0.0


class PageEnrichment(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    chunks: list[PageChunk] = Field(default_factory=list)
    markdown: str = ""
    content_kind: str | None = None
    fetch_mode: str | None = None
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


class OverviewLLMOutput(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class OverviewResult(OverviewLLMOutput):
    model_config = ConfigDict(validate_assignment=True)

    usage: LLMUsage = Field(default_factory=LLMUsage)


def _default_telemetry() -> dict[str, Any]:
    return {"enabled": False, "trace_id": "noop", "spans": []}


class SearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: str
    results: list[ResultItem] = Field(default_factory=list)
    overview: OverviewResult | None = None
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=_default_telemetry)


__all__ = [
    "Citation",
    "OverviewLLMOutput",
    "OverviewResult",
    "PageChunk",
    "PageEnrichment",
    "ResultItem",
    "SearchResponse",
]

# Ensure forward references are resolved (Pydantic v2 + postponed annotations).
SearchResponse.model_rebuild()
