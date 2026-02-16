from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from serpsage.models.errors import AppError  # noqa: TC001


class PageAbstract(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    abstract_id: str | None = None
    text: str
    score: float = 0.0


class PageEnrichment(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    abstracts: list[PageAbstract] = Field(default_factory=list)
    markdown: str = ""
    content_kind: str | None = None
    fetch_mode: str | None = None
    timing_ms: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
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


def _default_telemetry() -> dict[str, Any]:
    return {"enabled": False, "trace_id": "noop", "spans": []}


class SearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: str
    results: list[ResultItem] = Field(default_factory=list)
    overview: str | object | None = None
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=_default_telemetry)


class FetchOthersResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    links: list[str] = Field(default_factory=list)
    image_links: list[str] = Field(default_factory=list)


class FetchSubpagesResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    url: str
    title: str
    content: str
    abstracts: list[str]
    abstract_scores: list[float]
    overview: str | object | None = None

    @model_validator(mode="after")
    def _validate_abstract_alignment(self) -> FetchSubpagesResult:
        if len(self.abstracts) != len(self.abstract_scores):
            raise ValueError("abstracts and abstract_scores length mismatch")
        return self


class FetchResultItem(FetchSubpagesResult):
    subpages: list[FetchSubpagesResult] = Field(default_factory=list)
    others: FetchOthersResult = Field(default_factory=FetchOthersResult)


class FetchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    results: list[FetchResultItem] = Field(default_factory=list)
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=_default_telemetry)


__all__ = [
    "FetchOthersResult",
    "FetchResponse",
    "FetchResultItem",
    "FetchSubpagesResult",
    "PageAbstract",
    "PageEnrichment",
    "ResultItem",
    "SearchResponse",
]

SearchResponse.model_rebuild()
FetchResultItem.model_rebuild()
FetchResponse.model_rebuild()
