from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from raysearch.models.app.base import BaseResponse
from raysearch.models.app.request import ResearchSearchMode
from raysearch.utils import normalize_iso8601_string

FetchErrorTag = Literal[
    "CRAWL_NOT_FOUND",
    "CRAWL_TIMEOUT",
    "CRAWL_LIVECRAWL_TIMEOUT",
    "SOURCE_NOT_AVAILABLE",
    "UNSUPPORTED_URL",
    "CRAWL_UNKNOWN_ERROR",
]

ResearchTaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "canceled",
    "failed",
]


class FetchOthersResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    links: list[str] = Field(default_factory=list)
    image_links: list[str] = Field(default_factory=list)


class FetchSubpagesResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    url: str
    title: str
    published_date: str = ""
    author: str = ""
    image: str = ""
    favicon: str = ""
    content: str
    abstracts: list[str]
    abstract_scores: list[float]
    overview: str | object | None = None

    @field_validator("published_date")
    @classmethod
    def _validate_published_date(cls, value: str) -> str:
        return normalize_iso8601_string(value, allow_blank=True)

    @model_validator(mode="after")
    def _validate_abstract_alignment(self) -> FetchSubpagesResult:
        if len(self.abstracts) != len(self.abstract_scores):
            raise ValueError("abstracts and abstract_scores length mismatch")
        return self


class FetchResultItem(FetchSubpagesResult):
    subpages: list[FetchSubpagesResult] = Field(default_factory=list)
    others: FetchOthersResult | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):  # type: ignore[no-untyped-def]
        payload = handler(self)
        if payload.get("others") is None:
            payload.pop("others", None)
        return payload


class FetchResponse(BaseResponse):
    results: list[FetchResultItem] = Field(default_factory=list)
    statuses: list[FetchStatusItem]


class FetchStatusError(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    tag: FetchErrorTag
    detail: str | None = None


class FetchStatusItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    url: str
    status: Literal["success", "error"]
    error: FetchStatusError | None = None


class SearchResponse(BaseResponse):
    search_mode: str
    results: list[FetchResultItem] = Field(default_factory=list)


class AnswerCitation(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    id: str
    url: str
    title: str
    content: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):  # type: ignore[no-untyped-def]
        payload = handler(self)
        if payload.get("content") is None:
            payload.pop("content", None)
        return payload


class AnswerResponse(BaseResponse):
    answer: str | object
    citations: list[AnswerCitation] = Field(default_factory=list)


class ResearchResponse(BaseResponse):
    content: str
    structured: object | None = None


class ResearchTaskResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    research_id: str
    create_at: int
    themes: str
    search_mode: ResearchSearchMode
    json_schema: dict[str, Any] | None = None
    status: ResearchTaskStatus
    output: ResearchResponse | None = None
    finished_at: int | None = None
    error: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):  # type: ignore[no-untyped-def]
        payload = handler(self)
        if payload.get("output") is None:
            payload.pop("output", None)
        if payload.get("finished_at") is None:
            payload.pop("finished_at", None)
        if payload.get("error") is None:
            payload.pop("error", None)
        return payload


class ResearchTaskListResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    data: list[ResearchTaskResponse] = Field(default_factory=list)
    has_more: bool
    next_cursor: str = ""


__all__ = [
    "FetchOthersResult",
    "FetchErrorTag",
    "FetchResponse",
    "FetchResultItem",
    "FetchStatusError",
    "FetchStatusItem",
    "FetchSubpagesResult",
    "SearchResponse",
    "AnswerCitation",
    "AnswerResponse",
    "ResearchResponse",
    "ResearchTaskStatus",
    "ResearchTaskResponse",
    "ResearchTaskListResponse",
]
FetchResultItem.model_rebuild()
FetchStatusError.model_rebuild()
FetchStatusItem.model_rebuild()
FetchResponse.model_rebuild()
SearchResponse.model_rebuild()
ResearchTaskResponse.model_rebuild()
ResearchTaskListResponse.model_rebuild()
