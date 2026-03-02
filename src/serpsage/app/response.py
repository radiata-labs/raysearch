from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

FetchErrorTag = Literal[
    "CRAWL_NOT_FOUND",
    "CRAWL_TIMEOUT",
    "CRAWL_LIVECRAWL_TIMEOUT",
    "SOURCE_NOT_AVAILABLE",
    "UNSUPPORTED_URL",
    "CRAWL_UNKNOWN_ERROR",
]


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
    others: FetchOthersResult | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):  # type: ignore[no-untyped-def]
        payload = handler(self)
        if payload.get("others") is None:
            payload.pop("others", None)
        return payload


class FetchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    request_id: str
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


class SearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    request_id: str
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


class AnswerResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    request_id: str
    answer: str | object
    citations: list[AnswerCitation] = Field(default_factory=list)


class ResearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    request_id: str
    content: str
    structured: object | None = None


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
]
FetchResultItem.model_rebuild()
FetchStatusError.model_rebuild()
FetchStatusItem.model_rebuild()
FetchResponse.model_rebuild()
SearchResponse.model_rebuild()
