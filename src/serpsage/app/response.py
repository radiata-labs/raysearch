from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from serpsage.models.errors import AppError  # noqa: TC001


def _default_telemetry() -> dict[str, Any]:
    return {"enabled": False, "trace_id": "noop", "spans": []}


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

    results: list[FetchResultItem] = Field(default_factory=list)
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=_default_telemetry)


class SearchResponse(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    search_depth: str
    results: list[FetchResultItem] = Field(default_factory=list)
    errors: list[AppError] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=_default_telemetry)


__all__ = [
    "FetchOthersResult",
    "FetchResponse",
    "FetchResultItem",
    "FetchSubpagesResult",
    "SearchResponse",
]

FetchResultItem.model_rebuild()
FetchResponse.model_rebuild()
SearchResponse.model_rebuild()
