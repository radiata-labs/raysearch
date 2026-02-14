from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SearchDepth = Literal["simple", "low", "medium", "high"]
FetchContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]


class SearchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: SearchDepth = "simple"
    max_results: int | None = None
    profile: str | None = None
    overview: bool | None = None
    params: dict[str, object] = Field(default_factory=dict)


class FetchContentRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_chars: int | None = None
    depth: Literal["low", "medium", "high"] = "low"
    include_html_tags: bool = False
    include_tags: list[FetchContentTag] = Field(default_factory=list)
    exclude_tags: list[FetchContentTag] = Field(default_factory=list)

    @field_validator("max_chars")
    @classmethod
    def _validate_max_chars(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_chars must be > 0")
        return value

    @model_validator(mode="after")
    def _validate_tag_overlap(self) -> FetchContentRequest:
        overlap = sorted(set(self.include_tags) & set(self.exclude_tags))
        if overlap:
            raise ValueError(
                "include_tags and exclude_tags must not overlap: "
                + ", ".join(overlap)
            )
        return self


class FetchChunksRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    max_chars: int | None = None
    top_k_chunks: int | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = str(value or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("max_chars")
    @classmethod
    def _validate_max_chars(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_chars must be > 0")
        return value

    @field_validator("top_k_chunks")
    @classmethod
    def _validate_top_k_chunks(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("top_k_chunks must be > 0")
        return value


class FetchOverviewRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    max_chars: int | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = str(value or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("max_chars")
    @classmethod
    def _validate_max_chars(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_chars must be > 0")
        return value


class FetchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    url: str
    content: bool | FetchContentRequest
    profile: str | None = None
    chunks: FetchChunksRequest | None = None
    overview: FetchOverviewRequest | None = None
    params: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "FetchContentTag",
    "FetchRequest",
    "FetchContentRequest",
    "FetchChunksRequest",
    "FetchOverviewRequest",
    "SearchDepth",
    "SearchRequest",
]
