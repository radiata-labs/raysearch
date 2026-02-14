from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchDepth = Literal["simple", "low", "medium", "high"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: SearchDepth = "simple"
    max_results: int | None = None
    profile: str | None = None
    overview: bool | None = None
    params: dict[str, object] = Field(default_factory=dict)


class FetchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    url: str
    query: str | None = None
    profile: str | None = None
    include_chunks: bool | None = None
    top_k_chunks: int | None = None
    include_secondary_content: bool | None = None
    overview: bool | None = None
    params: dict[str, object] = Field(default_factory=dict)


__all__ = ["FetchRequest", "SearchDepth", "SearchRequest"]
