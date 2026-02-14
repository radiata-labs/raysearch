from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SearchDepth = Literal["simple", "low", "medium", "high"]
FetchContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]
CrawlMode = Literal["never", "fallback", "preferred", "always"]


class SearchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: SearchDepth = "simple"
    max_results: int | None = None
    profile: str | None = None
    overview: bool | None = None
    params: dict[str, object] = Field(default_factory=dict)


class FetchRuntimeRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_links: int | None = None

    @field_validator("max_links")
    @classmethod
    def _validate_max_links(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_links must be > 0")
        return value


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
                "include_tags and exclude_tags must not overlap: " + ", ".join(overlap)
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

    urls: list[str]
    crawl_mode: CrawlMode = "fallback"
    crawl_timeout: float | None = None
    content: bool | FetchContentRequest
    chunks: FetchChunksRequest | None = None
    overview: FetchOverviewRequest | None = None
    runtime: FetchRuntimeRequest = Field(default_factory=FetchRuntimeRequest)

    @field_validator("urls")
    @classmethod
    def _validate_urls(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for raw in value:
            url = str(raw or "").strip()
            if not url:
                raise ValueError("urls must not contain empty value")
            if not (url.startswith(("http://", "https://"))):
                raise ValueError(f"unsupported url scheme: {url}")
            out.append(url)
        if not out:
            raise ValueError("urls must not be empty")
        return out

    @field_validator("crawl_timeout")
    @classmethod
    def _validate_crawl_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("crawl_timeout must be > 0")
        return float(value)


__all__ = [
    "CrawlMode",
    "FetchContentTag",
    "FetchRequest",
    "FetchRuntimeRequest",
    "FetchContentRequest",
    "FetchChunksRequest",
    "FetchOverviewRequest",
    "SearchDepth",
    "SearchRequest",
]
