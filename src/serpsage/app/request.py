from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SearchDepth = Literal["simple", "low", "medium", "high"]
FetchContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]
CrawlMode = Literal["never", "fallback", "preferred", "always"]


def _validate_json_schema(value: object | None) -> object | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("json_schema must be a JSON object")
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:  # noqa: BLE001
        raise ValueError("jsonschema dependency is required") from exc
    Draft202012Validator.check_schema(value)
    return value


class SearchOverviewRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    json_schema: object | None = None

    @field_validator("json_schema")
    @classmethod
    def _validate_schema(cls, value: object | None) -> object | None:
        return _validate_json_schema(value)


class SearchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    depth: SearchDepth = "simple"
    max_results: int | None = None
    overview: bool | SearchOverviewRequest | None = None
    params: dict[str, object] = Field(default_factory=dict)


class FetchOthersRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_links: int | None = None
    max_image_links: int | None = None

    @field_validator("max_links", "max_image_links")
    @classmethod
    def _validate_positive_limit(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("limit must be > 0")
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


class FetchAbstractsRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    max_chars: int | None = None
    top_k_abstracts: int | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = str(value or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("max_chars", "top_k_abstracts")
    @classmethod
    def _validate_positive_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("value must be > 0")
        return value


class FetchOverviewRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    json_schema: object | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = str(value or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("json_schema")
    @classmethod
    def _validate_schema(cls, value: object | None) -> object | None:
        return _validate_json_schema(value)


class FetchSubpagesRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_subpages: int | None = None
    subpage_keywords: str | None = None

    @field_validator("max_subpages")
    @classmethod
    def _validate_max_subpages(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("max_subpages must be > 0")
        return value

    @field_validator("subpage_keywords")
    @classmethod
    def _validate_subpage_keywords(cls, value: str | None) -> str | None:
        if value is None:
            return None
        out = str(value).strip()
        return out or None


class FetchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    urls: list[str]
    crawl_mode: CrawlMode = "fallback"
    crawl_timeout: float | None = None
    content: bool | FetchContentRequest
    abstracts: FetchAbstractsRequest | None = None
    subpages: FetchSubpagesRequest | None = None
    overview: FetchOverviewRequest | None = None
    others: FetchOthersRequest = Field(default_factory=FetchOthersRequest)

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
    "FetchOthersRequest",
    "FetchContentRequest",
    "FetchAbstractsRequest",
    "FetchSubpagesRequest",
    "FetchOverviewRequest",
    "SearchOverviewRequest",
    "SearchDepth",
    "SearchRequest",
]
