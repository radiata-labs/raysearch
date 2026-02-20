from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from serpsage.utils.normalize import clean_whitespace

SearchDepth = Literal["auto", "deep"]
FetchContentDetail = Literal["concise", "standard", "full"]
FetchContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]
CrawlMode = Literal["never", "fallback", "preferred", "always"]

_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]")


def _normalize_domain(value: str) -> str:
    text = clean_whitespace(value).lower().strip(".")
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0].split(":", 1)[0].strip()
    return text.removeprefix("www.")


def _normalize_string_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = clean_whitespace(str(raw or ""))
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out or None


def _validate_text_phrase_limit(value: str) -> None:
    cleaned = clean_whitespace(value)
    cjk_count = len(_CJK_CHAR_RE.findall(cleaned))
    word_count = len([x for x in cleaned.split(" ") if x])
    latin_word_count = len(_LATIN_WORD_RE.findall(cleaned))
    if cjk_count > 0 and latin_word_count <= 1:
        if cjk_count > 6:
            raise ValueError(
                "each text filter phrase supports at most 6 Chinese/Japanese characters"
            )
        return
    if word_count > 5:
        raise ValueError("each text filter phrase supports at most 5 words")


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
    detail: FetchContentDetail = "concise"
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

    query: str | None = None
    max_chars: int | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str | None) -> str | None:
        query = str(value or "").strip()
        return query or None

    @field_validator("max_chars")
    @classmethod
    def _validate_positive_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("value must be > 0")
        return value


class FetchOverviewRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str | None = None
    json_schema: object | None = None

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str | None) -> str | None:
        query = str(value or "").strip()
        return query or None

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


class FetchRequestBase(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    crawl_mode: CrawlMode = "fallback"
    crawl_timeout: float | None = None
    content: bool | FetchContentRequest = False
    abstracts: bool | FetchAbstractsRequest = False
    subpages: FetchSubpagesRequest | None = None
    overview: bool | FetchOverviewRequest = False
    others: FetchOthersRequest | None = None

    @field_validator("crawl_timeout")
    @classmethod
    def _validate_crawl_timeout(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("crawl_timeout must be > 0")
        return float(value)

    @model_validator(mode="after")
    def _validate_has_action(self) -> FetchRequestBase:
        content_enabled = not isinstance(self.content, bool) or bool(self.content)
        abstracts_enabled = not isinstance(self.abstracts, bool) or bool(self.abstracts)
        overview_enabled = not isinstance(self.overview, bool) or bool(self.overview)
        subpages_enabled = self.subpages is not None
        others_enabled = self.others is not None and (
            self.others.max_links is not None or self.others.max_image_links is not None
        )
        if not (
            content_enabled
            or abstracts_enabled
            or overview_enabled
            or subpages_enabled
            or others_enabled
        ):
            raise ValueError(
                "fetch request has nothing to do: enable at least one of "
                "content/abstracts/overview/subpages/others"
            )
        return self


class FetchRequest(FetchRequestBase):
    urls: list[str]

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


class SearchRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    query: str
    additional_queries: list[str] | None = None
    depth: SearchDepth = "auto"
    max_results: int | None = None
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    include_text: list[str] | None = None
    exclude_text: list[str] | None = None
    fetchs: FetchRequestBase

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = clean_whitespace(str(value or ""))
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("additional_queries")
    @classmethod
    def _validate_additional_queries(cls, value: list[str] | None) -> list[str] | None:
        return _normalize_string_list(value)

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def _validate_domains(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            domain = _normalize_domain(str(raw or ""))
            if not domain:
                continue
            if domain in seen:
                continue
            seen.add(domain)
            out.append(domain)
        return out or None

    @field_validator("include_text", "exclude_text")
    @classmethod
    def _validate_text_filters(cls, value: list[str] | None) -> list[str] | None:
        normalized = _normalize_string_list(value)
        if normalized is None:
            return None
        for item in normalized:
            _validate_text_phrase_limit(item)
        return normalized

    @field_validator("max_results")
    @classmethod
    def _validate_max_results(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("max_results must be > 0")
        return int(value)

    @model_validator(mode="after")
    def _validate_search_request(self) -> SearchRequest:
        if self.depth == "auto" and self.additional_queries:
            raise ValueError("additional_queries is only supported when depth=deep")
        include_set = set(self.include_domains or [])
        exclude_set = set(self.exclude_domains or [])
        overlap = sorted(include_set & exclude_set)
        if overlap:
            raise ValueError(
                "include_domains and exclude_domains must not overlap: "
                + ", ".join(overlap)
            )
        return self


class AnswerRequest(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    query: str
    json_schema: object | None = None
    content: bool = False

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        query = clean_whitespace(str(value or ""))
        if not query:
            raise ValueError("query must not be empty")
        return query

    @field_validator("json_schema")
    @classmethod
    def _validate_schema(cls, value: object | None) -> object | None:
        return _validate_json_schema(value)


__all__ = [
    "CrawlMode",
    "FetchContentDetail",
    "FetchContentTag",
    "FetchRequestBase",
    "FetchRequest",
    "FetchOthersRequest",
    "FetchContentRequest",
    "FetchAbstractsRequest",
    "FetchSubpagesRequest",
    "FetchOverviewRequest",
    "SearchDepth",
    "SearchRequest",
    "AnswerRequest",
]
