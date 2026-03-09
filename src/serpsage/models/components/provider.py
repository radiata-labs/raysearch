from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from serpsage.models.base import MutableModel
from serpsage.utils import clean_whitespace


class SearchProviderResult(MutableModel):
    url: str
    title: str = ""
    snippet: str = ""
    display_url: str = ""
    source_engine: str = ""
    position: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        out = clean_whitespace(str(value or ""))
        if not out:
            raise ValueError("url must not be empty")
        return out

    @field_validator("title", "snippet", "display_url", "source_engine")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("position")
    @classmethod
    def _validate_position(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("position must be > 0")
        return int(value)


class SearchProviderResponse(MutableModel):
    provider_backend: str
    query: str
    page: int = 1
    language: str = ""
    total_results: int | None = None
    suggestions: list[str] = Field(default_factory=list)
    results: list[SearchProviderResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_backend", "query")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        out = clean_whitespace(str(value or ""))
        if not out:
            raise ValueError("value must not be empty")
        return out

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("page")
    @classmethod
    def _validate_page(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("page must be > 0")
        return int(value)

    @field_validator("total_results")
    @classmethod
    def _validate_total_results(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) < 0:
            raise ValueError("total_results must be >= 0")
        return int(value)

    @field_validator("suggestions")
    @classmethod
    def _validate_suggestions(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = clean_whitespace(str(raw or ""))
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out


__all__ = [
    "SearchProviderResponse",
    "SearchProviderResult",
]
