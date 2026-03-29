from __future__ import annotations

from pydantic import field_validator

from raysearch.models.base import MutableModel
from raysearch.utils import clean_whitespace, normalize_iso8601_string


class SearchProviderResult(MutableModel):
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""
    published_date: str = ""
    pre_fetched_content: str = ""
    pre_fetched_author: str = ""

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        out = clean_whitespace(str(value or ""))
        if not out:
            raise ValueError("url must not be empty")
        return out

    @field_validator("title", "snippet", "engine")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("published_date")
    @classmethod
    def _validate_published_date(cls, value: str) -> str:
        token = clean_whitespace(str(value or ""))
        if not token:
            return ""
        return normalize_iso8601_string(token)


__all__ = [
    "SearchProviderResult",
]
