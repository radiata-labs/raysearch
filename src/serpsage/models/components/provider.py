from __future__ import annotations

from pydantic import field_validator

from serpsage.models.base import MutableModel
from serpsage.utils import clean_whitespace


class SearchProviderResult(MutableModel):
    url: str
    title: str = ""
    snippet: str = ""
    engine: str = ""

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        out = clean_whitespace(str(value or ""))
        if not out:
            raise ValueError("url must not be empty")
        return out

    @field_validator("title", "snippet")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))


__all__ = [
    "SearchProviderResult",
]
