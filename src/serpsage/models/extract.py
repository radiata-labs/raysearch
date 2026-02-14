from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.core.model_base import FrozenModel


class ExtractedLink(FrozenModel):
    url: str = ""
    anchor_text: str = ""
    section: Literal["primary", "secondary"] = "primary"
    is_internal: bool = False
    nofollow: bool = False
    same_page: bool = False
    source_hint: str = ""
    position: int = 0


class ExtractedDocument(FrozenModel):
    markdown: str = ""
    plain_text: str = ""
    title: str = ""
    content_kind: Literal["html", "pdf", "text", "binary"] = "binary"
    extractor_used: str = ""
    quality_score: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int | float | str | bool] = Field(default_factory=dict)
    links: list[ExtractedLink] = Field(default_factory=list)


__all__ = ["ExtractedDocument", "ExtractedLink"]
