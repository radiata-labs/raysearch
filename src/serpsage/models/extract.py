from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.core.model_base import FrozenModel

ExtractContentDepth = Literal["low", "medium", "high"]
ExtractContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]


class ExtractContentOptions(FrozenModel):
    depth: ExtractContentDepth = "low"
    include_html_tags: bool = False
    include_tags: list[ExtractContentTag] = Field(default_factory=list)
    exclude_tags: list[ExtractContentTag] = Field(default_factory=list)


class ExtractedLink(FrozenModel):
    url: str = ""
    anchor_text: str = ""
    section: Literal["primary", "secondary"] = "primary"
    is_internal: bool = False
    nofollow: bool = False
    same_page: bool = False
    source_hint: str = ""
    position: int = 0


class ExtractedImageLink(FrozenModel):
    url: str = ""
    alt_text: str = ""
    section: Literal["primary", "secondary"] = "primary"
    is_internal: bool = False
    source_hint: str = ""
    position: int = 0


class ExtractedDocument(FrozenModel):
    markdown: str = ""
    title: str = ""
    content_kind: Literal["html", "pdf", "text", "binary"] = "binary"
    extractor_used: str = ""
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int | float | str | bool] = Field(default_factory=dict)
    links: list[ExtractedLink] = Field(default_factory=list)
    image_links: list[ExtractedImageLink] = Field(default_factory=list)


__all__ = [
    "ExtractContentDepth",
    "ExtractContentOptions",
    "ExtractContentTag",
    "ExtractedDocument",
    "ExtractedImageLink",
    "ExtractedLink",
]
