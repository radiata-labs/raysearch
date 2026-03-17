from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.models.base import FrozenModel

ExtractContentDetail = Literal["concise", "standard", "full"]
ExtractContentTag = Literal[
    "header", "navigation", "banner", "body", "sidebar", "footer", "metadata"
]
ExtractRefZone = Literal["primary", "secondary"]
ExtractContentKind = Literal["html", "pdf", "text", "json", "binary"]


class ExtractSpec(FrozenModel):
    detail: ExtractContentDetail = "concise"
    keep_html: bool = False
    sections: list[ExtractContentTag] = Field(default_factory=list)
    emit_output: bool = True
    keep_markdown_links: bool = False
    output_max_chars: int | None = None


class ExtractRef(FrozenModel):
    url: str = ""
    text: str = ""
    zone: ExtractRefZone = "primary"
    internal: bool = False
    nofollow: bool = False
    same_page: bool = False
    source: str = ""
    position: int = 0


class ExtractContent(FrozenModel):
    markdown: str = ""
    output_markdown: str = ""
    abstract_text: str = ""


class ExtractMeta(FrozenModel):
    title: str = ""
    published_date: str = ""
    author: str = ""
    image: str = ""
    favicon: str = ""


class ExtractTrace(FrozenModel):
    kind: ExtractContentKind = "binary"
    engine: str = ""
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int | float | str | bool] = Field(default_factory=dict)


class ExtractRefs(FrozenModel):
    links: list[ExtractRef] = Field(default_factory=list)
    images: list[ExtractRef] = Field(default_factory=list)


class ExtractedDocument(FrozenModel):
    content: ExtractContent = Field(default_factory=ExtractContent)
    meta: ExtractMeta = Field(default_factory=ExtractMeta)
    refs: ExtractRefs = Field(default_factory=ExtractRefs)
    trace: ExtractTrace = Field(default_factory=ExtractTrace)


__all__ = [
    "ExtractContent",
    "ExtractContentDetail",
    "ExtractContentKind",
    "ExtractContentTag",
    "ExtractMeta",
    "ExtractRef",
    "ExtractRefs",
    "ExtractRefZone",
    "ExtractedDocument",
    "ExtractSpec",
    "ExtractTrace",
]
