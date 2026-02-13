from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.core.model_base import FrozenModel


class ExtractedDocument(FrozenModel):
    markdown: str = ""
    plain_text: str = ""
    title: str = ""
    content_kind: Literal["html", "pdf", "text", "binary"] = "binary"
    extractor_used: str = ""
    quality_score: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int | float | str] = Field(default_factory=dict)


__all__ = ["ExtractedDocument"]
