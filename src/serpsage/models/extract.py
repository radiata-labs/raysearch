from __future__ import annotations

from pydantic import Field

from serpsage.core.model_base import FrozenModel


class ExtractedText(FrozenModel):
    text: str = ""
    blocks: list[str] = Field(default_factory=list)


__all__ = ["ExtractedText"]
