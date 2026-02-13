from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import ExtractorBase
    from serpsage.core.runtime import Runtime


def build_extractor(*, rt: Runtime) -> ExtractorBase:
    from serpsage.components.extract.markdown import MarkdownExtractor

    return MarkdownExtractor(rt=rt)


__all__ = [
    "build_extractor",
]
