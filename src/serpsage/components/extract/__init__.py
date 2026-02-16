from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import ExtractorBase
    from serpsage.core.runtime import Runtime


def build_extractor(*, rt: Runtime) -> ExtractorBase:
    from serpsage.components.extract.auto import AutoExtractor
    from serpsage.components.extract.markdown import MarkdownExtractor
    from serpsage.components.extract.pdf import PdfExtractor

    markdown_extractor = MarkdownExtractor(rt=rt)
    pdf_extractor = PdfExtractor(rt=rt)
    return AutoExtractor(
        rt=rt,
        markdown_extractor=markdown_extractor,
        pdf_extractor=pdf_extractor,
    )


__all__ = [
    "build_extractor",
]
