from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import ExtractorBase
    from serpsage.core.runtime import Runtime


def build_extractor(*, rt: Runtime) -> ExtractorBase:
    backend = str(rt.settings.enrich.extractor.backend or "markdown").lower()
    if backend != "markdown":
        raise ValueError(
            f"unsupported extractor backend `{backend}`; expected markdown"
        )
    from serpsage.components.extract.markdown import MarkdownExtractor

    return MarkdownExtractor(rt=rt)


__all__ = [
    "build_extractor",
]
