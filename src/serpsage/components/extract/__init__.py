from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import ExtractorBase
    from serpsage.core.runtime import Runtime


def build_extractor(*, rt: Runtime) -> ExtractorBase:
    backend = str(rt.settings.enrich.extractor.backend or "main_content").lower()
    if backend == "main_content":
        from serpsage.components.extract.main import MainContentHtmlExtractor

        return MainContentHtmlExtractor(rt=rt)
    if backend == "basic":
        from serpsage.components.extract.basic import BasicHtmlExtractor

        return BasicHtmlExtractor(rt=rt)
    raise ValueError(
        f"unsupported extractor backend `{backend}`; expected basic|main_content"
    )


__all__ = [
    "build_extractor",
]
