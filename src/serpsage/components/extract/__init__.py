from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.extract.html_basic import BasicHtmlExtractor
from serpsage.components.extract.html_main import MainContentHtmlExtractor
from serpsage.contracts.services import ExtractorBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


def build_extractor(*, rt: Runtime) -> ExtractorBase:
    backend = str(rt.settings.enrich.extractor.backend or "main_content").lower()
    if backend == "main_content":
        return MainContentHtmlExtractor(rt=rt)
    if backend == "basic":
        return BasicHtmlExtractor(rt=rt)
    raise ValueError(
        f"unsupported extractor backend `{backend}`; expected basic|main_content"
    )


__all__ = [
    "BasicHtmlExtractor",
    "MainContentHtmlExtractor",
    "build_extractor",
]
