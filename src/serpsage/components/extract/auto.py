from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.extract.base import ExtractorBase
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.models.extract import ExtractContentOptions, ExtractedDocument

if TYPE_CHECKING:
    from serpsage.components.extract.markdown import MarkdownExtractor
    from serpsage.components.extract.pdf import PdfExtractor
    from serpsage.core.runtime import Runtime


class AutoExtractor(ExtractorBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        markdown_extractor: MarkdownExtractor,
        pdf_extractor: PdfExtractor,
    ) -> None:
        super().__init__(rt=rt)
        self._markdown_extractor = markdown_extractor
        self._pdf_extractor = pdf_extractor
        self.bind_deps(markdown_extractor, pdf_extractor)

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        kind = classify_content_kind(
            content_type=content_type,
            url=url,
            content=content,
        )
        if kind == "pdf":
            return await self._pdf_extractor.extract(
                url=url,
                content=content,
                content_type=content_type,
                content_options=content_options,
                include_secondary_content=include_secondary_content,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        return await self._markdown_extractor.extract(
            url=url,
            content=content,
            content_type=content_type,
            content_options=content_options,
            include_secondary_content=include_secondary_content,
            collect_links=collect_links,
            collect_images=collect_images,
        )


__all__ = ["AutoExtractor"]
