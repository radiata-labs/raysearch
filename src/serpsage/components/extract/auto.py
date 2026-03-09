from __future__ import annotations

from typing_extensions import override

from serpsage.components.base import ComponentMeta
from serpsage.components.extract.base import ExtractConfigBase, ExtractorBase
from serpsage.components.extract.html import HtmlExtractor
from serpsage.components.extract.pdf import PdfExtractor
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.components.registry import register_component
from serpsage.dependencies import Inject
from serpsage.models.components.extract import ExtractedDocument, ExtractSpec

_AUTO_EXTRACTOR_META = ComponentMeta(
    family="extract",
    name="auto",
    version="1.0.0",
    summary="Automatic extractor dispatching between HTML and PDF handlers.",
    provides=("extractor.document",),
    config_model=ExtractConfigBase,
)


@register_component(meta=_AUTO_EXTRACTOR_META)
class AutoExtractor(ExtractorBase):
    meta = _AUTO_EXTRACTOR_META

    html_extractor: HtmlExtractor = Inject()
    pdf_extractor: PdfExtractor = Inject()

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractSpec | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        kind = classify_content_kind(
            content_type=content_type, url=url, content=content
        )
        if kind == "pdf":
            return await self.pdf_extractor.extract(
                url=url,
                content=content,
                content_type=content_type,
                content_options=content_options,
                include_secondary_content=include_secondary_content,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        return await self.html_extractor.extract(
            url=url,
            content=content,
            content_type=content_type,
            content_options=content_options,
            include_secondary_content=include_secondary_content,
            collect_links=collect_links,
            collect_images=collect_images,
        )


__all__ = ["AutoExtractor"]
