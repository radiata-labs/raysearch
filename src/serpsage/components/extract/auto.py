from __future__ import annotations

from typing import Any
from typing_extensions import override

from serpsage.components.base import ComponentMeta
from serpsage.components.crawl.utils import classify_content_kind
from serpsage.components.extract.base import ExtractConfigBase, ExtractorBase
from serpsage.models.components.extract import ExtractedDocument, ExtractSpec


class AutoExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "auto"


_AUTO_EXTRACTOR_META = ComponentMeta(
    version="1.0.0",
    summary="Automatic extractor dispatching between HTML and PDF handlers.",
)


class AutoExtractor(ExtractorBase[AutoExtractorConfig]):
    meta = _AUTO_EXTRACTOR_META

    def __init__(self) -> None:
        super().__init__()
        self._html_extractor = _coerce_extractor(
            "html",
            self.components.require_component_optional("extract", "html"),
        )
        self._pdf_extractor = _coerce_extractor(
            "pdf",
            self.components.require_component_optional("extract", "pdf"),
        )
        self.bind_deps(self._html_extractor, self._pdf_extractor)

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
            pdf_extractor = _require_extractor("pdf", self._pdf_extractor)
            return await pdf_extractor.extract(
                url=url,
                content=content,
                content_type=content_type,
                content_options=content_options,
                include_secondary_content=include_secondary_content,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        html_extractor = _require_extractor("html", self._html_extractor)
        return await html_extractor.extract(
            url=url,
            content=content,
            content_type=content_type,
            content_options=content_options,
            include_secondary_content=include_secondary_content,
            collect_links=collect_links,
            collect_images=collect_images,
        )


def _coerce_extractor(
    name: str,
    value: object | None,
) -> ExtractorBase[Any] | None:
    if value is None:
        return None
    if not isinstance(value, ExtractorBase):
        raise TypeError(f"extract component `{name}` must implement ExtractorBase")
    return value


def _require_extractor(
    name: str,
    value: ExtractorBase[Any] | None,
) -> ExtractorBase[Any]:
    if value is None:
        raise RuntimeError(f"extract component `{name}` is not enabled")
    return value


__all__ = ["AutoExtractor", "AutoExtractorConfig"]
