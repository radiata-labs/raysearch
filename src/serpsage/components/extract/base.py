from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.components.extract.utils import (
    finalize_markdown,
    markdown_to_abstract_text,
    strip_markdown_links,
)

if TYPE_CHECKING:
    from serpsage.models.components.extract import (
        ExtractContent,
        ExtractedDocument,
        ExtractSpec,
    )


class ExtractConfigBase(ComponentConfigBase):
    max_markdown_chars: int = 160_000
    min_text_chars: int = 220
    min_primary_chars: int = 220
    min_total_chars_with_secondary: int = 220
    include_secondary_content_default: bool = False
    collect_links_default: bool = False
    link_max_count: int = 800
    link_keep_hash: bool = False


class ExtractorBase(ComponentBase[ExtractConfigBase], ABC):
    def _finalize_content(
        self,
        *,
        doc: ExtractedDocument,
        content_options: ExtractSpec | None,
    ) -> ExtractedDocument:
        options = content_options
        raw_markdown = str(doc.content.markdown or "")
        output_markdown = ""
        if options is None or options.emit_output:
            output_markdown = raw_markdown
            if options is not None and not options.keep_markdown_links:
                output_markdown = strip_markdown_links(output_markdown)
            if (
                options is not None
                and options.output_max_chars is not None
                and options.output_max_chars > 0
            ):
                output_markdown = finalize_markdown(
                    markdown=output_markdown,
                    max_chars=int(options.output_max_chars),
                )
        return doc.model_copy(
            update={
                "content": self._build_content_fields(
                    content=doc.content,
                    raw_markdown=raw_markdown,
                    output_markdown=output_markdown,
                )
            }
        )

    def _build_content_fields(
        self,
        *,
        content: ExtractContent,
        raw_markdown: str,
        output_markdown: str,
    ) -> ExtractContent:
        return content.model_copy(
            update={
                "markdown": raw_markdown,
                "output_markdown": output_markdown,
                "abstract_text": markdown_to_abstract_text(raw_markdown),
            }
        )

    @abstractmethod
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
        raise NotImplementedError


__all__ = ["ExtractConfigBase", "ExtractorBase"]
