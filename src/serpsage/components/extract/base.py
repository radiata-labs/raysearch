from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic
from typing_extensions import TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.components.extract.utils import (
    finalize_markdown,
    markdown_to_abstract_text,
    strip_markdown_links,
)

if TYPE_CHECKING:
    from serpsage.models.components.extract import (
        ExtractedDocument,
        ExtractSpec,
    )


class ExtractConfigBase(ComponentConfigBase):
    max_markdown_chars: int = 160_000
    min_text_chars: int = 220
    min_primary_chars: int = 220
    min_total_chars_with_secondary: int = 220
    collect_links_default: bool = False
    link_max_count: int = 800
    link_keep_hash: bool = False


ExtractConfigT = TypeVar(
    "ExtractConfigT",
    bound=ExtractConfigBase,
    default=ExtractConfigBase,
)


class ExtractorBase(ComponentBase[ExtractConfigT], ABC, Generic[ExtractConfigT]):
    @abstractmethod
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractSpec | None = None,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        raise NotImplementedError

    def _finalize_content(
        self,
        *,
        doc: ExtractedDocument,
        content_options: ExtractSpec | None,
    ) -> ExtractedDocument:
        options = content_options
        raw_markdown = str(doc.content.markdown or "")
        output_markdown = ""

        # Build metadata dict from doc.meta for frontmatter
        meta = doc.meta
        metadata: dict[str, str] = {}
        if meta.title:
            metadata["title"] = meta.title
        if meta.author:
            metadata["author"] = meta.author
        if meta.published_date:
            metadata["published_date"] = meta.published_date
        if meta.image:
            metadata["image"] = meta.image
        if meta.favicon:
            metadata["favicon"] = meta.favicon

        if options is None or options.emit_output:
            output_markdown = raw_markdown
            if options is not None and not options.keep_markdown_links:
                output_markdown = strip_markdown_links(output_markdown)

            # Determine max_chars for output
            output_max_chars = (
                int(options.output_max_chars)
                if options is not None and options.output_max_chars is not None and options.output_max_chars > 0
                else 10_000_000  # Large enough to not clip
            )

            # Always finalize with metadata (adds frontmatter and cleans up)
            output_markdown = finalize_markdown(
                markdown=output_markdown,
                max_chars=output_max_chars,
                metadata=metadata or None,
            )

        return doc.model_copy(
            update={
                "content": doc.content.model_copy(
                    update={
                        "markdown": raw_markdown,
                        "output_markdown": output_markdown,
                        "abstract_text": markdown_to_abstract_text(raw_markdown),
                    }
                )
            }
        )


class SpecializedExtractorBase(
    ExtractorBase[ExtractConfigT], ABC, Generic[ExtractConfigT]
):
    """Base class for specialized extractors that handle specific content types.

    Specialized extractors implement `can_handle` to declare what they can process.
    The AutoExtractor will try each specialized extractor in order and use the
    first one that returns True. HtmlExtractor is used as fallback.
    """

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        *,
        url: str,
        content_type: str | None,
        content: bytes | None = None,
    ) -> bool:
        """Return True if this extractor should handle the given content.

        Args:
            url: The URL being extracted
            content_type: The Content-Type header
            content: The raw content bytes (may be None for early detection)

        Returns:
            True if this extractor should handle the content
        """
        raise NotImplementedError


__all__ = [
    "ExtractConfigBase",
    "ExtractConfigT",
    "ExtractorBase",
    "SpecializedExtractorBase",
]
