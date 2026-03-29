"""Auto-selecting extractor that dispatches to specialized extractors.

Resolution order:
1. Try each SpecializedExtractorBase's can_handle()
2. Use the first one that returns True
3. Fall back to HtmlExtractor

Configuration
=============

Specialized extractors are discovered automatically from the component registry.
To add a new specialized extractor:

1. Create a class inheriting from SpecializedExtractorBase
2. Implement can_handle() classmethod
3. Register it in the component configuration

Example configuration in this project:

.. code:: yaml

   extract:
     auto:
       enabled: true
     html:
       enabled: true
     pdf:
       enabled: true
     paper:
       enabled: true
"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import override

from raysearch.components.extract.base import (
    ExtractConfigBase,
    ExtractorBase,
    SpecializedExtractorBase,
)
from raysearch.components.extract.html import HtmlExtractor
from raysearch.components.loads import ComponentRegistry
from raysearch.dependencies import CACHE_TOKEN, Depends, solve_dependencies
from raysearch.models.components.extract import ExtractedDocument, ExtractSpec


class AutoExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "auto"


async def specialized_extractors_factory(
    cache: dict[Any, Any] = Depends(CACHE_TOKEN),
    registry: ComponentRegistry = Depends(),
) -> tuple[SpecializedExtractorBase, ...]:
    """Factory function: collect all enabled specialized extractors."""
    extractors: list[SpecializedExtractorBase] = []
    for spec in registry.enabled_specs("extract"):
        # Skip auto and html (html is the fallback)
        if spec.name in ("auto", "html"):
            continue
        if not issubclass(spec.cls, SpecializedExtractorBase):
            continue
        instance = await solve_dependencies(spec.cls, dependency_cache=cache)
        if isinstance(instance, SpecializedExtractorBase):
            extractors.append(instance)
    return tuple(extractors)


class AutoExtractor(ExtractorBase[AutoExtractorConfig]):
    html_extractor: HtmlExtractor = Depends()

    def __init__(
        self,
        *,
        specialized: tuple[SpecializedExtractorBase, ...] = Depends(
            specialized_extractors_factory
        ),
    ) -> None:
        self.specialized = specialized

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        crawl_backend: str = "curl_cffi",
        content_kind: Literal[
            "html", "pdf", "text", "markdown", "json", "binary", "unknown"
        ] = "unknown",
        content_options: ExtractSpec | None = None,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        # Try specialized extractors in order
        for extractor in self.specialized:
            if type(extractor).can_handle(
                url=url,
                content_type=content_type,
                crawl_backend=crawl_backend,
                content_kind=content_kind,
                content=content,
            ):
                return await extractor.extract(
                    url=url,
                    content=content,
                    content_type=content_type,
                    crawl_backend=crawl_backend,
                    content_kind=content_kind,
                    content_options=content_options,
                    collect_links=collect_links,
                    collect_images=collect_images,
                )

        # Fall back to HtmlExtractor
        return await self.html_extractor.extract(
            url=url,
            content=content,
            content_type=content_type,
            crawl_backend=crawl_backend,
            content_kind=content_kind,
            content_options=content_options,
            collect_links=collect_links,
            collect_images=collect_images,
        )


__all__ = [
    "AutoExtractor",
    "AutoExtractorConfig",
    "specialized_extractors_factory",
]
