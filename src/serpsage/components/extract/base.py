from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.models.extract import ExtractContentOptions, ExtractedDocument


class ExtractorBase(WorkUnit, ABC):
    @abstractmethod
    def extract(
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
        raise NotImplementedError
