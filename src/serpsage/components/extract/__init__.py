from __future__ import annotations

from typing import Any, cast

from serpsage.components.extract.base import ExtractorBase


def build_extractor(*, rt: Any) -> ExtractorBase:
    return cast("ExtractorBase", rt.services.require(ExtractorBase))


__all__ = ["ExtractorBase", "build_extractor"]
