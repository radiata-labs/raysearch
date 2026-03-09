from __future__ import annotations

from typing import Any

from serpsage.components.extract.base import ExtractorBase


def build_extractor(*, rt: Any) -> ExtractorBase:
    return rt.components.resolve_default("extract", expected_type=ExtractorBase)


__all__ = ["ExtractorBase", "build_extractor"]
