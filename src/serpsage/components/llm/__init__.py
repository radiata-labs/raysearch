from __future__ import annotations

from typing import Any, cast

from serpsage.components.llm.base import LLMClientBase


def build_overview_client(*, rt: Any, http: Any | None = None) -> LLMClientBase:
    _ = http
    return cast("LLMClientBase", rt.services.require(LLMClientBase))


__all__ = ["LLMClientBase", "build_overview_client"]
