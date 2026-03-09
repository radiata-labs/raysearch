from __future__ import annotations

from typing import Any

from serpsage.components.llm.base import LLMClientBase


def build_overview_client(*, rt: Any, http: Any | None = None) -> LLMClientBase:
    _ = http
    return rt.components.resolve_default("llm", expected_type=LLMClientBase)


__all__ = ["LLMClientBase", "build_overview_client"]
