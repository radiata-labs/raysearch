from __future__ import annotations

from typing import Any

from pydantic import Field

from serpsage.core.model_base import FrozenModel


class LLMUsage(FrozenModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatJSONResult(FrozenModel):
    data: dict[str, Any] = Field(default_factory=dict)
    usage: LLMUsage = Field(default_factory=LLMUsage)


__all__ = ["ChatJSONResult", "LLMUsage"]
