from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ChatJSONResult:
    data: dict[str, Any]
    usage: LLMUsage = LLMUsage()


__all__ = ["ChatJSONResult", "LLMUsage"]
