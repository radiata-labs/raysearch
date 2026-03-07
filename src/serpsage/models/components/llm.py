from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from serpsage.models.base import FrozenModel

TModel = TypeVar("TModel", bound=BaseModel)


class LLMUsage(FrozenModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResultBase(FrozenModel):
    text: str = ""
    data: Any | None = None
    usage: LLMUsage = Field(default_factory=LLMUsage)


class ChatTextResult(ChatResultBase):
    data: None = None


class ChatDictResult(ChatResultBase):
    data: dict[str, Any] = Field(default_factory=dict)


class ChatModelResult(ChatResultBase, Generic[TModel]):
    data: TModel = Field(default_factory=lambda: None)  # type: ignore


__all__ = [
    "ChatDictResult",
    "ChatModelResult",
    "ChatResultBase",
    "ChatTextResult",
    "LLMUsage",
]
