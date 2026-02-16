from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.models.llm import ChatResult


class LLMClientBase(WorkUnit, ABC):
    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError
