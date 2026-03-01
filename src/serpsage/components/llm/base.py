from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from serpsage.core.workunit import WorkUnit
from serpsage.models.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)

TModel = TypeVar("TModel", bound=BaseModel)


class LLMClientBase(WorkUnit, ABC):
    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
    ) -> ChatModelResult[TModel]: ...

    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        raise NotImplementedError
