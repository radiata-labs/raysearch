from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload
from typing_extensions import override

from pydantic import BaseModel

from serpsage.components.llm.base import LLMClientBase
from serpsage.models.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

TModel = TypeVar("TModel", bound=BaseModel)


class NullLLMClient(LLMClientBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        timeout_s: float | None = None,
    ) -> ChatTextResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        timeout_s: float | None = None,
    ) -> ChatDictResult: ...

    @overload
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        timeout_s: float | None = None,
    ) -> ChatModelResult[TModel]: ...

    @override
    async def _chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        _ = model, messages, response_format, timeout_s
        raise RuntimeError("LLM is not configured (missing api_key or disabled).")


__all__ = ["NullLLMClient"]
