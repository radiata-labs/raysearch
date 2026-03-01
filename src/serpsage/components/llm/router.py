from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, overload
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


class RoutedLLMClient(LLMClientBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        routes: dict[str, tuple[LLMClientBase, str]],
    ) -> None:
        super().__init__(rt=rt)
        self._routes = dict(routes)
        self.bind_deps(*[unit for unit, _ in self._routes.values()])

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

    @override
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResultBase:
        route = self._routes.get(str(model))
        if route is None:
            raise ValueError(f"llm model route `{model}` is not configured")
        client, provider_model = route
        return await client.chat(
            model=provider_model,
            messages=messages,
            response_format=response_format,
            timeout_s=timeout_s,
        )


__all__ = ["RoutedLLMClient"]
