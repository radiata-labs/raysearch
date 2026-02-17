from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.models.llm import ChatResult


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

    @override
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        route = self._routes.get(str(model))
        if route is None:
            raise ValueError(f"llm model route `{model}` is not configured")
        client, provider_model = route
        return await client.chat(
            model=provider_model,
            messages=messages,
            schema=schema,
            timeout_s=timeout_s,
        )


__all__ = ["RoutedLLMClient"]
