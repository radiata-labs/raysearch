from __future__ import annotations

from typing import Any, TypeVar, overload
from typing_extensions import override

from pydantic import BaseModel

from serpsage.components.base import (
    ComponentConfigBase,
)
from serpsage.components.llm.base import (
    LLMClientBase,
    LLMConfig,
)
from serpsage.components.loads import ComponentRegistry
from serpsage.dependencies import CACHE_TOKEN, Depends, solve_dependencies
from serpsage.models.components.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)


async def llm_routes_factory(
    cache: dict[Any, Any] = Depends(CACHE_TOKEN),
    registry: ComponentRegistry = Depends(),
) -> tuple[LLMClientBase[Any], ...]:
    """Factory function: collect all enabled LLM routes (excluding RoutedLLMClient)."""
    routes: list[LLMClientBase[Any]] = []
    for spec in registry.enabled_specs("llm"):
        if spec.cls.__name__ == "RoutedLLMClient":
            continue
        if not issubclass(spec.cls, LLMClientBase):
            continue
        instance = await solve_dependencies(spec.cls, dependency_cache=cache)
        if isinstance(instance, LLMClientBase):
            routes.append(instance)
    return tuple(routes)


class LLMRouterConfig(ComponentConfigBase):
    __setting_family__ = "llm"
    __setting_name__ = "router"


TModel = TypeVar("TModel", bound=BaseModel)


class RoutedLLMClient(LLMClientBase[LLMRouterConfig]):
    def __init__(
        self,
        *,
        routes: tuple[LLMClientBase[Any], ...] = Depends(llm_routes_factory),
    ) -> None:
        self.routes: dict[str, tuple[LLMClientBase[LLMConfig], str]] = {}
        for client in routes:
            for model_cfg in client.configured_models():
                if not model_cfg.api_key:
                    continue
                route_name = str(model_cfg.name).strip()
                if not route_name:
                    raise ValueError(
                        f"{type(client).__name__} contains a model route without a name"
                    )
                if route_name in self.routes:
                    raise ValueError(f"duplicate llm route `{route_name}`")
                self.routes[route_name] = (client, str(model_cfg.model))

    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: None = None,
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatTextResult: ...

    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        format_override: None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatDictResult: ...

    @overload
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: type[TModel],
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatModelResult[TModel]: ...

    @override
    async def _create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, object] | type[BaseModel] | None = None,
        format_override: dict[str, object] | None = None,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> ChatResultBase:
        route_key = str(model)
        route = self.routes.get(route_key)
        if route is None:
            raise RuntimeError(f"llm model route `{model}` is not configured")
        client, provider_model = route
        if response_format is None:
            return await client._create(
                model=provider_model,
                messages=messages,
                response_format=None,
                format_override=None,
                timeout_s=timeout_s,
                _route_name=route_key,
                **kwargs,
            )
        if isinstance(response_format, dict):
            return await client._create(
                model=provider_model,
                messages=messages,
                response_format=response_format,
                format_override=None,
                timeout_s=timeout_s,
                _route_name=route_key,
                **kwargs,
            )
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return await client._create(
                model=provider_model,
                messages=messages,
                response_format=response_format,
                format_override=format_override,
                timeout_s=timeout_s,
                _route_name=route_key,
                **kwargs,
            )
        raise TypeError(
            "response_format must be dict[str, object] | type[BaseModel] | None"
        )


__all__ = ["RoutedLLMClient", "llm_routes_factory"]
