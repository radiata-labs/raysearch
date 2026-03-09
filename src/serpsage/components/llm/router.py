from __future__ import annotations

from contextlib import suppress
from typing import Any, TypeVar, cast, overload
from typing_extensions import override

from pydantic import BaseModel

from serpsage.components.base import ComponentMeta
from serpsage.components.llm.base import (
    LLMClientBase,
    LLMModelConfig,
    LLMRouterConfig,
)
from serpsage.components.registry import register_component
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject
from serpsage.models.components.llm import (
    ChatDictResult,
    ChatModelResult,
    ChatResultBase,
    ChatTextResult,
)
from serpsage.models.components.telemetry import EventStatus, MeterPayload

TModel = TypeVar("TModel", bound=BaseModel)
_ROUTER_META = ComponentMeta(
    family="llm",
    name="router",
    version="1.0.0",
    summary="Named model router over llm.route instances.",
    provides=("llm.client",),
    config_model=LLMRouterConfig,
)


@register_component(meta=_ROUTER_META)
class RoutedLLMClient(LLMClientBase[LLMRouterConfig]):
    meta = _ROUTER_META

    def __init__(
        self,
        *,
        rt: Runtime = Inject(),
        config: LLMRouterConfig = Inject(),
        routes: tuple[LLMClientBase[Any], ...] = Inject(),
    ) -> None:
        super().__init__(rt=rt, config=config)
        route_clients = routes
        self._routes: dict[str, tuple[LLMClientBase[Any], str]] = {}
        for client in route_clients:
            model_cfg = client.describe_model(client.config.name)
            if not model_cfg.api_key:
                continue
            self._routes[model_cfg.name] = (client, model_cfg.model)

    @override
    def describe_model(self, name: str) -> LLMModelConfig:
        route = self._routes.get(str(name))
        if route is None:
            super().describe_model(name)
            raise AssertionError("unreachable")
        client, _provider_model = route
        return client.describe_model(name)

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
        route = self._routes.get(route_key)
        if route is None:
            raise RuntimeError(f"llm model route `{model}` is not configured")
        client, provider_model = route
        provider_client = type(client).__name__
        route_attrs = self._route_attrs(
            route_model=route_key,
            provider_model=provider_model,
            provider_client=provider_client,
        )
        started_ms = int(self.clock.now_ms())
        await self._emit_safe(
            event_name="llm.call",
            status="start",
            component="llm_router",
            stage="chat",
            attrs={**route_attrs, "message_count": len(messages)},
        )
        try:
            result: ChatResultBase
            if response_format is None:
                result = await client.create(
                    model=provider_model,
                    messages=messages,
                    response_format=None,
                    format_override=None,
                    timeout_s=timeout_s,
                    **kwargs,
                )
            elif isinstance(response_format, dict):
                result = await client.create(
                    model=provider_model,
                    messages=messages,
                    response_format=response_format,
                    format_override=None,
                    timeout_s=timeout_s,
                    **kwargs,
                )
            elif isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                result = await client.create(
                    model=provider_model,
                    messages=messages,
                    response_format=response_format,
                    format_override=format_override,
                    timeout_s=timeout_s,
                    **kwargs,
                )
            else:
                raise TypeError(
                    "response_format must be dict[str, object] | type[BaseModel] | None"
                )
            usage = result.usage
            await self._emit_safe(
                event_name="llm.result",
                status="ok",
                component="llm_router",
                stage="chat",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs=route_attrs,
            )
            await self._emit_safe(
                event_name="meter.usage.llm_tokens",
                status="ok",
                component="llm_router",
                stage="chat",
                idempotency_key=f"{route_key}:{provider_model}:{started_ms}:{usage.total_tokens}",
                attrs=route_attrs,
                meter=MeterPayload(
                    meter_type="llm_tokens",
                    unit="token",
                    quantity=float(usage.total_tokens),
                    provider=provider_client,
                    model=str(provider_model),
                    prompt_tokens=int(usage.prompt_tokens),
                    completion_tokens=int(usage.completion_tokens),
                    total_tokens=int(usage.total_tokens),
                ),
            )
            return result
        except Exception as exc:
            await self._emit_safe(
                event_name="llm.error",
                status="error",
                component="llm_router",
                stage="chat",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs=route_attrs,
            )
            raise

    @staticmethod
    def _route_attrs(
        *,
        route_model: str,
        provider_model: str,
        provider_client: str,
    ) -> dict[str, object]:
        return {
            "route_model": str(route_model),
            "provider_model": str(provider_model),
            "provider_client": str(provider_client),
        }

    async def _emit_safe(
        self,
        *,
        event_name: str,
        status: EventStatus = "ok",
        request_id: str = "",
        trace_id: str = "",
        span_id: str = "",
        component: str = "",
        stage: str = "",
        duration_ms: int | None = None,
        error_code: str = "",
        error_type: str = "",
        attrs: dict[str, object] | None = None,
        meter: MeterPayload | None = None,
        idempotency_key: str = "",
    ) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(
                event_name=event_name,
                status=status,
                request_id=request_id,
                trace_id=trace_id,
                span_id=span_id,
                component=component,
                stage=stage,
                duration_ms=duration_ms,
                error_code=error_code,
                error_type=error_type,
                attrs=attrs,
                meter=meter,
                idempotency_key=idempotency_key,
            )


__all__ = ["RoutedLLMClient"]
