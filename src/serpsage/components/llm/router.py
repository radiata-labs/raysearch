from __future__ import annotations

from contextlib import suppress
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
from serpsage.models.telemetry import EventStatus, MeterPayload

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
        route = self._routes.get(str(model))
        if route is None:
            raise ValueError(f"llm model route `{model}` is not configured")
        client, provider_model = route
        started_ms = int(self.clock.now_ms())
        await self._emit_safe(
            event_name="llm.call",
            status="start",
            component="llm_router",
            stage="chat",
            attrs={
                "route_model": str(model),
                "provider_model": str(provider_model),
                "provider_client": type(client).__name__,
                "message_count": len(messages),
            },
        )
        try:
            result = await client.chat(
                model=provider_model,
                messages=messages,
                response_format=response_format,
                timeout_s=timeout_s,
            )
            usage = result.usage
            await self._emit_safe(
                event_name="llm.result",
                status="ok",
                component="llm_router",
                stage="chat",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                attrs={
                    "route_model": str(model),
                    "provider_model": str(provider_model),
                    "provider_client": type(client).__name__,
                },
            )
            await self._emit_safe(
                event_name="meter.usage.llm_tokens",
                status="ok",
                component="llm_router",
                stage="chat",
                idempotency_key=(
                    f"{model}:{provider_model}:{started_ms}:{usage.total_tokens}"
                ),
                attrs={
                    "route_model": str(model),
                    "provider_model": str(provider_model),
                    "provider_client": type(client).__name__,
                },
                meter=MeterPayload(
                    meter_type="llm_tokens",
                    unit="token",
                    quantity=float(usage.total_tokens),
                    provider=type(client).__name__,
                    model=str(provider_model),
                    prompt_tokens=int(usage.prompt_tokens),
                    completion_tokens=int(usage.completion_tokens),
                    total_tokens=int(usage.total_tokens),
                ),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            await self._emit_safe(
                event_name="llm.error",
                status="error",
                component="llm_router",
                stage="chat",
                duration_ms=max(0, int(self.clock.now_ms()) - started_ms),
                error_type=type(exc).__name__,
                attrs={
                    "route_model": str(model),
                    "provider_model": str(provider_model),
                    "provider_client": type(client).__name__,
                },
            )
            raise

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
