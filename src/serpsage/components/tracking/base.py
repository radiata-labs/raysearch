from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Generic, TypeVar

from serpsage.components.base import ComponentBase, ComponentConfigBase
from serpsage.models.components.tracking import (
    DebugEvent,
    ErrorEvent,
    EventEnvelope,
    InfoEvent,
    WarningEvent,
    sanitize_tracking_data,
)

TrackingSinkConfigT = TypeVar("TrackingSinkConfigT", bound=ComponentConfigBase)
TrackingEmitterConfigT = TypeVar("TrackingEmitterConfigT", bound=ComponentConfigBase)


class TrackingSinkBase(
    ComponentBase[TrackingSinkConfigT],
    ABC,
    Generic[TrackingSinkConfigT],
):
    @abstractmethod
    async def emit(self, *, event: EventEnvelope) -> None:
        raise NotImplementedError


class TrackingEmitterBase(
    ComponentBase[TrackingEmitterConfigT],
    ABC,
    Generic[TrackingEmitterConfigT],
):
    @abstractmethod
    async def emit_event(self, *, event: EventEnvelope) -> None:
        raise NotImplementedError

    @abstractmethod
    def push_request_context(self, *, request_id: str) -> object:
        raise NotImplementedError

    @abstractmethod
    def pop_request_context(self, token: object) -> None:
        raise NotImplementedError

    async def debug(
        self,
        *,
        name: str,
        request_id: str = "",
        step: str = "",
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with suppress(Exception):
            await self.emit_event(
                event=DebugEvent(
                    id=uuid.uuid4().hex,
                    ts_ms=int(self.clock.now_ms()),
                    name=str(name),
                    request_id=str(request_id or ""),
                    step=str(step or ""),
                    duration_ms=duration_ms,
                    data=sanitize_tracking_data(data),
                )
            )

    async def info(
        self,
        *,
        name: str,
        request_id: str = "",
        step: str = "",
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with suppress(Exception):
            await self.emit_event(
                event=InfoEvent(
                    id=uuid.uuid4().hex,
                    ts_ms=int(self.clock.now_ms()),
                    name=str(name),
                    request_id=str(request_id or ""),
                    step=str(step or ""),
                    duration_ms=duration_ms,
                    data=sanitize_tracking_data(data),
                )
            )

    async def warning(
        self,
        *,
        name: str,
        request_id: str = "",
        step: str = "",
        duration_ms: int | None = None,
        warning_code: str = "",
        warning_message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        extracted, attrs = self._extract_event_fields(
            data,
            field_names=("warning_code", "warning_message"),
        )
        with suppress(Exception):
            await self.emit_event(
                event=WarningEvent(
                    id=uuid.uuid4().hex,
                    ts_ms=int(self.clock.now_ms()),
                    name=str(name),
                    request_id=str(request_id or ""),
                    step=str(step or ""),
                    duration_ms=duration_ms,
                    warning_code=str(warning_code or extracted["warning_code"]),
                    warning_message=str(
                        warning_message or extracted["warning_message"]
                    ),
                    data=attrs,
                )
            )

    async def error(
        self,
        *,
        name: str,
        request_id: str = "",
        step: str = "",
        duration_ms: int | None = None,
        error_code: str = "",
        error_type: str = "",
        error_message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        extracted, attrs = self._extract_event_fields(
            data,
            field_names=("error_code", "error_type", "error_message"),
        )
        with suppress(Exception):
            await self.emit_event(
                event=ErrorEvent(
                    id=uuid.uuid4().hex,
                    ts_ms=int(self.clock.now_ms()),
                    name=str(name),
                    request_id=str(request_id or ""),
                    step=str(step or ""),
                    duration_ms=duration_ms,
                    error_code=str(error_code or extracted["error_code"]),
                    error_type=str(error_type or extracted["error_type"]),
                    error_message=str(error_message or extracted["error_message"]),
                    data=attrs,
                )
            )

    @staticmethod
    def _extract_event_fields(
        data: dict[str, Any] | None,
        *,
        field_names: tuple[str, ...],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        raw = dict(data or {})
        extracted: dict[str, str] = {}
        for field_name in field_names:
            extracted[field_name] = str(raw.pop(field_name, "") or "").strip()
        return extracted, sanitize_tracking_data(raw)


__all__ = ["TrackingEmitterBase", "TrackingSinkBase"]
