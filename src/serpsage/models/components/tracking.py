from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from serpsage.models.base import FrozenModel

TrackingLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_MAX_ATTR_DEPTH = 5
_MAX_ATTR_ITEMS = 64
_MAX_ATTR_TEXT_LEN = 4000
_LEVEL_ORDER: dict[TrackingLevel, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
}


def normalize_tracking_level(value: object) -> TrackingLevel:
    level_name = str(value or "INFO").strip().upper()
    if level_name == "WARN":
        level_name = "WARNING"
    if level_name not in _LEVEL_ORDER:
        raise ValueError("tracking level must be one of DEBUG, INFO, WARNING, ERROR")
    return level_name  # type: ignore[return-value]


def should_emit_tracking(
    *, event_level: TrackingLevel, minimum_level: TrackingLevel
) -> bool:
    return _LEVEL_ORDER[event_level] >= _LEVEL_ORDER[minimum_level]


def sanitize_tracking_data(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    out: dict[str, Any] = {}
    count = 0
    for key, item in value.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = _sanitize_attr_value(item, depth=0)
        count += 1
        if count >= _MAX_ATTR_ITEMS:
            break
    return out


def _sanitize_attr_value(value: Any, *, depth: int) -> Any:
    if depth >= _MAX_ATTR_DEPTH:
        return str(value)[:_MAX_ATTR_TEXT_LEN]
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:_MAX_ATTR_TEXT_LEN]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")[:_MAX_ATTR_TEXT_LEN]
    if isinstance(value, list | tuple | set):
        return [
            _sanitize_attr_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_ATTR_ITEMS]
        ]
    if isinstance(value, dict):
        out_map: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _MAX_ATTR_ITEMS:
                break
            out_map[str(key)] = _sanitize_attr_value(item, depth=depth + 1)
        return out_map
    return str(value)[:_MAX_ATTR_TEXT_LEN]


class EventEnvelope(FrozenModel):
    id: str
    ts_ms: int
    level: TrackingLevel = "INFO"
    name: str
    request_id: str = ""
    step: str = ""
    duration_ms: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, value: object) -> TrackingLevel:
        return normalize_tracking_level(value)

    @field_validator("name", "request_id", "step", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _normalize_duration(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        if isinstance(value, str):
            token = value.strip()
            return max(0, int(token or "0"))
        return None

    @field_validator("data", mode="before")
    @classmethod
    def _normalize_data(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return {"value": _sanitize_attr_value(value, depth=0)}
        return sanitize_tracking_data(value)


class DebugEvent(EventEnvelope):
    level: TrackingLevel = "DEBUG"


class InfoEvent(EventEnvelope):
    level: TrackingLevel = "INFO"


class WarningEvent(EventEnvelope):
    level: TrackingLevel = "WARNING"
    warning_code: str = ""
    warning_message: str = ""

    @field_validator("warning_code", "warning_message", mode="before")
    @classmethod
    def _normalize_warning_text(cls, value: object) -> str:
        return str(value or "").strip()


class ErrorEvent(EventEnvelope):
    level: TrackingLevel = "ERROR"
    error_code: str = ""
    error_type: str = ""
    error_message: str = ""

    @field_validator(
        "error_code",
        "error_type",
        "error_message",
        mode="before",
    )
    @classmethod
    def _normalize_error_text(cls, value: object) -> str:
        return str(value or "").strip()


TrackingEvent = DebugEvent | InfoEvent | WarningEvent | ErrorEvent


__all__ = [
    "DebugEvent",
    "ErrorEvent",
    "EventEnvelope",
    "InfoEvent",
    "TrackingEvent",
    "TrackingLevel",
    "WarningEvent",
    "normalize_tracking_level",
    "sanitize_tracking_data",
    "should_emit_tracking",
]
