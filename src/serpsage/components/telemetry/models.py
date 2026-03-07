from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_serializer

from serpsage.core.model_base import FrozenModel

EventStatus = Literal["start", "ok", "error"]
_MAX_ATTR_DEPTH = 5
_MAX_ATTR_ITEMS = 64
_MAX_ATTR_TEXT_LEN = 4000


def sanitize_attr_map(value: dict[str, Any] | None) -> dict[str, Any]:
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
        for idx, (k, item) in enumerate(value.items()):
            if idx >= _MAX_ATTR_ITEMS:
                break
            out_map[str(k)] = _sanitize_attr_value(item, depth=depth + 1)
        return out_map
    return str(value)[:_MAX_ATTR_TEXT_LEN]


class EventAttributes(FrozenModel):
    values: dict[str, Any] = Field(default_factory=dict)

    @field_validator("values", mode="before")
    @classmethod
    def _validate_values(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return {"value": _sanitize_attr_value(value, depth=0)}
        return sanitize_attr_map(value)

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return dict(self.values)


class MeterPayload(FrozenModel):
    meter_type: str
    unit: str
    quantity: float = 0.0
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EventEnvelope(FrozenModel):
    event_id: str
    idempotency_key: str = ""
    event_name: str
    status: EventStatus = "ok"
    ts_ms: int
    request_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    component: str = ""
    stage: str = ""
    duration_ms: int | None = None
    error_code: str = ""
    error_type: str = ""
    attrs: EventAttributes = Field(default_factory=EventAttributes)
    meter: MeterPayload | None = None


def is_critical_event(*, event_name: str, status: EventStatus) -> bool:
    if status == "error":
        return True
    return str(event_name).startswith("meter.")


__all__ = [
    "EventAttributes",
    "EventEnvelope",
    "EventStatus",
    "MeterPayload",
    "is_critical_event",
    "sanitize_attr_map",
]
