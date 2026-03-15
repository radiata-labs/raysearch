from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from serpsage.models.base import FrozenModel

MeterName = Literal["fetch.page", "llm.tokens", "request", "search.query"]
MeterUnit = Literal["call", "page", "request", "token"]


class TokenUsage(FrozenModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @field_validator(
        "prompt_tokens", "completion_tokens", "total_tokens", mode="before"
    )
    @classmethod
    def _normalize_count(cls, value: object) -> int:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        if isinstance(value, str):
            token = value.strip()
            return max(0, int(token or "0"))
        return 0


class MeterRecord(FrozenModel):
    id: str
    ts_ms: int
    name: MeterName
    unit: MeterUnit
    request_id: str = ""
    key: str = ""
    provider: str = ""
    model: str = ""
    tokens: TokenUsage | None = None

    @field_validator("request_id", "key", "provider", "model", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()


__all__ = ["MeterName", "MeterRecord", "MeterUnit", "TokenUsage"]
