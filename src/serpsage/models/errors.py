from __future__ import annotations

from typing import Any

from pydantic import Field

from serpsage.core.model_base import MutableModel


class AppError(MutableModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AppError"]
