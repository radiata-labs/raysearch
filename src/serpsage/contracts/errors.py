from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AppError(BaseModel):
    """A serializable error that does not necessarily abort the whole request."""

    model_config = ConfigDict(validate_assignment=True)

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = ["AppError"]

