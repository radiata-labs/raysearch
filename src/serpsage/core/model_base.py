from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


__all__ = ["FrozenModel", "MutableModel"]
