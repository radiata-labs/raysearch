from __future__ import annotations

from serpsage.models.base import MutableModel


class BaseStepContext(MutableModel):
    request_id: str = ""
