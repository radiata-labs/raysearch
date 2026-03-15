from __future__ import annotations

from typing_extensions import override

import structlog

from serpsage.components.base import ComponentConfigBase
from serpsage.components.tracking.base import TrackingSinkBase
from serpsage.models.components.tracking import EventEnvelope, TrackingLevel

_LEVEL_MAP: dict[TrackingLevel, str] = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
}


class LogTrackingSinkConfig(ComponentConfigBase):
    __setting_family__ = "tracking"
    __setting_name__ = "log"


class LogTrackingSink(TrackingSinkBase[LogTrackingSinkConfig]):
    """Tracking sink that outputs events via structlog."""

    def __init__(self) -> None:
        self._logger = structlog.get_logger()

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        level = _LEVEL_MAP.get(event.level, "info")
        log_method = getattr(self._logger, level, self._logger.info)

        log_method(
            event.name,
            id=event.id,
            ts_ms=event.ts_ms,
            request_id=event.request_id,
            step=event.step,
            duration_ms=event.duration_ms,
            **event.data,
        )


__all__ = ["LogTrackingSink", "LogTrackingSinkConfig"]
