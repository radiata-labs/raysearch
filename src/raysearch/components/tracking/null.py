from __future__ import annotations

from typing_extensions import override

from raysearch.components.base import ComponentConfigBase
from raysearch.components.tracking.base import TrackingSinkBase
from raysearch.models.components.tracking import EventEnvelope


class NullTrackingSinkConfig(ComponentConfigBase):
    __setting_family__ = "tracking"
    __setting_name__ = "null"


class NullTrackingSink(TrackingSinkBase[NullTrackingSinkConfig]):
    """No-op tracking sink that discards all events."""

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        _ = event


__all__ = ["NullTrackingSink", "NullTrackingSinkConfig"]
