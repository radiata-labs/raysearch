from __future__ import annotations

from typing_extensions import override

from serpsage.components.base import ComponentConfigBase
from serpsage.components.metering.base import MeteringSinkBase
from serpsage.models.components.metering import MeterRecord


class NullMeteringSinkConfig(ComponentConfigBase):
    __setting_family__ = "metering"
    __setting_name__ = "null"


class NullMeteringSink(MeteringSinkBase[NullMeteringSinkConfig]):
    """No-op metering sink that discards all records."""

    @override
    async def emit(self, *, record: MeterRecord) -> None:
        _ = record


__all__ = ["NullMeteringSink", "NullMeteringSinkConfig"]
