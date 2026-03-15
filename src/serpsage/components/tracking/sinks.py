from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing_extensions import override

import anyio
from pydantic import field_validator

from serpsage.components.base import ComponentConfigBase
from serpsage.components.tracking.base import TrackingSinkBase
from serpsage.models.components.tracking import EventEnvelope


class NullTrackingSinkConfig(ComponentConfigBase):
    __setting_family__ = "tracking"
    __setting_name__ = "null_sink"


class JsonlTrackingSinkConfig(ComponentConfigBase):
    __setting_family__ = "tracking"
    __setting_name__ = "jsonl_sink"

    jsonl_path: str = ".serpsage_tracking.jsonl"

    @field_validator("jsonl_path")
    @classmethod
    def _validate_jsonl_path(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("tracking jsonl_path must be non-empty")
        return token


class NullTrackingSink(TrackingSinkBase[NullTrackingSinkConfig]):
    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        _ = event


class JsonlTrackingSink(TrackingSinkBase[JsonlTrackingSinkConfig]):
    def __init__(self) -> None:
        self._file_path = str(self.config.jsonl_path).strip()
        self._file: Any | None = None
        self._lock = anyio.Lock()

    @override
    async def on_init(self) -> None:
        if self._file is not None:
            return
        path = Path(self._file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = await anyio.open_file(path, mode="a", encoding="utf-8")

    @override
    async def emit(self, *, event: EventEnvelope) -> None:
        if self._file is None:
            return
        line = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            await self._file.write(line + "\n")
            await self._file.flush()

    @override
    async def on_close(self) -> None:
        if self._file is None:
            return
        await self._file.aclose()
        self._file = None


__all__ = [
    "JsonlTrackingSink",
    "JsonlTrackingSinkConfig",
    "NullTrackingSink",
    "NullTrackingSinkConfig",
]
