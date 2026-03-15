from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing_extensions import override

import anyio
from anyio.abc import TaskGroup
from pydantic import Field, field_validator

from serpsage.components.base import ComponentConfigBase
from serpsage.components.metering.base import MeteringSinkBase
from serpsage.models.components.metering import MeterRecord


class JsonlMeteringSinkConfig(ComponentConfigBase):
    __setting_family__ = "metering"
    __setting_name__ = "jsonl"

    jsonl_path: str = ".serpsage_metering.jsonl"
    batch_size: int = Field(default=100, ge=1, le=10000)
    flush_interval_ms: int = Field(default=1000, ge=100, le=60000)

    @field_validator("jsonl_path")
    @classmethod
    def _validate_jsonl_path(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("metering jsonl_path must be non-empty")
        return token


class JsonlMeteringSink(MeteringSinkBase[JsonlMeteringSinkConfig]):
    """Metering sink that writes records to a JSONL file with batched writes."""

    def __init__(self) -> None:
        self._file_path = str(self.config.jsonl_path).strip()
        self._batch_size = int(self.config.batch_size)
        self._flush_interval_s = int(self.config.flush_interval_ms) / 1000.0
        self._file: Any | None = None
        self._buffer: list[str] = []
        self._lock = anyio.Lock()
        self._flush_event = anyio.Event()
        self._tg_cm: Any | None = None
        self._tg: TaskGroup | None = None

    @override
    async def on_init(self) -> None:
        if self._file is not None:
            return
        path = Path(self._file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = await anyio.open_file(path, mode="a", encoding="utf-8")
        self._tg_cm = anyio.create_task_group()
        tg = await self._tg_cm.__aenter__()
        self._tg = tg
        tg.start_soon(self._flush_loop)

    @override
    async def emit(self, *, record: MeterRecord) -> None:
        if self._file is None:
            return
        line = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        should_flush = False
        async with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._batch_size:
                should_flush = True
        if should_flush:
            self._flush_event.set()

    async def _flush_loop(self) -> None:
        while self._file is not None:
            try:
                with anyio.move_on_after(self._flush_interval_s):
                    await self._flush_event.wait()
            except Exception:  # noqa: BLE001
                pass
            self._flush_event = anyio.Event()
            await self._do_flush()

    async def _do_flush(self) -> None:
        if self._file is None:
            return
        async with self._lock:
            if not self._buffer:
                return
            lines = self._buffer
            self._buffer = []
        content = "\n".join(lines) + "\n"
        await self._file.write(content)
        await self._file.flush()

    @override
    async def on_close(self) -> None:
        # Final flush before stopping the loop
        await self._do_flush()
        # Signal the flush loop to exit by setting file to None first
        # This MUST happen before waiting for the task group, otherwise deadlock
        file = self._file
        self._file = None
        self._tg = None
        tg_cm = self._tg_cm
        self._tg_cm = None
        # Now safe to wait for the task group to finish
        if tg_cm is not None:
            await tg_cm.__aexit__(None, None, None)
        # Finally close the file handle
        if file is not None:
            await file.aclose()


__all__ = ["JsonlMeteringSink", "JsonlMeteringSinkConfig"]
