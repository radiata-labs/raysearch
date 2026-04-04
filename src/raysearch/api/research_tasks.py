from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass

from raysearch.app.engine import Engine
from raysearch.models.app.request import ResearchRequest
from raysearch.models.app.response import (
    ResearchResponse,
    ResearchTaskListResponse,
    ResearchTaskResponse,
    ResearchTaskStatus,
)


@dataclass(slots=True)
class _ResearchTaskEntry:
    research_id: str
    create_at: int
    request: ResearchRequest
    status: ResearchTaskStatus = "pending"
    output: ResearchResponse | None = None
    finished_at: int | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = None


class ResearchTaskManager:
    def __init__(self) -> None:
        self._entries: dict[str, _ResearchTaskEntry] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        payload: ResearchRequest,
        engine: Engine,
    ) -> ResearchTaskResponse:
        research_id = uuid.uuid4().hex
        entry = _ResearchTaskEntry(
            research_id=research_id,
            create_at=self._now_ms(),
            request=payload.model_copy(deep=True),
        )
        async with self._lock:
            self._entries[research_id] = entry
            task = asyncio.create_task(
                self._run_task(research_id=research_id, engine=engine),
                name=f"research-task:{research_id}",
            )
            task.add_done_callback(self._drain_task_exception)
            entry.task = task
            return self._snapshot(entry)

    async def get_task(self, research_id: str) -> ResearchTaskResponse:
        async with self._lock:
            entry = self._entries.get(research_id)
            if entry is None:
                raise KeyError(research_id)
            return self._snapshot(entry)

    async def cancel_task(self, research_id: str) -> ResearchTaskResponse:
        async with self._lock:
            entry = self._entries.get(research_id)
            if entry is None:
                raise KeyError(research_id)
            if entry.status in {"completed", "canceled", "failed"}:
                return self._snapshot(entry)
            entry.status = "canceled"
            entry.finished_at = self._now_ms()
            entry.output = None
            entry.error = None
            task = entry.task
            snapshot = self._snapshot(entry)
        if task is not None:
            task.cancel()
        return snapshot

    async def list_tasks(
        self,
        *,
        cursor: str | None,
        limit: int,
    ) -> ResearchTaskListResponse:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        offset = self._parse_cursor(cursor)
        async with self._lock:
            ordered_entries = sorted(
                self._entries.values(),
                key=lambda item: (item.create_at, item.research_id),
                reverse=True,
            )
            page = ordered_entries[offset : offset + limit]
            next_offset = offset + len(page)
            has_more = next_offset < len(ordered_entries)
            return ResearchTaskListResponse(
                data=[self._snapshot(item) for item in page],
                has_more=has_more,
                next_cursor=str(next_offset) if has_more else "",
            )

    async def close(self) -> None:
        async with self._lock:
            tasks = [
                entry.task
                for entry in self._entries.values()
                if entry.task is not None and not entry.task.done()
            ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_task(
        self,
        *,
        research_id: str,
        engine: Engine,
    ) -> None:
        async with self._lock:
            entry = self._entries.get(research_id)
            if entry is None:
                return
            if entry.status == "canceled":
                if entry.finished_at is None:
                    entry.finished_at = self._now_ms()
                return
            entry.status = "running"
            request = entry.request.model_copy(deep=True)
        try:
            output = await engine.research(request)
        except asyncio.CancelledError:
            async with self._lock:
                entry = self._entries.get(research_id)
                if entry is not None and entry.status != "completed":
                    entry.status = "canceled"
                    entry.finished_at = entry.finished_at or self._now_ms()
                    entry.output = None
                    entry.error = None
            raise
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                entry = self._entries.get(research_id)
                if entry is not None and entry.status != "canceled":
                    entry.status = "failed"
                    entry.finished_at = self._now_ms()
                    entry.output = None
                    entry.error = str(exc) or type(exc).__name__
        else:
            async with self._lock:
                entry = self._entries.get(research_id)
                if entry is not None and entry.status != "canceled":
                    entry.status = "completed"
                    entry.finished_at = self._now_ms()
                    entry.output = output.model_copy(deep=True)
                    entry.error = None
        finally:
            async with self._lock:
                entry = self._entries.get(research_id)
                current_task = asyncio.current_task()
                if entry is not None and entry.task is current_task:
                    entry.task = None

    @staticmethod
    def _drain_task_exception(task: asyncio.Task[None]) -> None:
        with suppress(asyncio.CancelledError):
            task.exception()

    @staticmethod
    def _now_ms() -> int:
        return time.time_ns() // 1_000_000

    @staticmethod
    def _parse_cursor(cursor: str | None) -> int:
        token = str(cursor or "").strip()
        if not token:
            return 0
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("cursor must be a non-negative integer string") from exc
        if value < 0:
            raise ValueError("cursor must be a non-negative integer string")
        return value

    @staticmethod
    def _snapshot(entry: _ResearchTaskEntry) -> ResearchTaskResponse:
        return ResearchTaskResponse(
            research_id=entry.research_id,
            create_at=entry.create_at,
            themes=entry.request.themes,
            search_mode=entry.request.search_mode,
            json_schema=entry.request.json_schema,
            status=entry.status,
            output=(
                entry.output.model_copy(deep=True) if entry.output is not None else None
            ),
            finished_at=entry.finished_at,
            error=entry.error,
        )


__all__ = ["ResearchTaskManager"]
