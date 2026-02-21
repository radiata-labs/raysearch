from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

import anyio

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import AppSettings
    from serpsage.telemetry.base import ClockBase, SpanBase, TelemetryBase


class WorkUnit:
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        forbidden = [name for name in ("ainit", "aclose") if name in cls.__dict__]
        if forbidden:
            names = ", ".join(forbidden)
            raise TypeError(
                f"{cls.__name__} must not override {names}; "
                "override on_ainit/on_aclose instead"
            )

    def __init__(self, *, rt: Runtime) -> None:
        self.rt = rt
        self._wu_bootstrapped = True
        self._wu_deps: list[WorkUnit] = []
        self._wu_dep_ids: set[int] = set()
        self._wu_op_lock = anyio.Lock()
        self._wu_node_lock = anyio.Lock()
        self._wu_initialized = False
        self._wu_closed = False
        self._wu_init_order: list[WorkUnit] | None = None

    @property
    def settings(self) -> AppSettings:
        return self.rt.settings

    @property
    def telemetry(self) -> TelemetryBase:
        return self.rt.telemetry

    @property
    def clock(self) -> ClockBase:
        return self.rt.clock

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[SpanBase]:
        sp = self.telemetry.start_span(name, **attrs)
        try:
            yield sp
        finally:
            sp.end()

    def bind_deps(self, *deps: WorkUnit | None) -> None:
        def _bind_dep_one(dep: WorkUnit | None) -> WorkUnit | None:
            if dep is None:
                return None
            if not isinstance(dep, WorkUnit):
                raise TypeError(
                    f"dependency must be WorkUnit, got: {type(dep).__name__}"
                )
            if dep is self:
                raise ValueError("workunit cannot depend on itself")
            dep_id = id(dep)
            if dep_id in self._wu_dep_ids:
                return dep
            self._wu_dep_ids.add(dep_id)
            self._wu_deps.append(dep)
            return dep

        for dep in deps:
            _bind_dep_one(dep)

    @property
    def dependencies(self) -> tuple[WorkUnit, ...]:
        return tuple(self._wu_deps)

    async def on_init(self) -> None:
        return

    async def on_close(self) -> None:
        return

    def _collect_post_order(self) -> list[WorkUnit]:
        visited: set[int] = set()
        visiting: set[int] = set()
        ordered: list[WorkUnit] = []

        def walk(node: WorkUnit) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            if node_id in visiting:
                raise RuntimeError(
                    f"dependency cycle detected at {type(node).__name__}"
                )
            visiting.add(node_id)
            for dep in node.dependencies:
                walk(dep)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node)

        walk(self)
        return ordered

    async def _wu_init_self(self) -> bool:
        async with self._wu_node_lock:
            if self._wu_closed:
                raise RuntimeError(f"{type(self).__name__} is already closed")
            if self._wu_initialized:
                return False
            await self.on_init()
            self._wu_initialized = True
            return True

    async def _wu_close_self(self) -> bool:
        async with self._wu_node_lock:
            if self._wu_closed:
                return False
            try:
                await self.on_close()
            finally:
                self._wu_closed = True
                self._wu_initialized = False
                self._wu_init_order = None
            return True

    async def ainit(self) -> None:
        async with self._wu_op_lock:
            if self._wu_closed:
                raise RuntimeError(f"{type(self).__name__} is already closed")
            if self._wu_initialized:
                return

            order = self._collect_post_order()
            initialized_now: list[WorkUnit] = []
            try:
                for node in order:
                    did_init = await node._wu_init_self()
                    if did_init:
                        initialized_now.append(node)
            except Exception as exc:
                rollback_errors: list[Exception] = []
                for node in reversed(initialized_now):
                    try:
                        await node._wu_close_self()
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_errors.append(
                            rollback_exc
                            if isinstance(rollback_exc, Exception)
                            else Exception(str(rollback_exc))
                        )
                if rollback_errors:
                    raise ExceptionGroup(
                        "workunit ainit failed and rollback had errors",
                        [
                            exc if isinstance(exc, Exception) else Exception(str(exc)),
                            *rollback_errors,
                        ],
                    ) from exc
                raise

            self._wu_init_order = order

    async def aclose(self) -> None:
        async with self._wu_op_lock:
            if self._wu_closed:
                return
            order = self._wu_init_order or self._collect_post_order()
            close_errors: list[Exception] = []
            for node in reversed(order):
                try:
                    await node._wu_close_self()
                except Exception as exc:  # noqa: BLE001
                    close_errors.append(
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
            self._wu_init_order = None
            if close_errors:
                raise ExceptionGroup("workunit aclose failed", close_errors)

    async def __aenter__(self) -> Self:
        await self.ainit()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["WorkUnit"]
