from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

import anyio

from serpsage.dependencies import Depends

if TYPE_CHECKING:
    from types import TracebackType

    from serpsage.components.loads import ComponentRegistry
    from serpsage.components.telemetry import TelemetryEmitterBase
    from serpsage.core.runtime import ClockBase, Runtime
    from serpsage.settings.models import AppSettings


class WorkUnit:
    rt: Runtime = Depends()

    _wu_bootstrapped: bool
    _wu_deps: list[WorkUnit]
    _wu_dep_ids: set[int]
    _wu_op_lock: anyio.Lock
    _wu_node_lock: anyio.Lock
    _wu_initialized: bool
    _wu_closed: bool
    _wu_init_order: list[WorkUnit] | None

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        super().__init_subclass__()
        forbidden = [name for name in ("ainit", "aclose") if name in cls.__dict__]
        if forbidden:
            names = ", ".join(forbidden)
            raise TypeError(
                f"{cls.__name__} must not override {names}; "
                "override on_init/on_close instead"
            )

    def _require_bootstrapped(self) -> None:
        if not bool(getattr(self, "_wu_bootstrapped", False)):
            raise RuntimeError(
                f"{type(self).__name__} is not bootstrapped; construct it through the dependency solver"
            )

    def _wu_bootstrap(self, rt: Runtime | object) -> None:
        if bool(getattr(self, "_wu_bootstrapped", False)):
            raise RuntimeError(f"{type(self).__name__} is already bootstrapped")
        self.rt = rt  # type: ignore[assignment]
        self._wu_bootstrapped = True
        self._wu_deps = []
        self._wu_dep_ids = set()
        self._wu_op_lock = anyio.Lock()
        self._wu_node_lock = anyio.Lock()
        self._wu_initialized = False
        self._wu_closed = False
        self._wu_init_order = None

    def _wu_bind_injected(self, *values: object) -> None:
        self._require_bootstrapped()
        deps = list(self._wu_collect_injected(values))
        if deps:
            self.bind_deps(*deps)

    def _wu_collect_injected(
        self,
        values: tuple[object, ...],
    ) -> tuple[WorkUnit, ...]:
        seen: set[int] = set()
        found: list[WorkUnit] = []
        for value in values:
            self._wu_collect_injected_one(value, seen=seen, found=found)
        return tuple(found)

    def _wu_collect_injected_one(
        self,
        value: object,
        *,
        seen: set[int],
        found: list[WorkUnit],
    ) -> None:
        if value is None:
            return
        if isinstance(value, WorkUnit):
            if value is self:
                return
            dep_id = id(value)
            if dep_id in seen:
                return
            seen.add(dep_id)
            found.append(value)
            return
        value_id = id(value)
        if value_id in seen:
            return
        if isinstance(value, Mapping):
            seen.add(value_id)
            for item in value.values():
                self._wu_collect_injected_one(item, seen=seen, found=found)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            seen.add(value_id)
            for item in value:
                self._wu_collect_injected_one(item, seen=seen, found=found)

    @property
    def settings(self) -> AppSettings:
        return self.rt.settings

    @property
    def clock(self) -> ClockBase:
        return self.rt.clock

    @property
    def telemetry(self) -> TelemetryEmitterBase[Any] | None:
        return self.rt.telemetry

    @property
    def components(self) -> ComponentRegistry:
        container = self.rt.components
        if container is None:
            raise RuntimeError("component registry is not attached to the runtime")
        return container

    def bind_deps(self, *deps: WorkUnit | None) -> None:
        self._require_bootstrapped()

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
        self._require_bootstrapped()
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
        self._require_bootstrapped()
        async with self._wu_node_lock:
            if self._wu_closed:
                raise RuntimeError(f"{type(self).__name__} is already closed")
            if self._wu_initialized:
                return False
            await self.on_init()
            self._wu_initialized = True
            return True

    async def _wu_close_self(self) -> bool:
        self._require_bootstrapped()
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
        self._require_bootstrapped()
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
        self._require_bootstrapped()
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
