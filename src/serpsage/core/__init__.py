from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.core.workunit import WorkUnit
    from serpsage.models.base import FrozenModel, MutableModel

__all__ = [
    "Runtime",
    "FrozenModel",
    "MutableModel",
    "WorkUnit",
]


def __getattr__(name: str) -> Any:
    if name in {"FrozenModel", "MutableModel"}:
        from serpsage.models.base import FrozenModel, MutableModel

        return {"FrozenModel": FrozenModel, "MutableModel": MutableModel}[name]
    if name == "Runtime":
        from serpsage.core.runtime import Runtime

        return Runtime
    if name == "WorkUnit":
        from serpsage.core.workunit import WorkUnit

        return WorkUnit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
