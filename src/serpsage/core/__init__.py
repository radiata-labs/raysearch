from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serpsage.core.overrides import Overrides
    from serpsage.core.workunit import ClockBase, WorkUnit
    from serpsage.models.base import FrozenModel, MutableModel

__all__ = [
    "ClockBase",
    "Overrides",
    "FrozenModel",
    "MutableModel",
    "WorkUnit",
]


def __getattr__(name: str) -> Any:
    if name in {"FrozenModel", "MutableModel"}:
        from serpsage.models.base import FrozenModel, MutableModel

        return {"FrozenModel": FrozenModel, "MutableModel": MutableModel}[name]
    if name == "ClockBase":
        from serpsage.core.workunit import ClockBase

        return ClockBase
    if name == "Overrides":
        from serpsage.core.overrides import Overrides

        return Overrides
    if name == "WorkUnit":
        from serpsage.core.workunit import WorkUnit

        return WorkUnit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
