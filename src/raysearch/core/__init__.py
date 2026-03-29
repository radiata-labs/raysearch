from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raysearch.core.overrides import Overrides
    from raysearch.core.workunit import ClockBase, WorkUnit
    from raysearch.models.base import FrozenModel, MutableModel

__all__ = [
    "ClockBase",
    "Overrides",
    "FrozenModel",
    "MutableModel",
    "WorkUnit",
]


def __getattr__(name: str) -> Any:
    if name in {"FrozenModel", "MutableModel"}:
        from raysearch.models.base import FrozenModel, MutableModel

        return {"FrozenModel": FrozenModel, "MutableModel": MutableModel}[name]
    if name == "ClockBase":
        from raysearch.core.workunit import ClockBase

        return ClockBase
    if name == "Overrides":
        from raysearch.core.overrides import Overrides

        return Overrides
    if name == "WorkUnit":
        from raysearch.core.workunit import WorkUnit

        return WorkUnit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
