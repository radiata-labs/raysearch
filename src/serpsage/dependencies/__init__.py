from __future__ import annotations

from typing import Final, TypeVar, cast

from serpsage.dependencies.utils import (
    Dependency,
    InnerDepends,
    solve_dependencies,
)

_T = TypeVar("_T")

SEARCH_RUNNER: Final[str] = "app.search_runner"
FETCH_RUNNER: Final[str] = "app.fetch_runner"
CHILD_FETCH_RUNNER: Final[str] = "app.child_fetch_runner"
ANSWER_RUNNER: Final[str] = "app.answer_runner"
RESEARCH_ROUND_RUNNER: Final[str] = "app.research_round_runner"
RESEARCH_RUNNER: Final[str] = "app.research_runner"
RESEARCH_SUBREPORT_STEP: Final[str] = "app.research_subreport_step"


def Depends(dependency: Dependency[_T] | None = None, *, use_cache: bool = True) -> _T:
    return cast("_T", InnerDepends(dependency=dependency, use_cache=use_cache))


__all__ = [
    "ANSWER_RUNNER",
    "CHILD_FETCH_RUNNER",
    "Dependency",
    "Depends",
    "FETCH_RUNNER",
    "RESEARCH_ROUND_RUNNER",
    "RESEARCH_RUNNER",
    "RESEARCH_SUBREPORT_STEP",
    "SEARCH_RUNNER",
    "solve_dependencies",
]
