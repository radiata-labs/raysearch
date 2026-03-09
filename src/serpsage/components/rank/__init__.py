from __future__ import annotations

from typing import Any

from serpsage.components.rank.base import RankerBase, RankMode


def build_ranker(*, rt: Any) -> RankerBase:
    return rt.components.resolve_default("rank", expected_type=RankerBase)


__all__ = ["RankMode", "RankerBase", "build_ranker"]
