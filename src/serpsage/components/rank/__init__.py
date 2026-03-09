from __future__ import annotations

from typing import Any, cast

from serpsage.components.rank.base import RankerBase, RankMode


def build_ranker(*, rt: Any) -> RankerBase:
    return cast("RankerBase", rt.services.require(RankerBase))


__all__ = ["RankMode", "RankerBase", "build_ranker"]
