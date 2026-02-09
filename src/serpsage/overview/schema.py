from __future__ import annotations

from serpsage.app.response import OverviewResult


def overview_json_schema() -> dict:
    return OverviewResult.model_json_schema()


__all__ = ["overview_json_schema"]

