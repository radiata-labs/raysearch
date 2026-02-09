from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.app.container import Overrides
from serpsage.settings.models import AppSettings


class FakeProvider:
    def __init__(self, items):
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


@pytest.mark.anyio
async def test_pipeline_smoke_basic_ranking_and_ids():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0, "include_raw": False},
            "enrich": {"enabled": False},
            "overview": {"enabled": False},
            "cache": {"enabled": False},
        }
    )
    provider = FakeProvider(
        [
            {"url": "https://example.com/a", "title": "alpha python", "snippet": "x"},
            {"url": "https://example.com/b", "title": "beta", "snippet": "python y"},
        ]
    )
    overrides = Overrides(provider=provider)

    async with Engine(settings, overrides=overrides) as engine:
        resp = await engine.run(SearchRequest(query="python", depth="simple", max_results=10))

    assert resp.errors == []
    assert len(resp.results) == 2
    assert resp.results[0].source_id == "S1"
    assert resp.results[1].source_id == "S2"
    assert resp.results[0].score >= resp.results[1].score

