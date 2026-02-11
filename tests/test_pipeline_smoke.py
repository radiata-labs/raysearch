from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.app.bootstrap import build_runtime
from serpsage.contracts.services import SearchProviderBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.settings.models import AppSettings


class FakeProvider(SearchProviderBase):
    def __init__(self, *, rt: Runtime, items):
        super().__init__(rt=rt)
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
    rt = build_runtime(settings=settings)
    provider = FakeProvider(
        rt=rt,
        items=[
            {"url": "https://example.com/a", "title": "alpha python", "snippet": "x"},
            {"url": "https://example.com/b", "title": "beta", "snippet": "python y"},
        ],
    )
    overrides = Overrides(provider=provider)

    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(
            SearchRequest(query="python", depth="simple", max_results=10)
        )

    assert resp.errors == []
    assert len(resp.results) == 2
    assert resp.results[0].source_id == "S1"
    assert resp.results[1].source_id == "S2"
    assert resp.results[0].score >= resp.results[1].score
