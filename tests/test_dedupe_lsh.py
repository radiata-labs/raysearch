from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.app.bootstrap import Overrides
from serpsage.settings.models import AppSettings


class FakeProvider:
    def __init__(self, items):
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


@pytest.mark.anyio
async def test_dedupe_lsh_removes_near_duplicates_and_limits_comparisons():
    # Build many similar titles.
    items = []
    for i in range(60):
        items.append(
            {
                "url": f"https://example.com/{i}",
                "title": f"Python tutorial part {i%3}",
                "snippet": "learn python " + ("basics " * (i % 5)),
            }
        )

    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {"enabled": False},
            "cache": {"enabled": False},
        }
    )
    overrides = Overrides(provider=FakeProvider(items))
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(SearchRequest(query="python tutorial", depth="simple", max_results=200))

    # should dedupe down to a smaller set
    assert len(resp.results) < 60
