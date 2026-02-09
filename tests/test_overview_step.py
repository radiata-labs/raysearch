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


class FakeLLM:
    async def chat_json(self, *, model, messages, schema, timeout_s=None):  # noqa: ANN001
        _ = model, messages, schema, timeout_s
        return {
            "summary": "ok",
            "key_points": ["p1"],
            "citations": [],
        }


@pytest.mark.anyio
async def test_overview_runs_when_enabled_and_key_present():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {"enabled": True, "llm": {"api_key": "dummy"}},
            "cache": {"enabled": False},
        }
    )
    overrides = Overrides(
        provider=FakeProvider([{"url": "https://e.com", "title": "python", "snippet": "x"}]),
        llm=FakeLLM(),
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(SearchRequest(query="python", depth="simple", max_results=5))

    assert resp.overview is not None
    assert resp.overview.summary == "ok"
