from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.contracts.services import SearchProviderBase
from serpsage.core.runtime import ComponentOverrides
from serpsage.settings.models import AppSettings


class FakeProvider(SearchProviderBase):
    def __init__(self, items):
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


@pytest.mark.anyio
async def test_trace_telemetry_has_parent_child_spans():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {"enabled": False},
            "cache": {"enabled": False},
            "telemetry": {"enabled": True, "include_events": True},
        }
    )
    overrides = ComponentOverrides(
        provider=FakeProvider([{"url": "https://e.com", "title": "python", "snippet": "x"}])
    )

    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(SearchRequest(query="python", depth="simple", max_results=5))

    tel = resp.telemetry
    assert tel["enabled"] is True
    assert isinstance(tel["trace_id"], str) and tel["trace_id"]
    spans = tel["spans"]
    assert isinstance(spans, list) and spans

    engine_span = next(s for s in spans if s["name"] == "engine.run")
    engine_id = engine_span["span_id"]
    step_spans = [s for s in spans if isinstance(s.get("name"), str) and s["name"].startswith("step.")]
    assert step_spans
    assert all(s.get("parent_id") == engine_id for s in step_spans)
