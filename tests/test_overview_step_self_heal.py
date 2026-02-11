from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.contracts.services import LLMClientBase, SearchProviderBase
from serpsage.core.runtime import ComponentOverrides
from serpsage.models.llm import ChatJSONResult, LLMUsage
from serpsage.settings.models import AppSettings


class FakeProvider(SearchProviderBase):
    def __init__(self, items):
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


class FlakyLLM(LLMClientBase):
    def __init__(self) -> None:
        self.calls = 0

    async def chat_json(self, *, model, messages, schema, timeout_s=None):  # noqa: ANN001
        _ = model, messages, schema, timeout_s
        self.calls += 1
        if self.calls == 1:
            # Invalid type: summary must be str.
            return ChatJSONResult(
                data={"summary": 123, "key_points": [], "citations": []},
                usage=LLMUsage(),
            )
        return ChatJSONResult(
            data={"summary": "ok", "key_points": ["p1"], "citations": []},
            usage=LLMUsage(),
        )


@pytest.mark.anyio
async def test_overview_self_heal_retries_on_validation_error():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {"enabled": True, "self_heal_retries": 1, "llm": {"api_key": "dummy"}},
            "cache": {"enabled": False},
        }
    )
    llm = FlakyLLM()
    overrides = ComponentOverrides(
        provider=FakeProvider([{"url": "https://e.com", "title": "python", "snippet": "x"}]),
        llm=llm,
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(SearchRequest(query="python", depth="simple", max_results=5))

    assert llm.calls == 2
    assert resp.errors == []
    assert resp.overview is not None
    assert resp.overview.summary == "ok"
