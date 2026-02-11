from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.app.bootstrap import build_runtime
from serpsage.contracts.services import LLMClientBase, SearchProviderBase
from serpsage.core.runtime import Overrides, Runtime
from serpsage.models.llm import ChatJSONResult, LLMUsage
from serpsage.settings.models import AppSettings


class FakeProvider(SearchProviderBase):
    def __init__(self, *, rt: Runtime, items):
        super().__init__(rt=rt)
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


class FakeLLM(LLMClientBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    async def chat_json(self, *, model, messages, schema, timeout_s=None):  # noqa: ANN001
        _ = model, messages, schema, timeout_s
        return ChatJSONResult(
            data={
                "summary": "ok",
                "key_points": ["p1"],
                "citations": [
                    {
                        "cite_id": "X",
                        "source_id": "S99",
                        "url": "https://nope",
                        "title": "bad",
                        "chunk_id": None,
                        "quote": None,
                    },
                    {
                        "cite_id": "X",
                        "source_id": "S1",
                        "url": "https://wrong.example",
                        "title": "t",
                        "chunk_id": "S1:C999",
                        "quote": "q",
                    },
                ],
            },
            usage=LLMUsage(),
        )


@pytest.mark.anyio
async def test_overview_citations_are_sanitized():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {"enabled": True, "llm": {"api_key": "dummy"}},
            "cache": {"enabled": False},
        }
    )
    rt = build_runtime(settings=settings)
    overrides = Overrides(
        provider=FakeProvider(
            rt=rt, items=[{"url": "https://e.com", "title": "python", "snippet": "x"}]
        ),
        llm=FakeLLM(rt=rt),
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(
            SearchRequest(query="python", depth="simple", max_results=5)
        )

    assert resp.overview is not None
    # S99 citation dropped; remaining one is normalized.
    assert len(resp.overview.citations) == 1
    c = resp.overview.citations[0]
    assert c.source_id == "S1"
    assert c.url == "https://e.com"
    assert c.chunk_id is None
    assert c.cite_id == "C1"
