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


class VerboseLLM(LLMClientBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    async def chat_json(self, *, model, messages, schema, timeout_s=None):  # noqa: ANN001
        _ = model, messages, schema, timeout_s
        return ChatJSONResult(
            data={
                "summary": "abcd efgh ijkl mnop",
                "key_points": ["p1"],
                "citations": [],
            },
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


@pytest.mark.anyio
async def test_overview_summary_is_truncated_and_usage_present():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {
                "enabled": True,
                "max_output_tokens": 2,
                "self_heal_retries": 0,
                "llm": {"api_key": "dummy"},
            },
            "cache": {"enabled": False},
        }
    )
    rt = build_runtime(settings=settings)
    overrides = Overrides(
        provider=FakeProvider(
            rt=rt, items=[{"url": "https://e.com", "title": "python", "snippet": "x"}]
        ),
        llm=VerboseLLM(rt=rt),
    )

    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp = await engine.run(
            SearchRequest(query="python", depth="simple", max_results=5)
        )

    assert resp.overview is not None
    assert resp.overview.summary != "abcd efgh ijkl mnop"
    assert resp.overview.usage.total_tokens == 30
