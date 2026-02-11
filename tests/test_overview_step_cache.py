from __future__ import annotations

import pytest

from serpsage import Engine, SearchRequest
from serpsage.contracts.services import CacheBase, LLMClientBase, SearchProviderBase
from serpsage.core.runtime import ComponentOverrides
from serpsage.models.llm import ChatJSONResult, LLMUsage
from serpsage.settings.models import AppSettings


class FakeProvider(SearchProviderBase):
    def __init__(self, items):
        self._items = items

    async def asearch(self, *, query: str, params=None):  # noqa: ANN001
        _ = query, params
        return list(self._items)


class FakeCache(CacheBase):
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], bytes] = {}

    async def aget(self, *, namespace: str, key: str):  # noqa: ANN001
        return self._store.get((namespace, key))

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int):  # noqa: ANN001
        _ = ttl_s
        self._store[(namespace, key)] = bytes(value)

    async def aclose(self) -> None:
        return


class FakeLLM(LLMClientBase):
    def __init__(self) -> None:
        self.calls = 0

    async def chat_json(self, *, model, messages, schema, timeout_s=None):  # noqa: ANN001
        _ = model, messages, schema, timeout_s
        self.calls += 1
        return ChatJSONResult(
            data={"summary": "ok", "key_points": ["p1"], "citations": []},
            usage=LLMUsage(),
        )


@pytest.mark.anyio
async def test_overview_cache_hit_skips_llm_call():
    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.0},
            "enrich": {"enabled": False},
            "overview": {
                "enabled": True,
                "cache_ttl_s": 60,
                "llm": {"api_key": "dummy"},
            },
            "cache": {"enabled": False},
        }
    )
    llm = FakeLLM()
    overrides = ComponentOverrides(
        provider=FakeProvider(
            [{"url": "https://e.com", "title": "python", "snippet": "x"}]
        ),
        llm=llm,
        cache=FakeCache(),
    )
    async with Engine.from_settings(settings, overrides=overrides) as engine:
        resp1 = await engine.run(
            SearchRequest(query="python", depth="simple", max_results=5)
        )
        resp2 = await engine.run(
            SearchRequest(query="python", depth="simple", max_results=5)
        )

    assert resp1.overview is not None
    assert resp2.overview is not None
    assert llm.calls == 1
