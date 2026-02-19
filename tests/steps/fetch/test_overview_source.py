from __future__ import annotations

from typing import Any

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import FetchRequest
from serpsage.components.cache.base import CacheBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.models.extract import ExtractedDocument
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import (
    FetchRuntimeConfig,
    FetchStepContext,
    ScoredAbstract,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.overview import FetchOverviewStep


class _DummyCache(CacheBase):
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        del namespace, key
        return None

    async def aset(
        self, *, namespace: str, key: str, value: bytes, ttl_s: int
    ) -> None:
        del namespace, key, value, ttl_s


class _DummyLLM(LLMClientBase):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.last_messages: list[dict[str, str]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        del model, schema, timeout_s
        self.last_messages = list(messages)
        return ChatResult(text="overview-ok")


def _build_ctx(*, settings: AppSettings) -> FetchStepContext:
    request = FetchRequest(
        urls=["https://example.com"],
        content=False,
        overview={"query": "what is deepseek v3.2"},
    )
    return FetchStepContext(
        settings=settings,
        request=request,
        url="https://example.com",
        url_index=0,
        runtime=FetchRuntimeConfig(),
    )


def test_overview_source_prefers_md_for_abstract_when_no_scored_abstracts() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchOverviewStep(rt=rt, llm=_DummyLLM(rt=rt), cache=_DummyCache(rt=rt))
    ctx = _build_ctx(settings=settings)
    ctx.artifacts.extracted = ExtractedDocument(
        markdown="raw markdown content",
        md_for_abstract="clean abstract text\nsecond line",
    )

    out = step._build_source_items(ctx)

    assert out == ["clean abstract text second line"]


def test_overview_query_none_uses_default_query_instruction() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    llm = _DummyLLM(rt=rt)
    step = FetchOverviewStep(rt=rt, llm=llm, cache=_DummyCache(rt=rt))
    ctx = _build_ctx(settings=settings)
    ctx.request = FetchRequest(
        urls=["https://example.com"],
        content=False,
        overview=True,
    )
    ctx.artifacts.extracted = ExtractedDocument(
        title="DeepSeek V3.2",
        markdown="raw markdown content",
    )
    ctx.artifacts.overview_scored_abstracts = [
        ScoredAbstract(abstract_id="S1:A1", text="scored abstract text", score=0.9)
    ]

    out = anyio.run(step.run, ctx)

    assert out.artifacts.overview_output == "overview-ok"
    assert llm.last_messages
    user_message = llm.last_messages[1]["content"]
    assert "No explicit user query was provided." in user_message
    assert "SOURCE_ABSTRACTS:\n[A1] scored abstract text" in user_message
