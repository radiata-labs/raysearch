from __future__ import annotations

from typing import Any

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import FetchRequest
from serpsage.components.cache.base import CacheBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.models.extract import ExtractedDocument
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import FetchStepContext, FetchStepOthers
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
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        del model, messages, schema, timeout_s
        return ChatResult(text="")


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
        others=FetchStepOthers(),
    )


def test_overview_source_prefers_md_for_abstract_when_no_scored_abstracts() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchOverviewStep(rt=rt, llm=_DummyLLM(rt=rt), cache=_DummyCache(rt=rt))
    ctx = _build_ctx(settings=settings)
    ctx.extracted = ExtractedDocument(
        markdown="raw markdown content",
        md_for_abstract="clean abstract text\nsecond line",
    )

    out = step._build_source_items(ctx)

    assert out == ["clean abstract text second line"]
