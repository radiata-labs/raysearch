from __future__ import annotations

import time
from typing import Any, Literal

import pytest

from serpsage.app.request import FetchRequestBase, SearchRequest
from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import SearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.search.optimize import SearchOptimizeStep
from serpsage.telemetry.base import ClockBase
from serpsage.telemetry.trace import NoopTelemetry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _FakeLLM(LLMClientBase):
    def __init__(
        self, *, rt: Runtime, outputs: list[ChatResult | Exception] | None = None
    ) -> None:
        super().__init__(rt=rt)
        self._outputs = list(outputs or [])
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "schema": schema,
                "timeout_s": timeout_s,
            }
        )
        if not self._outputs:
            return ChatResult(
                data={
                    "search_query": "optimized default query",
                    "optimize_query": True,
                    "freshness_intent": False,
                    "query_language": "English",
                }
            )
        item = self._outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _build_runtime() -> Runtime:
    settings = AppSettings()
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_TestClock())


def _build_context(
    *, rt: Runtime, query: str, mode: Literal["fast", "auto", "deep"]
) -> SearchStepContext:
    req = SearchRequest(
        query=query,
        mode=mode,
        fetchs=FetchRequestBase(content=True, abstracts=False, overview=False),
    )
    return SearchStepContext(settings=rt.settings, request=req, request_id="opt-test")


@pytest.mark.anyio
async def test_optimize_step_skips_fast_mode() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt)
    step = SearchOptimizeStep(rt=rt, llm=llm)
    ctx = _build_context(rt=rt, query="latest ai papers", mode="fast")

    out = await step.run(ctx)

    assert out.request.query == "latest ai papers"
    assert llm.calls == []


@pytest.mark.anyio
async def test_optimize_step_auto_mode_updates_query_and_uses_time_prompt() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(
        rt=rt,
        outputs=[
            ChatResult(
                data={
                    "search_query": "latest ai papers 2026 site:arxiv.org",
                    "optimize_query": True,
                    "freshness_intent": True,
                    "query_language": "English",
                }
            )
        ],
    )
    step = SearchOptimizeStep(rt=rt, llm=llm)
    ctx = _build_context(rt=rt, query="latest ai papers", mode="auto")

    out = await step.run(ctx)

    assert out.request.query == "latest ai papers 2026 site:arxiv.org"
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "Current UTC timestamp:" in prompt
    assert "Decision policy:" in prompt
    assert "latest/current/today/now/as of/this year/month" in prompt


@pytest.mark.anyio
async def test_optimize_step_deep_mode_failure_falls_back_to_original_query() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt, outputs=[RuntimeError("optimizer unavailable")])
    step = SearchOptimizeStep(rt=rt, llm=llm)
    ctx = _build_context(rt=rt, query="current gpu benchmark", mode="deep")

    out = await step.run(ctx)

    assert out.request.query == "current gpu benchmark"
    assert any(err.code == "search_query_optimize_failed" for err in out.errors)
