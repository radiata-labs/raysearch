from __future__ import annotations

import time
from typing import Any

import pytest

from serpsage.app.request import FetchRequestBase, SearchRequest
from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import SearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.search.expand import SearchExpandStep
from serpsage.telemetry.base import ClockBase
from serpsage.telemetry.trace import NoopTelemetry
from serpsage.utils import clean_whitespace


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
        if self._outputs:
            item = self._outputs.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return ChatResult(
            data={
                "queries": [
                    "orthogonal source viewpoint",
                    "third-party independent analysis",
                ]
            }
        )


def _build_runtime() -> Runtime:
    settings = AppSettings()
    settings.search.deep.enabled = True
    settings.search.deep.max_expanded_queries = 6
    settings.search.deep.rule_max_queries = 4
    settings.search.deep.llm_max_queries = 2
    settings.search.deep.expansion_model = "gpt-4.1-mini"
    settings.search.deep.expansion_timeout_s = 10.0
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_TestClock())


def _build_context(
    *,
    rt: Runtime,
    query: str,
    additional_queries: list[str] | None = None,
    mode: str = "deep",
) -> SearchStepContext:
    req = SearchRequest(
        query=query,
        mode=mode,
        additional_queries=additional_queries,
        fetchs=FetchRequestBase(content=True, abstracts=False, overview=False),
    )
    return SearchStepContext(settings=rt.settings, request=req, request_id="expand-test")


_MANY_QUESTIONS = [
    "what is python asyncio",
    "latest ai benchmark papers 2026",
    "kubernetes crashloopbackoff troubleshooting",
    "how to parse PDF tables in python",
    "AAPL earnings date Q4 2025",
    "OpenAI API structured output json schema example",
    "best postgres indexing strategy for time series",
    "how to reduce docker image size for python apps",
    "\u5bf9\u6bd4 FastAPI \u548c Django \u7684\u6027\u80fd",
    "\u4eca\u5929\u7f8e\u8054\u50a8\u5229\u7387\u51b3\u8bae\u662f\u4ec0\u4e48",
    "\u6771\u4eac \u89b3\u5149 \u63a8\u8350 3\u5929",
    "  who   is  the   CEO of OpenAI now   ",
    "C++26 executors proposal latest draft",
    "rust async cancellation safety patterns",
    "how to evaluate RAG retrieval quality objectively",
]


def _expected_rule_suffixes(query: str) -> tuple[str, str]:
    if any("\u3040" <= ch <= "\u30ff" for ch in query):
        return (
            "\u516c\u5f0f \u30c9\u30ad\u30e5\u30e1\u30f3\u30c8 \u30ac\u30a4\u30c9 \u6bd4\u8f03",
            "\u30d9\u30f3\u30c1\u30de\u30fc\u30af \u30ec\u30dd\u30fc\u30c8 \u30bd\u30fc\u30b9",
        )
    if any("\u4e00" <= ch <= "\u9fff" for ch in query):
        return (
            "\u5b98\u65b9 \u6587\u6863 \u6307\u5357 \u5bf9\u6bd4",
            "\u8bc4\u6d4b \u62a5\u544a \u6765\u6e90",
        )
    return ("official docs guide comparison", "benchmark report source")


@pytest.mark.anyio
@pytest.mark.parametrize("query", _MANY_QUESTIONS)
async def test_expand_multi_question_quality_profile(query: str) -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt)
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_context(rt=rt, query=query)

    out = await step.run(ctx)

    assert out.deep.aborted is False
    assert out.deep.query_jobs
    assert out.deep.query_jobs[0].source == "primary"
    assert out.deep.query_jobs[0].query == clean_whitespace(query)

    jobs = list(out.deep.query_jobs)
    assert len(jobs) <= 1 + rt.settings.search.deep.max_expanded_queries
    assert len({job.query.casefold() for job in jobs}) == len(jobs)

    source_counts: dict[str, int] = {}
    for item in jobs:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
    assert source_counts.get("rule", 0) >= 2
    assert source_counts.get("llm", 0) >= 1

    rule_texts = [item.query for item in jobs if item.source == "rule"]
    intent_suffix, evidence_suffix = _expected_rule_suffixes(query)
    assert any(text.endswith(intent_suffix) for text in rule_texts)
    assert any(text.endswith(evidence_suffix) for text in rule_texts)


@pytest.mark.anyio
async def test_expand_manual_queries_are_merged_and_prioritized() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt)
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_context(
        rt=rt,
        query="kubernetes pod restart reason",
        additional_queries=[
            "pod restart root cause",
            "  pod restart root cause ",
            "POD RESTART ROOT CAUSE",
            "kubernetes pod restart reason",
            "container logs timeline",
        ],
    )

    out = await step.run(ctx)

    assert out.deep.aborted is False
    jobs = list(out.deep.query_jobs)
    sources = [item.source for item in jobs]
    assert sources[0] == "primary"
    assert sources.count("manual") >= 1

    manual_queries = [item.query for item in jobs if item.source == "manual"]
    assert manual_queries
    assert "pod restart root cause" in manual_queries
    assert len(manual_queries) == len({item.casefold() for item in manual_queries})

    first_rule = sources.index("rule")
    first_llm = sources.index("llm")
    first_manual = sources.index("manual")
    assert first_manual < first_rule
    assert first_manual < first_llm


@pytest.mark.anyio
async def test_expand_respects_global_expansion_cap() -> None:
    rt = _build_runtime()
    rt.settings.search.deep.max_expanded_queries = 2
    rt.settings.search.deep.rule_max_queries = 4
    rt.settings.search.deep.llm_max_queries = 2
    llm = _FakeLLM(
        rt=rt,
        outputs=[
            ChatResult(
                data={
                    "queries": [
                        "orthogonal source viewpoint",
                        "third-party independent analysis",
                        "extra candidate should be trimmed",
                    ]
                }
            )
        ],
    )
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_context(
        rt=rt,
        query="python context manager internals",
        additional_queries=["manual one", "manual two", "manual three"],
    )

    out = await step.run(ctx)

    assert out.deep.aborted is False
    jobs = list(out.deep.query_jobs)
    assert len(jobs) <= 3  # primary + max_expanded_queries(2)
    assert jobs[0].source == "primary"


@pytest.mark.anyio
async def test_expand_llm_failure_aborts_for_any_question() -> None:
    rt = _build_runtime()
    llm = _FakeLLM(rt=rt, outputs=[RuntimeError("llm expansion unavailable")])
    step = SearchExpandStep(rt=rt, llm=llm)
    ctx = _build_context(
        rt=rt,
        query="\u6700\u65b0 AI \u8bba\u6587 \u662f\u4ec0\u4e48",
    )

    out = await step.run(ctx)

    assert out.deep.aborted is True
    assert out.deep.query_jobs == []
    assert any(err.code == "search_query_expansion_failed" for err in out.errors)
