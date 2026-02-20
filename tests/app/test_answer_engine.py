from __future__ import annotations

import time
from typing import Any

import pytest

from serpsage.app.engine import Engine
from serpsage.app.request import AnswerRequest
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.models.errors import AppError
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import (
    AnswerStepContext,
    FetchStepContext,
    SearchStepContext,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.answer import AnswerGenerateStep, AnswerPlanStep, AnswerSearchStep
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.telemetry.base import ClockBase, SpanBase
from serpsage.telemetry.trace import NoopTelemetry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _FakeLLM(LLMClientBase):
    def __init__(self, *, rt: Runtime, outputs: list[ChatResult | Exception]) -> None:
        super().__init__(rt=rt)
        self._outputs = list(outputs)
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
            return ChatResult()
        item = self._outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _SearchStubStep(StepBase[SearchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        results: list[FetchResultItem] | None = None,
        errors: list[AppError] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        super().__init__(rt=rt)
        self.calls: list[object] = []
        self._results = list(results or [])
        self._errors = list(errors or [])
        self._raise_exc = raise_exc

    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        _ = span
        self.calls.append(ctx.request.model_copy(deep=True))
        if self._raise_exc is not None:
            raise self._raise_exc
        ctx.output.results = [item.model_copy(deep=True) for item in self._results]
        ctx.errors.extend(list(self._errors))
        return ctx


def _make_settings(*, with_custom_models: bool = False) -> AppSettings:
    if not with_custom_models:
        return AppSettings()
    return AppSettings(
        llm={"models": [{"name": "planner"}, {"name": "writer"}]},
        fetch={"overview": {"use_model": "planner"}},
        answer={
            "plan": {"use_model": "planner"},
            "generate": {"use_model": "writer", "max_abstract_chars": 3000},
        },
    )


def _build_engine(
    *,
    outputs: list[ChatResult | Exception],
    settings: AppSettings | None = None,
    search_results: list[FetchResultItem] | None = None,
    search_errors: list[AppError] | None = None,
    search_raise_exc: Exception | None = None,
) -> tuple[Engine, _FakeLLM, _SearchStubStep]:
    use_settings = settings or _make_settings()
    rt = Runtime(settings=use_settings, telemetry=NoopTelemetry(), clock=_TestClock())
    llm = _FakeLLM(rt=rt, outputs=outputs)
    search_step = _SearchStubStep(
        rt=rt,
        results=search_results,
        errors=search_errors,
        raise_exc=search_raise_exc,
    )
    search_runner = RunnerBase[SearchStepContext](
        rt=rt, steps=[search_step], kind="search"
    )
    fetch_runner = RunnerBase[FetchStepContext](rt=rt, steps=[], kind="fetch")
    answer_steps: list[StepBase[AnswerStepContext]] = [
        AnswerPlanStep(rt=rt, llm=llm),
        AnswerSearchStep(rt=rt, search_runner=search_runner),
        AnswerGenerateStep(rt=rt, llm=llm),
    ]
    answer_runner = RunnerBase[AnswerStepContext](
        rt=rt, steps=answer_steps, kind="search"
    )
    engine = Engine(
        rt=rt,
        search_runner=search_runner,
        fetch_runner=fetch_runner,
        answer_runner=answer_runner,
    )
    return engine, llm, search_step


def _plan_payload(
    *,
    query: str = "optimized query",
    depth: str = "auto",
    max_results: int = 5,
    additional_queries: list[str] | None = None,
    answer_mode: str = "direct",
    freshness_intent: bool = False,
    optimize_query: bool = True,
) -> dict[str, Any]:
    return {
        "answer_mode": answer_mode,
        "freshness_intent": freshness_intent,
        "search_query": query,
        "optimize_query": optimize_query,
        "search_depth": depth,
        "max_results": max_results,
        "additional_queries": list(additional_queries or []),
    }


@pytest.mark.anyio
async def test_answer_plan_drives_search_and_uses_content_flag() -> None:
    settings = _make_settings(with_custom_models=True)
    result = FetchResultItem(
        url="https://a.example.com",
        title="A",
        content="full markdown",
        abstracts=["a abstract"],
        abstract_scores=[0.9],
        subpages=[],
        overview="",
    )
    engine, llm, search_step = _build_engine(
        settings=settings,
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="optimized ai benchmark query",
                    depth="deep",
                    max_results=5,
                    additional_queries=["query a", "query b"],
                    answer_mode="summary",
                    freshness_intent=True,
                )
            ),
            ChatResult(text="Final answer [citation:1]"),
        ],
        search_results=[result],
    )

    async with engine:
        resp = await engine.answer(
            AnswerRequest(query="latest ai benchmark", content=True)
        )

    req = search_step.calls[0]
    assert req.query == "optimized ai benchmark query"
    assert req.depth == "deep"
    assert req.max_results == 5
    assert req.additional_queries == ["query a", "query b"]
    assert req.fetchs.content is True
    assert req.fetchs.abstracts.query == req.query
    assert req.fetchs.subpages is not None
    assert req.fetchs.subpages.max_subpages == 2
    assert req.fetchs.subpages.subpage_keywords == req.query
    assert resp.answer == "Final answer [citation:https://a.example.com]"
    assert [item.id for item in resp.citations] == ["https://a.example.com"]
    assert [item.url for item in resp.citations] == ["https://a.example.com"]
    assert [item.content for item in resp.citations] == ["full markdown"]
    assert "Current UTC timestamp:" in llm.calls[0]["messages"][1]["content"]
    assert "CURRENT_UTC_TIMESTAMP:" in llm.calls[1]["messages"][1]["content"]
    assert "Temporal reasoning rules:" in llm.calls[1]["messages"][0]["content"]
    assert (
        "Output language must strictly match QUERY language and script."
        in llm.calls[1]["messages"][0]["content"]
    )
    assert [call["model"] for call in llm.calls] == ["planner", "writer"]


@pytest.mark.anyio
async def test_answer_plan_normalizes_depth_constraints_and_caps_results() -> None:
    settings = _make_settings()
    settings.search.max_results = 9
    engine, llm, search_step = _build_engine(
        settings=settings,
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="optimized question",
                    depth="AUTO",
                    max_results=999,
                    additional_queries=["a", "b", "c"],
                )
            ),
            ChatResult(text="ok"),
        ],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="easy fact question"))

    req = search_step.calls[0]
    assert req.query == "optimized question"
    assert req.depth == "auto"
    assert req.max_results == 9
    assert req.additional_queries is None
    assert req.fetchs.content is False
    assert req.fetchs.subpages is None
    assert resp.citations == []
    assert "CURRENT_UTC_TIMESTAMP:" not in llm.calls[1]["messages"][1]["content"]
    assert "Temporal reasoning rules:" not in llm.calls[1]["messages"][0]["content"]


@pytest.mark.anyio
async def test_answer_citations_are_page_level_unique_and_reference_ordered() -> None:
    result = FetchResultItem(
        url="https://one.example.com/page",
        title="one",
        content="one md",
        abstracts=["one abstract"],
        abstract_scores=[0.9],
        subpages=[],
        overview="",
    )
    result2 = FetchResultItem(
        url="https://two.example.com/page",
        title="two",
        content="two md",
        abstracts=["two abstract"],
        abstract_scores=[0.8],
        subpages=[],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="q",
                    depth="deep",
                    max_results=5,
                    additional_queries=["qa"],
                    answer_mode="summary",
                )
            ),
            ChatResult(text="Answer [citation:2] and [citation:1] and [citation:2]."),
        ],
        search_results=[result, result2],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="question", content=False))

    assert (
        resp.answer == "Answer [citation:https://two.example.com/page] and "
        "[citation:https://one.example.com/page] and "
        "[citation:https://two.example.com/page]."
    )
    assert [item.id for item in resp.citations] == [
        "https://two.example.com/page",
        "https://one.example.com/page",
    ]
    assert [item.url for item in resp.citations] == [
        "https://two.example.com/page",
        "https://one.example.com/page",
    ]
    payload = resp.model_dump()
    assert "content" not in payload["citations"][0]


@pytest.mark.anyio
async def test_answer_citation_dedupes_same_page_url_fragment() -> None:
    result = FetchResultItem(
        url="https://main.example.com/report#overview",
        title="main",
        content="main markdown",
        abstracts=["main abstract"],
        abstract_scores=[0.9],
        subpages=[
            FetchSubpagesResult(
                url="https://main.example.com/report",
                title="sub",
                content="sub markdown",
                abstracts=["sub abstract"],
                abstract_scores=[0.8],
                overview="",
            )
        ],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="q",
                    depth="deep",
                    max_results=5,
                    additional_queries=["qa"],
                    answer_mode="summary",
                )
            ),
            ChatResult(text="Answer [citation:1] [citation:2] [citation:1]"),
        ],
        search_results=[result],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="question", content=True))

    assert (
        resp.answer == "Answer [citation:https://main.example.com/report#overview] "
        "[citation:2] "
        "[citation:https://main.example.com/report#overview]"
    )
    assert [item.id for item in resp.citations] == [
        "https://main.example.com/report#overview"
    ]
    assert [item.url for item in resp.citations] == [
        "https://main.example.com/report#overview"
    ]
    assert [item.content for item in resp.citations] == ["main markdown"]


@pytest.mark.anyio
async def test_answer_no_citation_marker_keeps_citations_empty() -> None:
    result = FetchResultItem(
        url="https://a.example.com",
        title="a",
        content="a md",
        abstracts=["a abstract"],
        abstract_scores=[0.8],
        subpages=[],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(data=_plan_payload(query="q")),
            ChatResult(text="No citation marker here."),
        ],
        search_results=[result],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="question", content=True))

    assert resp.citations == []


@pytest.mark.anyio
async def test_answer_generate_abstract_budget_applies_to_prompt() -> None:
    settings = _make_settings(with_custom_models=True)
    settings.answer.generate.max_abstract_chars = 10
    result = FetchResultItem(
        url="https://a.example.com",
        title="a",
        content="a md",
        abstracts=["abcdefghijklmno", "second item"],
        abstract_scores=[0.9, 0.8],
        subpages=[],
        overview="",
    )
    engine, llm, _ = _build_engine(
        settings=settings,
        outputs=[
            ChatResult(data=_plan_payload(query="q")),
            ChatResult(text="Answer [citation:1]"),
        ],
        search_results=[result],
    )

    async with engine:
        await engine.answer(AnswerRequest(query="question"))

    user_prompt = llm.calls[1]["messages"][1]["content"]
    abstract_lines = [
        line[2:] for line in user_prompt.splitlines() if line.startswith("- ")
    ]
    assert abstract_lines
    assert abstract_lines[0] == "abcdefghij"
    assert sum(len(item) for item in abstract_lines) <= 10


@pytest.mark.anyio
async def test_answer_plan_failure_falls_back_to_default_search() -> None:
    engine, llm, search_step = _build_engine(
        outputs=[RuntimeError("planner failed"), ChatResult(text="answer")]
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="fallback question"))

    req = search_step.calls[0]
    assert req.query == "fallback question"
    assert req.depth == "auto"
    assert req.additional_queries is None
    assert req.max_results == min(engine.settings.search.max_results, 5)
    assert any(item.code == "answer_plan_failed" for item in resp.errors)
    assert [call["model"] for call in llm.calls] == [
        engine.settings.answer.plan.use_model,
        engine.settings.answer.generate.use_model,
    ]


@pytest.mark.anyio
async def test_answer_schema_mode_extracts_citations_and_handles_empty() -> None:
    pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    result = FetchResultItem(
        url="https://a.example.com",
        title="a",
        content="a md",
        abstracts=["a abstract"],
        abstract_scores=[0.8],
        subpages=[],
        overview="",
    )

    engine_ok, _, _ = _build_engine(
        outputs=[
            ChatResult(data=_plan_payload(query="q")),
            ChatResult(
                data={"summary": "ok [citation:1]"},
                text='{"summary":"ok [citation:1]"}',
            ),
        ],
        search_results=[result],
    )
    async with engine_ok:
        resp_ok = await engine_ok.answer(
            AnswerRequest(query="q", json_schema=schema, content=True)
        )
    assert resp_ok.answer == {"summary": "ok"}
    assert [item.id for item in resp_ok.citations] == ["https://a.example.com"]
    assert [item.url for item in resp_ok.citations] == ["https://a.example.com"]

    engine_empty, _, _ = _build_engine(
        outputs=[
            ChatResult(data=_plan_payload(query="q")),
            ChatResult(data={"summary": "ok"}, text='{"summary":"ok"}'),
        ],
        search_results=[result],
    )
    async with engine_empty:
        resp_empty = await engine_empty.answer(
            AnswerRequest(query="q", json_schema=schema)
        )
    assert resp_empty.answer == {"summary": "ok"}
    assert resp_empty.citations == []


@pytest.mark.anyio
async def test_answer_direct_mode_hides_citation_markers() -> None:
    result = FetchResultItem(
        url="https://fr.example.com",
        title="capital",
        content="Paris is the capital of France.",
        abstracts=["Paris is the capital city of France."],
        abstract_scores=[0.9],
        subpages=[],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="capital of france",
                    answer_mode="direct",
                    depth="auto",
                    max_results=3,
                )
            ),
            ChatResult(text="Paris [citation:1]"),
        ],
        search_results=[result],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="What is the capital of France?"))

    assert resp.answer == "Paris"
    assert [item.id for item in resp.citations] == ["https://fr.example.com"]


@pytest.mark.anyio
async def test_answer_expands_compound_citation_markers() -> None:
    r1 = FetchResultItem(
        url="https://s1.example.com",
        title="s1",
        content="s1",
        abstracts=["a1"],
        abstract_scores=[0.9],
        subpages=[],
        overview="",
    )
    r2 = FetchResultItem(
        url="https://s2.example.com",
        title="s2",
        content="s2",
        abstracts=["a2"],
        abstract_scores=[0.8],
        subpages=[],
        overview="",
    )
    r3 = FetchResultItem(
        url="https://s3.example.com",
        title="s3",
        content="s3",
        abstracts=["a3"],
        abstract_scores=[0.7],
        subpages=[],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="open question",
                    answer_mode="summary",
                    depth="deep",
                    additional_queries=["q2"],
                )
            ),
            ChatResult(text="Summary [citation:1,2,2,3]."),
        ],
        search_results=[r1, r2, r3],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="state of x"))

    assert (
        resp.answer
        == "Summary [citation:https://s1.example.com]"
        "[citation:https://s2.example.com]"
        "[citation:https://s3.example.com]."
    )
    assert [item.id for item in resp.citations] == [
        "https://s1.example.com",
        "https://s2.example.com",
        "https://s3.example.com",
    ]


@pytest.mark.anyio
async def test_answer_normalizes_spaced_citation_tags() -> None:
    r1 = FetchResultItem(
        url="https://sp1.example.com",
        title="sp1",
        content="sp1",
        abstracts=["a1"],
        abstract_scores=[0.9],
        subpages=[],
        overview="",
    )
    r2 = FetchResultItem(
        url="https://sp2.example.com",
        title="sp2",
        content="sp2",
        abstracts=["a2"],
        abstract_scores=[0.8],
        subpages=[],
        overview="",
    )
    engine, _, _ = _build_engine(
        outputs=[
            ChatResult(
                data=_plan_payload(
                    query="spaced citation question",
                    answer_mode="summary",
                    depth="deep",
                    additional_queries=["q2"],
                )
            ),
            ChatResult(
                text=(
                    "Summary [ citation:1 ] [CITATION : 2] "
                    "and merged [ citation: 1,2 ]."
                )
            ),
        ],
        search_results=[r1, r2],
    )

    async with engine:
        resp = await engine.answer(AnswerRequest(query="state of y"))

    assert (
        resp.answer
        == "Summary [citation:https://sp1.example.com] "
        "[citation:https://sp2.example.com] and merged "
        "[citation:https://sp1.example.com][citation:https://sp2.example.com]."
    )
    assert [item.id for item in resp.citations] == [
        "https://sp1.example.com",
        "https://sp2.example.com",
    ]
