from __future__ import annotations

import inspect
import time
from typing import Any

import pytest

from serpsage.app.engine import Engine
from serpsage.app.request import ResearchRequest, SearchRequest
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.models.llm import ChatResult
from serpsage.models.pipeline import (
    AnswerStepContext,
    FetchStepContext,
    ResearchStepContext,
    SearchStepContext,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.research import (
    ResearchAbstractStep,
    ResearchContentStep,
    ResearchDecideStep,
    ResearchFinalizeStep,
    ResearchLoopStep,
    ResearchPlanStep,
    ResearchPrepareStep,
    ResearchRenderStep,
    ResearchSearchStep,
    ResearchThemeStep,
)
from serpsage.telemetry.base import ClockBase, SpanBase
from serpsage.telemetry.trace import NoopTelemetry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _NoopFetchStep(StepBase[FetchStepContext]):
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        _ = span
        return ctx


class _NoopAnswerStep(StepBase[AnswerStepContext]):
    async def run_inner(
        self, ctx: AnswerStepContext, *, span: SpanBase
    ) -> AnswerStepContext:
        _ = span
        return ctx


class _CaptureSearchStep(StepBase[SearchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        result_map: dict[str, list[FetchResultItem]],
        request_log: list[SearchRequest],
    ) -> None:
        super().__init__(rt=rt)
        self._result_map = result_map
        self._request_log = request_log

    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        _ = span
        self._request_log.append(ctx.request)
        ctx.output.results = list(self._result_map.get(ctx.request.query, []))
        return ctx


class _FakeLLM(LLMClientBase):
    def __init__(self, *, rt: Runtime, outputs: list[ChatResult | Exception]) -> None:
        super().__init__(rt=rt)
        self._outputs = list(outputs)

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        _ = model
        _ = messages
        _ = schema
        _ = timeout_s
        if not self._outputs:
            return ChatResult()
        item = self._outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fetch_item(url: str) -> FetchResultItem:
    return FetchResultItem(
        url=url,
        title="Main",
        content="Main content facts.",
        abstracts=["Main abstract."],
        abstract_scores=[0.91],
        subpages=[
            FetchSubpagesResult(
                url=f"{url}/sub",
                title="Sub",
                content="Sub content facts.",
                abstracts=["Sub abstract."],
                abstract_scores=[0.88],
                overview=None,
            )
        ],
        overview="",
    )


def _theme_output(*, detected_input_language: str = "en") -> ChatResult:
    return ChatResult(
        data={
            "detected_input_language": detected_input_language,
            "core_question": "What is the state of theme x?",
            "subthemes": ["subtheme-a"],
            "evidence_targets": ["official source"],
            "risk_conflicts": ["metric mismatch"],
            "initial_strategy": "balanced",
            "seed_queries": ["q1"],
        }
    )


def _plan_output(query: str = "q1") -> ChatResult:
    return ChatResult(
        data={
            "query_strategy": "coverage",
            "search_jobs": [
                {
                    "query": query,
                    "intent": "coverage",
                    "mode": "auto",
                    "include_domains": [],
                    "exclude_domains": [],
                    "include_text": [],
                    "exclude_text": [],
                    "expected_gain": "Improve subtheme coverage.",
                }
            ],
        }
    )


def _abstract_output(
    *,
    stop: bool,
    confidence: float,
    covered: list[str],
    next_queries: list[str],
    need_content_source_ids: list[int] | None = None,
) -> ChatResult:
    return ChatResult(
        data={
            "findings": ["Finding from abstracts [citation:1]"],
            "evidence_grades": [{"source_id": 1, "grade": "A", "reason": "direct"}],
            "conflict_arbitration": [],
            "covered_subthemes": list(covered),
            "coverage_delta": 1.0 if covered else 0.0,
            "critical_gaps": [],
            "confidence": confidence,
            "need_content_source_ids": list(need_content_source_ids or []),
            "next_query_strategy": "stop-ready" if stop else "coverage",
            "next_queries": list(next_queries),
            "stop": stop,
        }
    )


def _content_output(*, stop: bool) -> ChatResult:
    return ChatResult(
        data={
            "resolved_findings": ["Resolved finding [citation:1]"],
            "conflict_resolutions": [],
            "remaining_gaps": [],
            "confidence_adjustment": 0.05,
            "next_query_strategy": "stop-ready" if stop else "coverage",
            "next_queries": [],
            "stop": stop,
        }
    )


def _build_engine(
    *,
    llm_outputs: list[ChatResult | Exception],
    search_map: dict[str, list[FetchResultItem]] | None = None,
    no_progress_rounds_to_stop: int = 2,
) -> tuple[Engine, list[SearchRequest]]:
    settings = AppSettings()
    settings.research.tool_max_attempts = 1
    settings.research.llm_self_heal_retries = 0
    settings.research.no_progress_rounds_to_stop = no_progress_rounds_to_stop
    settings.research.research_fast.max_rounds = 3
    settings.research.research_fast.max_search_calls = 6
    settings.research.research_fast.max_fetch_calls = 24
    settings.research.research_fast.max_queries_per_round = 3
    settings.research.research_fast.max_fetch_per_round = 6

    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_TestClock())
    llm = _FakeLLM(rt=rt, outputs=llm_outputs)
    request_log: list[SearchRequest] = []
    search_runner = RunnerBase[SearchStepContext](
        rt=rt,
        steps=[
            _CaptureSearchStep(
                rt=rt,
                result_map=search_map or {"q1": [_fetch_item("https://a.example.com")]},
                request_log=request_log,
            )
        ],
        kind="search",
    )
    fetch_runner = RunnerBase[FetchStepContext](
        rt=rt,
        steps=[_NoopFetchStep(rt=rt)],
        kind="fetch",
    )
    answer_runner = RunnerBase[AnswerStepContext](
        rt=rt,
        steps=[_NoopAnswerStep(rt=rt)],
        kind="search",
    )
    round_runner = RunnerBase[ResearchStepContext](
        rt=rt,
        steps=[
            ResearchPlanStep(rt=rt, llm=llm),
            ResearchSearchStep(rt=rt, search_runner=search_runner),
            ResearchAbstractStep(rt=rt, llm=llm),
            ResearchContentStep(rt=rt, llm=llm),
            ResearchDecideStep(rt=rt),
        ],
        kind="search",
    )
    research_runner = RunnerBase[ResearchStepContext](
        rt=rt,
        steps=[
            ResearchPrepareStep(rt=rt),
            ResearchThemeStep(rt=rt, llm=llm),
            ResearchLoopStep(rt=rt, round_runner=round_runner),
            ResearchRenderStep(rt=rt, llm=llm),
            ResearchFinalizeStep(rt=rt),
        ],
        kind="search",
    )
    return (
        Engine(
            rt=rt,
            search_runner=search_runner,
            fetch_runner=fetch_runner,
            answer_runner=answer_runner,
            research_runner=research_runner,
        ),
        request_log,
    )


@pytest.mark.anyio
async def test_research_response_has_no_citations_field() -> None:
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text="## 1) Core Conclusions\nx [citation:1]\n\n## 2) Key Findings\ny\n\n## 3) Evidence and Citations\nz [citation:1]\n\n## 4) Uncertainty and Conflicts\nn\n\n## 5) Time Anchors\nt\n\n## 6) Next Research Questions\nq"),
        ]
    )
    async with engine:
        resp = await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    dumped = resp.model_dump()
    assert "citations" not in dumped
    assert not hasattr(resp, "citations")


@pytest.mark.anyio
async def test_research_with_schema_sets_content_json_string_and_structured_object() -> None:
    pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(data={"summary": "ok"}),
        ]
    )
    async with engine:
        resp = await engine.research(
            ResearchRequest(search_mode="research-fast", themes="theme x", json_schema=schema)
        )
    assert isinstance(resp.structured, dict)
    assert resp.structured == {"summary": "ok"}
    assert isinstance(resp.content, str)
    assert '"summary": "ok"' in resp.content


@pytest.mark.anyio
async def test_research_with_schema_strips_any_citation_markers_from_structured() -> None:
    pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(data={"summary": "fact [citation:1]"}),
        ]
    )
    async with engine:
        resp = await engine.research(
            ResearchRequest(search_mode="research-fast", themes="theme x", json_schema=schema)
        )
    assert resp.structured == {"summary": "fact"}
    assert any(item.code == "research_structured_citation_removed" for item in resp.errors)


@pytest.mark.anyio
async def test_research_without_schema_outputs_markdown_with_citation_url() -> None:
    markdown = "\n\n".join(
        [
            "## 1) Core Conclusions",
            "A [citation:1]",
            "## 2) Key Findings",
            "- B [citation:1]",
            "## 3) Evidence and Citations",
            "- C [citation:1]",
            "## 4) Uncertainty and Conflicts",
            "- none",
            "## 5) Time Anchors",
            "- now",
            "## 6) Next Research Questions",
            "- next",
        ]
    )
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text=markdown),
        ]
    )
    async with engine:
        resp = await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert "[citation:https://a.example.com]" in resp.content


@pytest.mark.anyio
async def test_research_search_step_uses_search_runner_not_provider_directly() -> None:
    engine, request_log = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output("q1"),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text="## 1) Core Conclusions\nx\n\n## 2) Key Findings\ny\n\n## 3) Evidence and Citations\n- [citation:1]\n\n## 4) Uncertainty and Conflicts\nn\n\n## 5) Time Anchors\nt\n\n## 6) Next Research Questions\nq"),
        ],
        search_map={"q1": [_fetch_item("https://a.example.com")]},
    )
    async with engine:
        await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert len(request_log) >= 1
    assert all(isinstance(item, SearchRequest) for item in request_log)


@pytest.mark.anyio
async def test_research_forces_subpages_abstracts_content_in_each_search_job() -> None:
    engine, request_log = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output("q1"),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text="## 1) Core Conclusions\nx\n\n## 2) Key Findings\ny\n\n## 3) Evidence and Citations\n- [citation:1]\n\n## 4) Uncertainty and Conflicts\nn\n\n## 5) Time Anchors\nt\n\n## 6) Next Research Questions\nq"),
        ]
    )
    async with engine:
        await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))

    assert request_log
    for req in request_log:
        assert isinstance(req.fetchs.content, object)
        assert not isinstance(req.fetchs.content, bool)
        assert getattr(req.fetchs.content, "detail", "") == "full"
        assert not isinstance(req.fetchs.abstracts, bool)
        assert req.fetchs.abstracts is not None
        assert getattr(req.fetchs.abstracts, "query", "") == "theme x"
        assert req.fetchs.subpages is not None


@pytest.mark.anyio
async def test_research_round_continue_then_stop_via_multi_signal_gate() -> None:
    engine, request_log = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output("q1"),
            _abstract_output(stop=False, confidence=0.55, covered=[], next_queries=["q2"]),
            _plan_output("q2"),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text="## 1) Core Conclusions\nx [citation:1]\n\n## 2) Key Findings\ny\n\n## 3) Evidence and Citations\n- [citation:1]\n\n## 4) Uncertainty and Conflicts\nn\n\n## 5) Time Anchors\nt\n\n## 6) Next Research Questions\nq"),
        ],
        search_map={
            "q1": [_fetch_item("https://a.example.com")],
            "q2": [_fetch_item("https://b.example.com")],
        },
    )
    async with engine:
        await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert len(request_log) == 2


@pytest.mark.anyio
async def test_research_no_progress_stop() -> None:
    engine, request_log = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output("q1"),
            _abstract_output(stop=False, confidence=0.40, covered=[], next_queries=["q1"]),
            ChatResult(text="## 1) Core Conclusions\nx [citation:1]\n\n## 2) Key Findings\ny\n\n## 3) Evidence and Citations\n- [citation:1]\n\n## 4) Uncertainty and Conflicts\nn\n\n## 5) Time Anchors\nt\n\n## 6) Next Research Questions\nq"),
        ],
        search_map={"q1": []},
        no_progress_rounds_to_stop=1,
    )
    async with engine:
        resp = await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert len(request_log) == 1
    assert any(item.code == "research_round_plan_failed" for item in resp.errors) is False


@pytest.mark.anyio
async def test_research_invalid_citation_index_records_error_and_removes_marker() -> None:
    markdown = "\n\n".join(
        [
            "## 1) Core Conclusions",
            "A [citation:99]",
            "## 2) Key Findings",
            "- B",
            "## 3) Evidence and Citations",
            "- C [citation:99]",
            "## 4) Uncertainty and Conflicts",
            "- none",
            "## 5) Time Anchors",
            "- now",
            "## 6) Next Research Questions",
            "- next",
        ]
    )
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text=markdown),
        ]
    )
    async with engine:
        resp = await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert "[citation:99]" not in resp.content
    assert any(item.code == "research_invalid_citation" for item in resp.errors)


@pytest.mark.anyio
async def test_research_language_follows_theme_llm_detection() -> None:
    chinese_markdown = "\n\n".join(
        [
            "## 1) 核心结论",
            "A [citation:1]",
            "## 2) 关键发现",
            "- B",
            "## 3) 证据与引用",
            "- C [citation:1]",
            "## 4) 不确定性与冲突",
            "- 无",
            "## 5) 时间锚点",
            "- 现在",
            "## 6) 后续研究问题",
            "- 下一步",
        ]
    )
    engine, _ = _build_engine(
        llm_outputs=[
            _theme_output(detected_input_language="zh"),
            _plan_output(),
            _abstract_output(stop=True, confidence=0.95, covered=["subtheme-a"], next_queries=[]),
            ChatResult(text=chinese_markdown),
        ]
    )
    async with engine:
        resp = await engine.research(ResearchRequest(search_mode="research-fast", themes="theme x"))
    assert "## 1) 核心结论" in resp.content
    assert "[citation:https://a.example.com]" in resp.content


def test_research_total_stepbase_classes_is_10() -> None:
    import serpsage.steps.research as module

    classes = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, StepBase):
            continue
        if obj.__module__.startswith("serpsage.steps.research"):
            classes.append(obj)
    unique = {id(item) for item in classes}
    assert len(unique) == 10
