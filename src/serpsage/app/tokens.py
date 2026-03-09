from __future__ import annotations

from serpsage.dependencies import InjectToken
from serpsage.models.steps.answer import AnswerStepContext
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.models.steps.research import ResearchStepContext
from serpsage.models.steps.search import SearchStepContext
from serpsage.steps.base import RunnerBase, StepBase

SEARCH_RUNNER = InjectToken[RunnerBase[SearchStepContext]]("app.search_runner")
FETCH_RUNNER = InjectToken[RunnerBase[FetchStepContext]]("app.fetch_runner")
CHILD_FETCH_RUNNER = InjectToken[RunnerBase[FetchStepContext]]("app.child_fetch_runner")
ANSWER_RUNNER = InjectToken[RunnerBase[AnswerStepContext]]("app.answer_runner")
RESEARCH_ROUND_RUNNER = InjectToken[RunnerBase[ResearchStepContext]](
    "app.research_round_runner"
)
RESEARCH_RUNNER = InjectToken[RunnerBase[ResearchStepContext]]("app.research_runner")
RESEARCH_SUBREPORT_STEP = InjectToken[StepBase[ResearchStepContext]](
    "app.research_subreport_step"
)

CHILD_FETCH_STEPS = InjectToken[tuple[StepBase[FetchStepContext], ...]](
    "app.child_fetch_steps"
)
FETCH_STEPS = InjectToken[tuple[StepBase[FetchStepContext], ...]]("app.fetch_steps")
SEARCH_STEPS = InjectToken[tuple[StepBase[SearchStepContext], ...]]("app.search_steps")
ANSWER_STEPS = InjectToken[tuple[StepBase[AnswerStepContext], ...]]("app.answer_steps")
RESEARCH_ROUND_STEPS = InjectToken[tuple[StepBase[ResearchStepContext], ...]](
    "app.research_round_steps"
)
RESEARCH_STEPS = InjectToken[tuple[StepBase[ResearchStepContext], ...]](
    "app.research_steps"
)

__all__ = [
    "ANSWER_RUNNER",
    "ANSWER_STEPS",
    "CHILD_FETCH_RUNNER",
    "CHILD_FETCH_STEPS",
    "FETCH_RUNNER",
    "FETCH_STEPS",
    "RESEARCH_ROUND_RUNNER",
    "RESEARCH_ROUND_STEPS",
    "RESEARCH_RUNNER",
    "RESEARCH_STEPS",
    "RESEARCH_SUBREPORT_STEP",
    "SEARCH_RUNNER",
    "SEARCH_STEPS",
]
