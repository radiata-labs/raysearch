from __future__ import annotations

from pydantic import Field

from serpsage.models.app.request import (
    AnswerRequest,
    SearchRequest,
)
from serpsage.models.app.response import (
    AnswerCitation,
    FetchResultItem,
)
from serpsage.models.base import MutableModel
from serpsage.models.steps.base import BaseStepContext
from serpsage.settings.models import AppSettings


class AnswerSubQuestionPlan(MutableModel):
    question: str = ""
    search_query: str = ""


class AnswerPlanState(MutableModel):
    answer_mode: str = "summary"
    freshness_intent: bool = False
    query_language: str = "same as query"
    search_query: str = ""
    search_mode: str = "auto"
    max_results: int = 1
    additional_queries: list[str] | None = None
    sub_questions: list[AnswerSubQuestionPlan] = Field(default_factory=list)


class AnswerSubSearchState(MutableModel):
    question: str = ""
    search_query: str = ""
    request: SearchRequest | None = None
    search_mode: str = "auto"
    results: list[FetchResultItem] = Field(default_factory=list)


class AnswerSearchState(MutableModel):
    request: SearchRequest | None = None
    search_mode: str = "auto"
    results: list[FetchResultItem] = Field(default_factory=list)
    sub_searches: list[AnswerSubSearchState] = Field(default_factory=list)


class AnswerOutputState(MutableModel):
    answers: str | object = ""
    citations: list[AnswerCitation] = Field(default_factory=list)


class AnswerStepContext(BaseStepContext):
    settings: AppSettings
    request: AnswerRequest
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    search: AnswerSearchState = Field(default_factory=AnswerSearchState)
    output: AnswerOutputState = Field(default_factory=AnswerOutputState)


__all__ = [
    "AnswerOutputState",
    "AnswerPlanState",
    "AnswerSubQuestionPlan",
    "AnswerSubSearchState",
    "AnswerSearchState",
    "AnswerStepContext",
]
