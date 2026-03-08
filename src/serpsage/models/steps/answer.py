from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.models.app.request import (
    AnswerRequest,
    SearchRequest,
)
from serpsage.models.app.response import (
    AnswerCitation,
    AnswerResponse,
    FetchResultItem,
)
from serpsage.models.base import MutableModel, UnvalidatedModel
from serpsage.models.steps.base import BaseStepContext
from serpsage.settings.models import AppSettings


class AnswerPlanPayload(UnvalidatedModel):
    answer_mode: Literal["direct", "summary"]
    freshness_intent: bool
    query_language: str
    sub_questions: list[str]


class PageSource(MutableModel):
    key: str
    url: str
    title: str
    content: str
    first_order: int
    abstracts: list[str] = Field(default_factory=list)


class PromptSource(MutableModel):
    key: str
    url: str
    title: str
    content: str
    abstracts: list[str]
    question_index: int
    source_index: int


class QuestionPromptContext(MutableModel):
    question: str
    sources: list[PromptSource]


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


class AnswerStepContext(BaseStepContext[AnswerRequest, AnswerResponse]):
    settings: AppSettings
    request: AnswerRequest
    response: AnswerResponse
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    search: AnswerSearchState = Field(default_factory=AnswerSearchState)
    output: AnswerOutputState = Field(default_factory=AnswerOutputState)


__all__ = [
    "AnswerPlanPayload",
    "PageSource",
    "PromptSource",
    "QuestionPromptContext",
    "AnswerOutputState",
    "AnswerPlanState",
    "AnswerSubQuestionPlan",
    "AnswerSubSearchState",
    "AnswerSearchState",
    "AnswerStepContext",
]
