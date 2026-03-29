from __future__ import annotations

from pydantic import Field

from raysearch.models.app.request import AnswerRequest, SearchRequest
from raysearch.models.app.response import AnswerCitation, AnswerResponse, FetchResultItem
from raysearch.models.base import MutableModel
from raysearch.models.steps.base import BaseStepContext
from raysearch.models.steps.search import QuerySourceSpec


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
    search_query: QuerySourceSpec | None = None


class AnswerPlanState(MutableModel):
    answer_mode: str = "summary"
    freshness_intent: bool = False
    query_language: str = "same as query"
    search_query: QuerySourceSpec | None = None
    search_mode: str = "auto"
    max_results: int = 1
    additional_queries: list[str] | None = None
    sub_questions: list[AnswerSubQuestionPlan] = Field(default_factory=list)


class AnswerSubSearchState(MutableModel):
    question: str = ""
    search_query: QuerySourceSpec | None = None
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
    request: AnswerRequest
    response: AnswerResponse
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    search: AnswerSearchState = Field(default_factory=AnswerSearchState)
    output: AnswerOutputState = Field(default_factory=AnswerOutputState)


__all__ = [
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
