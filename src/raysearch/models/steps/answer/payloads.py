from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from raysearch.models.base import UnvalidatedModel
from raysearch.models.steps.search import QuerySourceSpec


class AnswerSubQuestionPayload(UnvalidatedModel):
    question: str
    search_query: QuerySourceSpec

    @model_validator(mode="before")
    @classmethod
    def _coerce_raw(cls, value: object) -> object:
        if isinstance(value, str):
            return {"question": value, "search_query": value}
        return value


class AnswerPlanPayload(UnvalidatedModel):
    answer_mode: Literal["direct", "summary"]
    freshness_intent: bool
    query_language: str
    sub_questions: list[AnswerSubQuestionPayload]


__all__ = [
    "AnswerPlanPayload",
    "AnswerSubQuestionPayload",
]
