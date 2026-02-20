from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage import AnswerRequest, AnswerResponse
from serpsage.settings.models import AppSettings


def test_answer_settings_defaults_work_without_answer_node() -> None:
    settings = AppSettings()
    assert settings.answer.plan.use_model == "gpt-4.1-mini"
    assert settings.answer.generate.use_model == "gpt-4.1-mini"
    assert settings.answer.generate.max_abstract_chars == 3000


def test_answer_settings_model_links_must_exist() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            llm={"models": [{"name": "only-one"}]},
            fetch={"overview": {"use_model": "only-one"}},
            answer={
                "plan": {"use_model": "only-one"},
                "generate": {"use_model": "missing-model"},
            },
        )


def test_public_exports_include_answer_models() -> None:
    req = AnswerRequest(query="what is this")
    resp = AnswerResponse(request_id="x", answer="", citations=[], errors=[])
    assert req.query == "what is this"
    assert resp.request_id == "x"


def test_answer_request_query_validation() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(query="   ")


def test_answer_request_text_field_is_not_supported() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(query="q", text=True)  # type: ignore[call-arg]


def test_answer_generate_max_abstract_chars_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AppSettings(answer={"generate": {"max_abstract_chars": 0}})
