from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.settings.models import AppSettings


def test_research_mode_defaults_exist() -> None:
    settings = AppSettings()
    assert settings.research.research_fast.max_rounds == 3
    assert settings.research.research_fast.max_search_calls == 6
    assert settings.research.research_fast.max_fetch_calls == 12
    assert settings.research.research_fast.stop_confidence == 0.72
    assert settings.research.research_fast.min_coverage_ratio == 0.70
    assert settings.research.research_fast.max_unresolved_conflicts == 1
    assert settings.research.research.max_rounds == 5
    assert settings.research.research.max_search_calls == 12
    assert settings.research.research.max_fetch_calls == 24
    assert settings.research.research.stop_confidence == 0.80
    assert settings.research.research.min_coverage_ratio == 0.80
    assert settings.research.research.max_unresolved_conflicts == 1
    assert settings.research.research_pro.max_rounds == 8
    assert settings.research.research_pro.max_search_calls == 24
    assert settings.research.research_pro.max_fetch_calls == 48
    assert settings.research.research_pro.stop_confidence == 0.86
    assert settings.research.research_pro.min_coverage_ratio == 0.90
    assert settings.research.research_pro.max_unresolved_conflicts == 0
    assert settings.research.tool_max_attempts > 0
    assert settings.research.no_progress_rounds_to_stop == 2


def test_research_mode_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        AppSettings(research={"research_fast": {"max_rounds": 0}})
    with pytest.raises(ValidationError):
        AppSettings(research={"no_progress_rounds_to_stop": 0})
    with pytest.raises(ValidationError):
        AppSettings(research={"research_fast": {"stop_confidence": 1.5}})
    with pytest.raises(ValidationError):
        AppSettings(research={"research_fast": {"min_coverage_ratio": -0.1}})
    with pytest.raises(ValidationError):
        AppSettings(research={"research_fast": {"max_unresolved_conflicts": -1}})


def test_research_model_link_validation_for_optional_stage_model() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            llm={"models": [{"name": "only-one"}]},
            fetch={"overview": {"use_model": "only-one"}},
            answer={"plan": {"use_model": "only-one"}, "generate": {"use_model": "only-one"}},
            research={"plan": {"use_model": "missing-model"}},
        )
