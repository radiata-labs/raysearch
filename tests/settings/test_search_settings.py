from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.settings.models import AppSettings


def test_search_deep_settings_defaults_exist() -> None:
    settings = AppSettings()
    assert settings.search.deep.enabled is True
    assert settings.search.deep.max_expanded_queries >= 0
    assert settings.search.deep.prefetch_multiplier >= 1.0
    assert (
        settings.search.deep.final_page_weight
        + settings.search.deep.final_context_weight
        + settings.search.deep.final_prefetch_weight
        == pytest.approx(1.0)
    )


def test_search_deep_settings_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        AppSettings(
            search={
                "deep": {
                    "prefetch_multiplier": 0.5,
                    "final_page_weight": 0.2,
                    "final_context_weight": 0.2,
                    "final_prefetch_weight": 0.2,
                }
            }
        )
