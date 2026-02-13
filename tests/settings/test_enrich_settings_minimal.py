from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.settings.models import AppSettings


def _overview_block() -> dict[str, object]:
    return {
        "use_model": "gpt-4.1-mini",
        "models": [{"name": "gpt-4.1-mini", "backend": "openai"}],
    }


def test_enrich_accepts_minimal_public_shape() -> None:
    settings = AppSettings.model_validate(
        {
            "overview": _overview_block(),
            "enrich": {
                "enabled": True,
                "fetch": {"backend": "auto"},
                "depth_presets": {
                    "low": {
                        "pages_ratio": 0.25,
                        "min_pages": 1,
                        "max_pages": 3,
                        "top_chunks_per_page": 2,
                    },
                    "medium": {
                        "pages_ratio": 0.50,
                        "min_pages": 2,
                        "max_pages": 6,
                        "top_chunks_per_page": 3,
                    },
                    "high": {
                        "pages_ratio": 0.75,
                        "min_pages": 3,
                        "max_pages": 10,
                        "top_chunks_per_page": 5,
                    },
                },
            },
        }
    )
    assert settings.enrich.fetch.backend == "auto"
    assert settings.enrich.enabled is True


def test_enrich_rejects_removed_detailed_config_keys() -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "overview": _overview_block(),
                "enrich": {
                    "enabled": True,
                    "fetch": {
                        "backend": "auto",
                        "playwright": {"enabled": True},
                    },
                },
            }
        )
