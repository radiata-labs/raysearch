from __future__ import annotations

import pytest

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline


def test_explicit_profile_missing_raises():
    cfg = SearchConfig(
        default_profile="general",
        profiles={"general": SearchContextConfig()},
    )
    pipeline = SearchPipeline(cfg)
    with pytest.raises(ValueError, match="Profile not found"):
        pipeline._select_profile_name("hello", "missing")  # noqa: SLF001


def test_auto_match_selects_best_profile():
    cfg = SearchConfig(
        default_profile="general",
        profiles={
            "general": SearchContextConfig(),
            "music": SearchContextConfig.model_validate(
                {
                    "auto_match": {
                        "enabled": True,
                        "keywords": ("song",),
                        "regex": (),
                        "priority": 10,
                    }
                }
            ),
        },
    )
    pipeline = SearchPipeline(cfg)
    selected = pipeline._select_profile_name("new song 2025", None)  # noqa: SLF001
    assert selected == "music"


def test_fallback_to_default_profile_name():
    cfg = SearchConfig(
        default_profile="general",
        profiles={
            "general": SearchContextConfig.model_validate(
                {"auto_match": {"enabled": False}}
            )
        },
    )
    pipeline = SearchPipeline(cfg)
    selected = pipeline._select_profile_name("no match", None)  # noqa: SLF001
    assert selected == "general"


def test_default_profile_missing_returns_default_context_config():
    cfg = SearchConfig(default_profile="general", profiles={})
    pipeline = SearchPipeline(cfg)
    profile_cfg = pipeline._get_profile_config("general")  # noqa: SLF001
    assert isinstance(profile_cfg, SearchContextConfig)
