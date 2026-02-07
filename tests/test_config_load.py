from __future__ import annotations

import json

import pytest

from search_core.config import SearchConfig


def _write_json(path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_load_precedence_explicit_path_over_env(tmp_path, monkeypatch):
    file_path = _write_json(
        tmp_path / "a.json",
        {"searxng": {"base_url": "https://file.example/search"}},
    )
    env_path = _write_json(
        tmp_path / "b.json",
        {"searxng": {"base_url": "https://env.example/search"}},
    )

    monkeypatch.setenv("SEARCH_CONFIG_PATH", env_path)
    cfg = SearchConfig.load(path=file_path)
    assert cfg.searxng.base_url == "https://file.example/search"


def test_load_env_path(tmp_path, monkeypatch):
    env_path = _write_json(
        tmp_path / "b.json",
        {"searxng": {"base_url": "https://env.example/search"}},
    )
    monkeypatch.setenv("SEARCH_CONFIG_PATH", env_path)
    cfg = SearchConfig.load()
    assert cfg.searxng.base_url == "https://env.example/search"


def test_load_missing_default_returns_default_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEARCH_CONFIG_PATH", raising=False)
    cfg = SearchConfig.load(env={})
    assert isinstance(cfg, SearchConfig)


def test_apply_env_overrides(tmp_path):
    file_path = _write_json(
        tmp_path / "a.json",
        {
            "searxng": {
                "base_url": "https://file.example/search",
                "search_api_key": "filekey",
            }
        },
    )
    cfg = SearchConfig.load(
        path=file_path,
        env={
            "SEARXNG_BASE_URL": "https://override.example/search",
            "SEARCH_API_KEY": "overridekey",
        },
    )
    assert cfg.searxng.base_url == "https://override.example/search"
    assert cfg.searxng.search_api_key == "overridekey"


def test_load_explicit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SearchConfig.load(path=str(tmp_path / "missing.json"), env={})
