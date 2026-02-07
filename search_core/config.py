from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_BASE_URL = "https://searxng.lycoreco.dpdns.org/search"
DEFAULT_USER_AGENT = "searxng-bot/0.1"
DEFAULT_CONFIG_PATH = "search_config.yaml"
DEFAULT_NOISE_EXTENSIONS = ("txt", "dic", "pdf", "zip", "rar", "7z")
RANKING_STRATEGIES = {"heuristic", "bm25", "hybrid"}


class ConfigBaseModel(BaseModel):
    """Base model for configuration objects."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load configuration from a JSON/YAML file."""

        data = load_config_data(Path(path))
        return cls.model_validate(data)

    def with_overrides(self, **kwargs: Any) -> Self:
        """Return a copy with overrides applied."""

        return self.model_copy(update=kwargs)


def load_config_data(path: Path) -> dict[str, Any]:
    """Load configuration data from JSON or YAML."""

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(raw_text)
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # pyright: ignore[reportMissingModuleSource] # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to load YAML config files.") from exc
        return yaml.safe_load(raw_text) or {}
    raise ValueError(f"Unsupported config file type: {suffix}")


class SearxngConfig(ConfigBaseModel):
    """Configuration for SearxNG HTTP requests."""

    base_url: str = DEFAULT_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 20.0
    allow_redirects: bool = False
    search_api_key: str | None = None
    extra_headers: dict[str, str] | None = None

    def build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the request."""

        headers: dict[str, str] = {"User-Agent": self.user_agent}
        if self.search_api_key:
            headers["Authorization"] = f"Bearer {self.search_api_key}"
        if self.extra_headers:
            headers.update(dict(self.extra_headers))
        return headers


class RankingConfig(ConfigBaseModel):
    """Ranking configuration."""

    strategy: str = "heuristic"
    bm25_weight: float = 0.7
    heuristic_weight: float = 0.3
    min_relevance_score: int = 10
    min_intent_score: int = 14

    def normalized_weights(self) -> tuple[float, float]:
        """Normalize weights to sum to 1."""

        total = self.bm25_weight + self.heuristic_weight
        if total <= 0:
            return (0.5, 0.5)
        return (self.bm25_weight / total, self.heuristic_weight / total)


class AutoMatchConfig(ConfigBaseModel):
    """Automatic profile matching configuration."""

    enabled: bool = False
    keywords: tuple[str, ...] = Field(default_factory=tuple)
    regex: tuple[str, ...] = Field(default_factory=tuple)
    priority: int = 0


class SearchContextConfig(ConfigBaseModel):
    """Configuration for ranking and filtering search context."""

    fuzzy_threshold: float = 0.88
    auto_match: AutoMatchConfig = Field(default_factory=AutoMatchConfig)
    intent_terms: tuple[str, ...] = Field(default_factory=tuple)
    noise_words: tuple[str, ...] = Field(default_factory=tuple)
    noise_extensions: tuple[str, ...] = DEFAULT_NOISE_EXTENSIONS
    domain_bonus: dict[str, int] = Field(default_factory=dict)
    domain_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    title_tail_patterns: tuple[str, ...] = Field(default_factory=tuple)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


class SearchConfig(ConfigBaseModel):
    """Config file model (single source of truth)."""

    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    default_profile: str = "general"
    profiles: dict[str, SearchContextConfig] = Field(default_factory=dict)

    def get_profile(self, name: str) -> SearchContextConfig | None:
        return self.profiles.get(name)

    def has_profile(self, name: str) -> bool:
        return name in self.profiles

    def resolve_profile_name(self, name: str | None) -> str:
        return name or self.default_profile

    def apply_env_overrides(
        self, *, env: Mapping[str, str] = os.environ
    ) -> SearchConfig:
        """Return a copy with environment variable overrides applied (searxng only)."""

        overrides: dict[str, object] = {}
        env_base_url = env.get("SEARXNG_BASE_URL") if env else None
        env_api_key = env.get("SEARCH_API_KEY") if env else None
        if env_base_url:
            overrides["searxng"] = self.searxng.with_overrides(base_url=env_base_url)
        if env_api_key:
            searxng = overrides.get("searxng", self.searxng)
            assert isinstance(searxng, SearxngConfig)
            overrides["searxng"] = searxng.with_overrides(search_api_key=env_api_key)
        if not overrides:
            return self
        return self.with_overrides(**overrides)

    @classmethod
    def load(
        cls,
        path: str | None = None,
        *,
        env: Mapping[str, str] = os.environ,
    ) -> SearchConfig:
        """Load SearchConfig from path/env/default, then apply env overrides.

        Precedence:
        1) explicit `path`
        2) `SEARCH_CONFIG_PATH`
        3) `DEFAULT_CONFIG_PATH`
        """

        candidate = path or env.get("SEARCH_CONFIG_PATH") or DEFAULT_CONFIG_PATH
        candidate_path = Path(candidate)

        if path:
            if not candidate_path.is_file():
                raise FileNotFoundError(f"Config file not found: {candidate_path}")
            return cls.from_file(str(candidate_path)).apply_env_overrides(env=env)

        if candidate_path.is_file():
            return cls.from_file(str(candidate_path)).apply_env_overrides(env=env)

        return cls().apply_env_overrides(env=env)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_USER_AGENT",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_NOISE_EXTENSIONS",
    "RANKING_STRATEGIES",
    "SearxngConfig",
    "RankingConfig",
    "AutoMatchConfig",
    "SearchContextConfig",
    "SearchConfig",
]
