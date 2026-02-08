from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_BASE_URL = "https://searxng.lycoreco.dpdns.org/search"
DEFAULT_USER_AGENT = "searxng-bot/0.1"
DEFAULT_CONFIG_PATH = "search_config.yaml"
DEFAULT_NOISE_EXTENSIONS = ("txt", "dic", "pdf", "zip", "rar", "7z")


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


class HeuristicScoringConfig(ConfigBaseModel):
    """Generic heuristic scoring weights shared by any input text strings."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    unique_hit_weight: float = 6.0
    count_weight: float = 1.5
    intent_hit_weight: float = 5.0
    phrase_bonus: float = 8.0
    min_token_len: int = 2
    max_count_per_token: int = 5


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


class WebDepthPreset(ConfigBaseModel):
    pages_ratio: float = 0.25
    min_pages: int = 1
    max_pages: int = 3
    top_chunks_per_page: int = 2


class WebFetchConfig(ConfigBaseModel):
    timeout: float = 10.0
    max_bytes: int = 2_000_000
    max_extracted_chars: int = 50_000
    max_workers: int = 6
    allow_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
        "text/plain",
    )


class WebChunkingConfig(ConfigBaseModel):
    target_chars: int = 1200
    overlap_sentences: int = 1
    min_chunk_chars: int = 200
    max_sentence_chars: int = 600
    max_blocks: int = 120
    max_sentences: int = 400
    max_chunks: int = 80


class WebChunkSelectConfig(ConfigBaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    early_bonus: float = 1.15
    template_hard_drop_threshold: float = 0.95
    block_hard_drop_threshold: float = 0.90


class WebEnrichmentConfig(ConfigBaseModel):
    enabled: bool = True
    fetch: WebFetchConfig = Field(default_factory=WebFetchConfig)
    chunking: WebChunkingConfig = Field(default_factory=WebChunkingConfig)
    select: WebChunkSelectConfig = Field(default_factory=WebChunkSelectConfig)
    depth_presets: dict[Literal["low", "medium", "high"], WebDepthPreset] = Field(
        default_factory=lambda: {
            "low": WebDepthPreset(
                pages_ratio=0.25, min_pages=1, max_pages=3, top_chunks_per_page=2
            ),
            "medium": WebDepthPreset(
                pages_ratio=0.50, min_pages=2, max_pages=6, top_chunks_per_page=3
            ),
            "high": WebDepthPreset(
                pages_ratio=0.75, min_pages=3, max_pages=10, top_chunks_per_page=5
            ),
        }
    )


class ScoreNormalizationConfig(ConfigBaseModel):
    """Configuration for mapping raw scores to [0, 1] for output."""

    method: Literal["robust_sigmoid", "rank"] = "robust_sigmoid"
    temperature: float = 1.0
    min_items_for_sigmoid: int = 5
    flat_spread_eps: float = 1e-9
    z_clip: float = 8.0


class ScoringConfig(ConfigBaseModel):
    """Global scoring config: provider weights + heuristic + normalization."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Provider weights. Keys outside {"heuristic","bm25"} are ignored by ScoringEngine.
    providers: dict[Literal["heuristic", "bm25"], float] = Field(
        default_factory=lambda: {"heuristic": 1.0}
    )
    heuristic: HeuristicScoringConfig = Field(default_factory=HeuristicScoringConfig)
    normalization: ScoreNormalizationConfig = Field(
        default_factory=ScoreNormalizationConfig
    )  # type: ignore[name-defined]


class ScoreFilterConfig(ConfigBaseModel):
    """Global score floor for both results and web chunks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    min_score: float = 0.5


class SearchConfig(ConfigBaseModel):
    """Config file model (single source of truth)."""

    searxng: SearxngConfig = Field(default_factory=SearxngConfig)
    default_profile: str = "general"
    profiles: dict[str, SearchContextConfig] = Field(default_factory=dict)
    web_enrichment: WebEnrichmentConfig = Field(default_factory=WebEnrichmentConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    score_filter: ScoreFilterConfig = Field(default_factory=ScoreFilterConfig)

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
    "SearxngConfig",
    "HeuristicScoringConfig",
    "ScoringConfig",
    "ScoreFilterConfig",
    "AutoMatchConfig",
    "SearchContextConfig",
    "WebDepthPreset",
    "WebFetchConfig",
    "WebChunkingConfig",
    "WebChunkSelectConfig",
    "WebEnrichmentConfig",
    "ScoreNormalizationConfig",
    "SearchConfig",
]
