from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


DepthKey = Literal["low", "medium", "high"]
ProviderBackendKey = Literal["searxng"]
FetchBackendKey = Literal["curl_cffi", "playwright", "auto"]
RankBackendKey = Literal["blend", "heuristic", "bm25"]
RankBlendProviderKey = Literal["heuristic", "bm25"]
CacheBackendKey = Literal["sqlite", "memory", "redis", "mysql", "sqlalchemy"]
CacheMySQLDriverKey = Literal["auto", "asyncmy", "aiomysql"]
OverviewModelBackendKey = Literal["openai", "gemini"]


class RetrySettings(Model):
    max_attempts: int = 3
    delay_ms: int = 200


class SearxngSettings(Model):
    base_url: str = "https://searxng.lycoreco.dpdns.org/search"
    api_key: str | None = None
    timeout_s: float = 20.0
    allow_redirects: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)


class HttpSettings(Model):
    proxy: str | None = None
    trust_env: bool = False
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry_s: float = 5.0


class ProviderSettings(Model):
    backend: ProviderBackendKey = "searxng"
    searxng: SearxngSettings = Field(default_factory=SearxngSettings)


class SearchDepthProfile(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    pages_ratio: float = 0.25
    min_pages: int = 1
    max_pages: int = 3
    top_abstracts_per_page: int = 2
    step_timeout_s: float = 2.0
    page_timeout_s: float = 1.6


def _default_search_depth_profiles() -> dict[DepthKey, SearchDepthProfile]:
    return {
        "low": SearchDepthProfile(
            pages_ratio=0.25,
            min_pages=1,
            max_pages=3,
            top_abstracts_per_page=2,
            step_timeout_s=1.2,
            page_timeout_s=0.9,
        ),
        "medium": SearchDepthProfile(
            pages_ratio=0.50,
            min_pages=2,
            max_pages=6,
            top_abstracts_per_page=3,
            step_timeout_s=2.0,
            page_timeout_s=1.6,
        ),
        "high": SearchDepthProfile(
            pages_ratio=0.75,
            min_pages=3,
            max_pages=10,
            top_abstracts_per_page=5,
            step_timeout_s=4.0,
            page_timeout_s=2.5,
        ),
    }


def _default_rank_blend_providers() -> dict[RankBlendProviderKey, float]:
    return {"heuristic": 1.0}


class OverviewProfileBase(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_model: str = "gpt-4.1-mini"
    max_abstract_chars: int = 900
    max_output_tokens: int = 600
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: Literal["auto", "zh", "en"] = "auto"


class SearchOverviewSettings(OverviewProfileBase):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    max_sources: int = 8
    max_abstracts_per_source: int = 2


class FetchOverviewSettings(OverviewProfileBase):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False
    max_abstracts: int = 6


class SearchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_results: int = 16
    min_score: float = 0.3
    fuzzy_threshold: float = 0.88
    include_raw: bool = False
    depth_profiles: dict[DepthKey, SearchDepthProfile] = Field(
        default_factory=_default_search_depth_profiles
    )
    overview: SearchOverviewSettings = Field(default_factory=SearchOverviewSettings)


def _default_blocked_markers() -> list[str]:
    return [
        "cloudflare",
        "just a moment",
        "verify you are human",
        "access denied",
        "please enable javascript",
        "security check",
        "checking your browser",
    ]


class FetchConcurrencySettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    global_limit: int = 24
    per_host: int = 4
    politeness_delay_ms: int = 0


class FetchRenderSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    js_concurrency: int = 4
    nav_timeout_ms: int = 2_500
    wait_network_idle_ms: int = 220
    block_resources: bool = True


class FetchQualitySettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    min_text_chars: int = 220
    script_ratio_threshold: float = 0.35
    blocked_markers: list[str] = Field(default_factory=_default_blocked_markers)


class FetchExtractSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_markdown_chars: int = 160_000
    min_text_chars: int = 220
    min_primary_chars: int = 220
    min_total_chars_with_secondary: int = 220
    include_secondary_content_default: bool = False
    collect_links_default: bool = False
    link_max_count: int = 800
    link_keep_hash: bool = False


class FetchAbstractSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_abstracts: int = 42
    min_abstract_score: float = 0.20
    min_query_token_hits: int = 2
    default_top_k_abstracts: int = 3
    max_markdown_chars: int = 140_000
    max_segments: int = 420
    min_abstract_chars: int = 1
    query_prefilter_window: int = 320
    title_boost_alpha: float = 0.35


class FetchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    backend: FetchBackendKey = "auto"
    timeout_s: float = 2.0
    follow_redirects: bool = True
    user_agent: str = "serpsage-bot/4.0"
    concurrency: FetchConcurrencySettings = Field(
        default_factory=FetchConcurrencySettings
    )
    render: FetchRenderSettings = Field(default_factory=FetchRenderSettings)
    quality: FetchQualitySettings = Field(default_factory=FetchQualitySettings)
    extract: FetchExtractSettings = Field(default_factory=FetchExtractSettings)
    abstract: FetchAbstractSettings = Field(default_factory=FetchAbstractSettings)
    overview: FetchOverviewSettings = Field(default_factory=FetchOverviewSettings)


class HeuristicRankSettings(Model):
    early_bonus: float = 1.15
    unique_hit_weight: float = 6.0
    count_weight: float = 1.5
    intent_hit_weight: float = 5.0
    max_count_per_token: int = 5
    temperature: float = 1.0
    min_items_for_sigmoid: int = 5
    flat_spread_eps: float = 1e-9
    z_clip: float = 8.0


class RankBm25Settings(Model):
    pass


class RankBlendSettings(Model):
    providers: dict[RankBlendProviderKey, float] = Field(
        default_factory=_default_rank_blend_providers
    )


class RankSettings(Model):
    backend: RankBackendKey = "blend"
    blend: RankBlendSettings = Field(default_factory=RankBlendSettings)
    heuristic: HeuristicRankSettings = Field(default_factory=HeuristicRankSettings)
    bm25: RankBm25Settings = Field(default_factory=RankBm25Settings)


class CacheSqliteSettings(Model):
    db_path: str = ".serpsage_cache.sqlite3"
    table: str = "cache"


class CacheRedisSettings(Model):
    url: str = "redis://127.0.0.1:6379/0"
    key_prefix: str = "serpsage"


class CacheMySQLSettings(Model):
    driver: CacheMySQLDriverKey = "auto"
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "serpsage"
    table: str = "cache"
    minsize: int = 1
    maxsize: int = 10
    connect_timeout: float = 10.0
    charset: str = "utf8mb4"


class CacheSQLAlchemySettings(Model):
    url: str = "sqlite+aiosqlite:///./.serpsage_cache.sqlite3"
    table: str = "cache"


class CacheSettings(Model):
    enabled: bool = False
    backend: CacheBackendKey = "sqlite"
    search_ttl_s: int = 600
    fetch_ttl_s: int = 86_400
    sqlite: CacheSqliteSettings = Field(default_factory=CacheSqliteSettings)
    redis: CacheRedisSettings = Field(default_factory=CacheRedisSettings)
    mysql: CacheMySQLSettings = Field(default_factory=CacheMySQLSettings)
    sqlalchemy: CacheSQLAlchemySettings = Field(default_factory=CacheSQLAlchemySettings)


class OverviewModelSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = "gpt-4.1-mini"
    backend: OverviewModelBackendKey = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4.1-mini"
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    headers: dict[str, str] = Field(default_factory=dict)
    schema_strict: bool = True


def _default_overview_models() -> list[OverviewModelSettings]:
    return [OverviewModelSettings()]


class LLMSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    models: list[OverviewModelSettings] = Field(
        default_factory=_default_overview_models
    )

    @model_validator(mode="after")
    def _validate_models(self) -> LLMSettings:
        if not self.models:
            raise ValueError("llm.models must contain at least one model")

        names: set[str] = set()
        for idx, item in enumerate(self.models):
            if not item.name:
                raise ValueError(f"llm.models[{idx}].name must be non-empty")
            if item.name in names:
                raise ValueError(
                    f"duplicate llm model name `{item.name}` in llm.models"
                )
            names.add(item.name)
        return self

    def resolve_model(self, name: str) -> OverviewModelSettings:
        for item in self.models:
            if item.name == name:
                return item
        raise ValueError(f"llm model `{name}` does not exist in llm.models")


class TelemetrySettings(Model):
    enabled: bool = False
    include_events: bool = False


class AppSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    http: HttpSettings = Field(default_factory=HttpSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    rank: RankSettings = Field(default_factory=RankSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    @model_validator(mode="after")
    def _validate_model_links(self) -> AppSettings:
        names = {m.name for m in self.llm.models}
        if self.search.overview.use_model not in names:
            raise ValueError(
                "search.overview.use_model must match one of llm.models[].name"
            )
        if self.fetch.overview.use_model not in names:
            raise ValueError(
                "fetch.overview.use_model must match one of llm.models[].name"
            )
        return self


__all__ = [
    "AppSettings",
    "CacheMySQLSettings",
    "CacheRedisSettings",
    "CacheSQLAlchemySettings",
    "CacheSettings",
    "DepthKey",
    "FetchBackendKey",
    "FetchAbstractSettings",
    "FetchConcurrencySettings",
    "FetchExtractSettings",
    "FetchOverviewSettings",
    "FetchQualitySettings",
    "FetchRenderSettings",
    "FetchSettings",
    "HttpSettings",
    "HeuristicRankSettings",
    "LLMSettings",
    "OverviewModelBackendKey",
    "OverviewModelSettings",
    "ProviderSettings",
    "SearchDepthProfile",
    "SearchOverviewSettings",
    "SearchSettings",
    "RankBlendSettings",
    "RankSettings",
    "RetrySettings",
    "SearxngSettings",
    "TelemetrySettings",
]
