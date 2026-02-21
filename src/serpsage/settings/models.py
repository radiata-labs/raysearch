from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


ProviderBackendKey = Literal["searxng"]
FetchBackendKey = Literal["curl_cffi", "playwright", "auto"]
RankBackendKey = Literal["blend", "heuristic", "bm25"]
RankBlendProviderKey = Literal["heuristic", "bm25"]
CacheBackendKey = Literal["sqlite", "memory", "redis", "mysql", "sqlalchemy"]
CacheMySQLDriverKey = Literal["auto", "asyncmy", "aiomysql"]
OverviewModelBackendKey = Literal["openai", "gemini", "dashscope"]


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


def _default_rank_blend_providers() -> dict[RankBlendProviderKey, float]:
    return {"heuristic": 1.0}


class OverviewProfileBase(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_model: str = "gpt-4.1-mini"
    max_abstract_chars: int = 900
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: Literal["auto", "zh", "en"] = "auto"


class FetchOverviewSettings(OverviewProfileBase):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SearchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_results: int = 16
    additional_query_score_weight: float = 0.8

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchSettings:
        if int(self.max_results) <= 0:
            raise ValueError("search.max_results must be > 0")
        if float(self.additional_query_score_weight) <= 0:
            raise ValueError("search.additional_query_score_weight must be > 0")
        return self


class AnswerStageSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    use_model: str = "gpt-4.1-mini"


class AnswerGenerateSettings(AnswerStageSettings):
    max_abstract_chars: int = 3000

    @field_validator("max_abstract_chars")
    @classmethod
    def _validate_max_abstract_chars(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("answer.generate.max_abstract_chars must be > 0")
        return int(value)


class AnswerSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    plan: AnswerStageSettings = Field(default_factory=AnswerStageSettings)
    generate: AnswerGenerateSettings = Field(default_factory=AnswerGenerateSettings)


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
    nav_timeout_ms: int = 8_000
    wait_network_idle_ms: int = 800
    block_resources: bool = True

    @field_validator("js_concurrency")
    @classmethod
    def _validate_js_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("js_concurrency must be >= 1")
        return value


class FetchQualitySettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    min_text_chars: int = 100
    script_ratio_threshold: float = 0.35
    quality_score_threshold: float = 0.15
    blocked_markers: list[str] = Field(default_factory=_default_blocked_markers)

    @field_validator("min_text_chars")
    @classmethod
    def _validate_min_text_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("min_text_chars must be >= 0")
        return value

    @field_validator("script_ratio_threshold")
    @classmethod
    def _validate_script_ratio_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("script_ratio_threshold must be between 0 and 1")
        return value

    @field_validator("quality_score_threshold")
    @classmethod
    def _validate_quality_score_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("quality_score_threshold must be between 0 and 1")
        return value


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

    max_abstract_chars: int = 2000
    min_abstract_score: float = 0.20
    min_abstract_tokens: int = 4
    title_boost_alpha: float = 0.35


class FetchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    backend: FetchBackendKey = "auto"
    inflight_enabled: bool = True
    inflight_timeout_s: float = 60.0
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

    @field_validator("inflight_timeout_s")
    @classmethod
    def _validate_inflight_timeout_s(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError("inflight_timeout_s must be > 0")
        return float(value)


class RunnerSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    search_limit: int = Field(default=8, ge=1)
    fetch_limit: int = Field(default=24, ge=1)
    child_fetch_limit: int = Field(default=24, ge=1)
    queue_size: int = Field(default=256, ge=1)


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
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    rank: RankSettings = Field(default_factory=RankSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    @model_validator(mode="after")
    def _validate_model_links(self) -> AppSettings:
        names = {m.name for m in self.llm.models}
        if self.fetch.overview.use_model not in names:
            raise ValueError(
                "fetch.overview.use_model must match one of llm.models[].name"
            )
        if self.answer.plan.use_model not in names:
            raise ValueError(
                "answer.plan.use_model must match one of llm.models[].name"
            )
        if self.answer.generate.use_model not in names:
            raise ValueError(
                "answer.generate.use_model must match one of llm.models[].name"
            )
        return self


__all__ = [
    "AppSettings",
    "AnswerGenerateSettings",
    "AnswerSettings",
    "AnswerStageSettings",
    "CacheMySQLSettings",
    "CacheRedisSettings",
    "CacheSQLAlchemySettings",
    "CacheSettings",
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
    "RunnerSettings",
    "SearchSettings",
    "RankBlendSettings",
    "RankSettings",
    "RetrySettings",
    "SearxngSettings",
    "TelemetrySettings",
]
