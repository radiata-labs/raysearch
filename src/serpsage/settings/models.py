from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DEFAULT_SEARXNG_BASE_URL = "https://searx.be/search"
_FINAL_WEIGHT_SUM_TARGET = 1.0
_WEIGHT_SUM_EPS = 1e-6


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


ProviderBackendKey = Literal["searxng"]
FetchBackendKey = Literal["curl_cffi", "playwright", "auto"]
RankBackendKey = Literal["blend", "heuristic", "tfidf", "bm25", "cross_encoder"]
RankBlendProviderKey = Literal["heuristic", "tfidf", "bm25"]
CacheBackendKey = Literal["sqlite", "memory", "redis", "mysql", "sqlalchemy"]
CacheMySQLDriverKey = Literal["auto", "asyncmy", "aiomysql"]
OverviewModelBackendKey = Literal["openai", "gemini", "dashscope"]
TelemetryObsBackendKey = Literal["null", "jsonl"]
TelemetryMeteringBackendKey = Literal["null", "sqlite"]
ReportStyleKey = Literal["decision", "explainer", "execution"]


class RetrySettings(Model):
    max_attempts: int = 3
    delay_ms: int = 200


class SearxngSettings(Model):
    base_url: str = _DEFAULT_SEARXNG_BASE_URL
    api_key: str | None = None
    timeout_s: float = 20.0
    allow_redirects: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        url = str(value or "").strip()
        if not url:
            raise ValueError("provider.searxng.base_url must be non-empty")
        if not (url.startswith(("http://", "https://"))):
            raise ValueError(
                "provider.searxng.base_url must start with http:// or https://"
            )
        return url


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
    return {"heuristic": 0.7, "tfidf": 0.3}


class OverviewProfileBase(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    use_model: str = "gpt-4.1-mini"
    max_abstract_chars: int = 900
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: Literal["auto", "zh", "en"] = "auto"


class FetchOverviewSettings(OverviewProfileBase):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SearchDeepSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enabled: bool = True
    max_expanded_queries: int = 6
    rule_max_queries: int = 4
    llm_max_queries: int = 3
    prefetch_multiplier: float = 3.0
    prefetch_max_urls: int = 48
    manual_query_score_weight: float = 0.8
    rule_query_score_weight: float = 0.75
    llm_query_score_weight: float = 0.85
    coverage_bonus_weight: float = 0.08
    final_page_weight: float = 0.55
    final_context_weight: float = 0.30
    final_prefetch_weight: float = 0.15
    expansion_model: str = ""
    expansion_timeout_s: float = 20.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchDeepSettings:
        if int(self.max_expanded_queries) < 0:
            raise ValueError("search.deep.max_expanded_queries must be >= 0")
        if int(self.rule_max_queries) < 0:
            raise ValueError("search.deep.rule_max_queries must be >= 0")
        if int(self.llm_max_queries) < 0:
            raise ValueError("search.deep.llm_max_queries must be >= 0")
        if float(self.prefetch_multiplier) < 1.0:
            raise ValueError("search.deep.prefetch_multiplier must be >= 1.0")
        if int(self.prefetch_max_urls) <= 0:
            raise ValueError("search.deep.prefetch_max_urls must be > 0")
        if float(self.manual_query_score_weight) < 0:
            raise ValueError("search.deep.manual_query_score_weight must be >= 0")
        if float(self.rule_query_score_weight) < 0:
            raise ValueError("search.deep.rule_query_score_weight must be >= 0")
        if float(self.llm_query_score_weight) < 0:
            raise ValueError("search.deep.llm_query_score_weight must be >= 0")
        if float(self.coverage_bonus_weight) < 0:
            raise ValueError("search.deep.coverage_bonus_weight must be >= 0")
        if float(self.final_page_weight) < 0:
            raise ValueError("search.deep.final_page_weight must be >= 0")
        if float(self.final_context_weight) < 0:
            raise ValueError("search.deep.final_context_weight must be >= 0")
        if float(self.final_prefetch_weight) < 0:
            raise ValueError("search.deep.final_prefetch_weight must be >= 0")
        weight_sum = (
            float(self.final_page_weight)
            + float(self.final_context_weight)
            + float(self.final_prefetch_weight)
        )
        if abs(weight_sum - _FINAL_WEIGHT_SUM_TARGET) > _WEIGHT_SUM_EPS:
            raise ValueError(
                "search.deep final weights must sum to 1.0 "
                "(final_page_weight + final_context_weight + final_prefetch_weight)"
            )
        if float(self.expansion_timeout_s) <= 0:
            raise ValueError("search.deep.expansion_timeout_s must be > 0")
        return self


class SearchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_results: int = 16
    additional_query_score_weight: float = 0.8
    deep: SearchDeepSettings = Field(default_factory=SearchDeepSettings)

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


class ResearchModelsSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    plan: str = ""
    link_select: str = ""
    abstract_analyze: str = ""
    content_analyze: str = ""
    synthesize: str = ""
    markdown: str = ""


class ResearchModeSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_rounds: int = 5
    max_search_calls: int = 12
    max_fetch_calls: int = 24
    max_results_per_search: int = 6
    max_queries_per_round: int = 3
    stop_confidence: float = 0.80
    min_coverage_ratio: float = 0.80
    max_question_cards_effective: int = 4
    min_rounds_per_track: int = 2
    round_search_budget: int = 2
    round_fetch_budget: int = 6
    review_source_window: int = 48
    report_source_batch_size: int = 8
    report_source_batch_chars: int = 48_000
    fetch_page_max_chars: int = 10_000
    explore_target_pages_per_round: int = 3
    explore_links_per_page: int = 8

    @model_validator(mode="after")
    def _validate_limits(self) -> ResearchModeSettings:
        if int(self.max_rounds) <= 0:
            raise ValueError("research mode max_rounds must be > 0")
        if int(self.max_search_calls) <= 0:
            raise ValueError("research mode max_search_calls must be > 0")
        if int(self.max_fetch_calls) <= 0:
            raise ValueError("research mode max_fetch_calls must be > 0")
        if int(self.max_results_per_search) <= 0:
            raise ValueError("research mode max_results_per_search must be > 0")
        if int(self.max_queries_per_round) <= 0:
            raise ValueError("research mode max_queries_per_round must be > 0")
        if not 0.0 <= float(self.stop_confidence) <= 1.0:
            raise ValueError("research mode stop_confidence must be between 0 and 1")
        if not 0.0 <= float(self.min_coverage_ratio) <= 1.0:
            raise ValueError("research mode min_coverage_ratio must be between 0 and 1")
        if int(self.max_question_cards_effective) <= 0:
            raise ValueError("research mode max_question_cards_effective must be > 0")
        if int(self.min_rounds_per_track) <= 0:
            raise ValueError("research mode min_rounds_per_track must be > 0")
        if int(self.round_search_budget) <= 0:
            raise ValueError("research mode round_search_budget must be > 0")
        if int(self.round_fetch_budget) <= 0:
            raise ValueError("research mode round_fetch_budget must be > 0")
        if int(self.review_source_window) <= 0:
            raise ValueError("research mode review_source_window must be > 0")
        if int(self.report_source_batch_size) <= 0:
            raise ValueError("research mode report_source_batch_size must be > 0")
        if int(self.report_source_batch_chars) <= 0:
            raise ValueError("research mode report_source_batch_chars must be > 0")
        if int(self.fetch_page_max_chars) <= 0:
            raise ValueError("research mode fetch_page_max_chars must be > 0")
        if int(self.explore_target_pages_per_round) <= 0:
            raise ValueError("research mode explore_target_pages_per_round must be > 0")
        if int(self.explore_links_per_page) <= 0:
            raise ValueError("research mode explore_links_per_page must be > 0")
        return self


def _default_research_fast_mode() -> ResearchModeSettings:
    return ResearchModeSettings(
        max_rounds=3,
        max_search_calls=6,
        max_fetch_calls=12,
        max_results_per_search=5,
        max_queries_per_round=3,
        stop_confidence=0.72,
        min_coverage_ratio=0.70,
        max_question_cards_effective=2,
        min_rounds_per_track=1,
        round_search_budget=2,
        round_fetch_budget=4,
        review_source_window=24,
        report_source_batch_size=4,
        report_source_batch_chars=24_000,
        fetch_page_max_chars=24_000,
        explore_target_pages_per_round=2,
        explore_links_per_page=4,
    )


def _default_research_standard_mode() -> ResearchModeSettings:
    return ResearchModeSettings(
        max_rounds=5,
        max_search_calls=12,
        max_fetch_calls=24,
        max_results_per_search=8,
        max_queries_per_round=5,
        stop_confidence=0.80,
        min_coverage_ratio=0.80,
        max_question_cards_effective=4,
        min_rounds_per_track=2,
        round_search_budget=3,
        round_fetch_budget=8,
        review_source_window=48,
        report_source_batch_size=8,
        report_source_batch_chars=48_000,
        fetch_page_max_chars=48_000,
        explore_target_pages_per_round=4,
        explore_links_per_page=10,
    )


def _default_research_pro_mode() -> ResearchModeSettings:
    return ResearchModeSettings(
        max_rounds=8,
        max_search_calls=24,
        max_fetch_calls=48,
        max_results_per_search=10,
        max_queries_per_round=6,
        stop_confidence=0.86,
        min_coverage_ratio=0.90,
        max_question_cards_effective=6,
        min_rounds_per_track=3,
        round_search_budget=4,
        round_fetch_budget=12,
        review_source_window=72,
        report_source_batch_size=12,
        report_source_batch_chars=72_000,
        fetch_page_max_chars=72_000,
        explore_target_pages_per_round=6,
        explore_links_per_page=16,
    )


class ResearchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    tool_max_attempts: int = 3
    llm_self_heal_retries: int = 2
    models: ResearchModelsSettings = Field(default_factory=ResearchModelsSettings)
    research_fast: ResearchModeSettings = Field(
        default_factory=_default_research_fast_mode
    )
    research: ResearchModeSettings = Field(
        default_factory=_default_research_standard_mode
    )
    research_pro: ResearchModeSettings = Field(
        default_factory=_default_research_pro_mode
    )

    @model_validator(mode="after")
    def _validate_runtime_limits(self) -> ResearchSettings:
        if int(self.tool_max_attempts) <= 0:
            raise ValueError("research.tool_max_attempts must be > 0")
        if int(self.llm_self_heal_retries) < 0:
            raise ValueError("research.llm_self_heal_retries must be >= 0")
        return self


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
    js_concurrency: int = 12
    nav_timeout_ms: int = 8_000
    block_resources: bool = True
    readiness_poll_ms: int = 150
    readiness_stable_rounds: int = 2
    post_ready_wait_ms: int = 120

    @field_validator("js_concurrency")
    @classmethod
    def _validate_js_concurrency(cls, value: int) -> int:
        if value < 1:
            raise ValueError("js_concurrency must be >= 1")
        return value

    @field_validator(
        "readiness_poll_ms", "readiness_stable_rounds", "post_ready_wait_ms"
    )
    @classmethod
    def _validate_render_timing(cls, value: int) -> int:
        if value < 0:
            raise ValueError("render timing settings must be >= 0")
        return value


class FetchAutoSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    scout_bytes: int = 48_000
    route_memory_size: int = 4096
    direct_route_min_samples: int = 3
    direct_playwright_cost_ratio: float = 0.78
    direct_playwright_min_useful: float = 0.72
    learning_rate: float = 0.22

    @field_validator("scout_bytes", "route_memory_size", "direct_route_min_samples")
    @classmethod
    def _validate_auto_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("fetch auto integer settings must be > 0")
        return value

    @field_validator(
        "direct_playwright_cost_ratio",
        "direct_playwright_min_useful",
        "learning_rate",
    )
    @classmethod
    def _validate_auto_floats(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("fetch auto float settings must be between 0 and 1")
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
    auto: FetchAutoSettings = Field(default_factory=FetchAutoSettings)
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


class RankTfidfSettings(Model):
    pass


class RankCrossEncoderSettings(Model):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = Field(default=16, ge=1)
    max_length: int = Field(default=512, ge=1)

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        model_name = str(value or "").strip()
        if not model_name:
            raise ValueError("rank.cross_encoder.model_name must be non-empty")
        return model_name


class RankBlendRerankSettings(Model):
    retrieve_weight: float = 0.35
    cross_encoder_weight: float = 0.65

    @model_validator(mode="after")
    def _validate_weights(self) -> RankBlendRerankSettings:
        if float(self.retrieve_weight) < 0:
            raise ValueError("rank.blend.rerank.retrieve_weight must be >= 0")
        if float(self.cross_encoder_weight) < 0:
            raise ValueError("rank.blend.rerank.cross_encoder_weight must be >= 0")
        if float(self.retrieve_weight) + float(self.cross_encoder_weight) <= 0:
            raise ValueError("rank.blend.rerank weights must sum to a positive value")
        return self


class RankBlendSettings(Model):
    providers: dict[RankBlendProviderKey, float] = Field(
        default_factory=_default_rank_blend_providers
    )
    rerank: RankBlendRerankSettings = Field(default_factory=RankBlendRerankSettings)


class RankSettings(Model):
    backend: RankBackendKey = "blend"
    blend: RankBlendSettings = Field(default_factory=RankBlendSettings)
    heuristic: HeuristicRankSettings = Field(default_factory=HeuristicRankSettings)
    tfidf: RankTfidfSettings = Field(default_factory=RankTfidfSettings)
    bm25: RankBm25Settings = Field(default_factory=RankBm25Settings)
    cross_encoder: RankCrossEncoderSettings = Field(
        default_factory=RankCrossEncoderSettings
    )


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


class LLMModelSettings(Model):
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
    enable_structured: bool = True


def _default_overview_models() -> list[LLMModelSettings]:
    return [LLMModelSettings()]


class LLMSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    models: list[LLMModelSettings] = Field(default_factory=_default_overview_models)

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

    def resolve_model(self, name: str) -> LLMModelSettings:
        for item in self.models:
            if item.name == name:
                return item
        raise ValueError(f"llm model `{name}` does not exist in llm.models")


class TelemetryObsSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    backend: TelemetryObsBackendKey = "null"
    jsonl_path: str = ".serpsage_events.jsonl"

    @field_validator("jsonl_path")
    @classmethod
    def _validate_jsonl_path(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("telemetry.obs.jsonl_path must be non-empty")
        return token


class TelemetryMeteringSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    backend: TelemetryMeteringBackendKey = "null"
    sqlite_db_path: str = ".serpsage_metering.sqlite3"

    @field_validator("sqlite_db_path")
    @classmethod
    def _validate_sqlite_db_path(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("telemetry.metering.sqlite_db_path must be non-empty")
        return token


class TelemetrySettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enabled: bool = False
    queue_size: int = 2048
    drop_noncritical_when_full: bool = True
    obs: TelemetryObsSettings = Field(default_factory=TelemetryObsSettings)
    metering: TelemetryMeteringSettings = Field(
        default_factory=TelemetryMeteringSettings
    )

    @model_validator(mode="after")
    def _validate_telemetry(self) -> TelemetrySettings:
        if int(self.queue_size) <= 0:
            raise ValueError("telemetry.queue_size must be > 0")
        return self


class AppSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    http: HttpSettings = Field(default_factory=HttpSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
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
        _validate_optional_model_name(
            model_name=self.research.models.plan,
            names=names,
            path="research.models.plan",
        )
        _validate_optional_model_name(
            model_name=self.research.models.abstract_analyze,
            names=names,
            path="research.models.abstract_analyze",
        )
        _validate_optional_model_name(
            model_name=self.research.models.content_analyze,
            names=names,
            path="research.models.content_analyze",
        )
        _validate_optional_model_name(
            model_name=self.research.models.synthesize,
            names=names,
            path="research.models.synthesize",
        )
        _validate_optional_model_name(
            model_name=self.research.models.markdown,
            names=names,
            path="research.models.markdown",
        )
        return self


def _validate_optional_model_name(
    *, model_name: str, names: set[str], path: str
) -> None:
    token = str(model_name or "").strip()
    if not token:
        return
    if token not in names:
        raise ValueError(f"{path} must match one of llm.models[].name")


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
    "FetchAutoSettings",
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
    "LLMModelSettings",
    "ProviderSettings",
    "ReportStyleKey",
    "ResearchModelsSettings",
    "ResearchModeSettings",
    "ResearchSettings",
    "RunnerSettings",
    "SearchSettings",
    "RankBlendSettings",
    "RankSettings",
    "RetrySettings",
    "SearchDeepSettings",
    "SearxngSettings",
    "TelemetryMeteringSettings",
    "TelemetryObsSettings",
    "TelemetrySettings",
]
