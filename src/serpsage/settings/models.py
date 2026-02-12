from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


DepthKey = Literal["low", "medium", "high"]
ProviderBackendKey = Literal["searxng"]
ExtractorBackendKey = Literal["basic", "main_content"]
FetchBackendKey = Literal["httpx", "curl_cffi", "auto"]
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


class AutoMatchSettings(Model):
    enabled: bool = False
    keywords: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)
    priority: int = 0


class ProfileSettings(Model):
    fuzzy_threshold: float = 0.88
    auto_match: AutoMatchSettings = Field(default_factory=AutoMatchSettings)
    intent_terms: list[str] = Field(default_factory=list)
    noise_words: list[str] = Field(default_factory=list)
    noise_extensions: list[str] = Field(
        default_factory=lambda: ["txt", "dic", "pdf", "zip", "rar", "7z"]
    )
    domain_bonus: dict[str, int] = Field(default_factory=dict)
    domain_groups: dict[str, list[str]] = Field(default_factory=dict)
    title_tail_patterns: list[str] = Field(default_factory=list)


class PipelineSettings(Model):
    default_profile: str = "general"
    profiles: dict[str, ProfileSettings] = Field(
        default_factory=lambda: {"general": ProfileSettings()}
    )
    max_results: int = 16
    min_score: float = 0.5
    include_raw: bool = False


class EnrichDepthPreset(Model):
    pages_ratio: float = 0.25
    min_pages: int = 1
    max_pages: int = 3
    top_chunks_per_page: int = 2


def _default_depth_presets() -> dict[DepthKey, EnrichDepthPreset]:
    return {
        "low": EnrichDepthPreset(
            pages_ratio=0.25, min_pages=1, max_pages=3, top_chunks_per_page=2
        ),
        "medium": EnrichDepthPreset(
            pages_ratio=0.50, min_pages=2, max_pages=6, top_chunks_per_page=3
        ),
        "high": EnrichDepthPreset(
            pages_ratio=0.75, min_pages=3, max_pages=10, top_chunks_per_page=5
        ),
    }


def _default_rank_blend_providers() -> dict[RankBlendProviderKey, float]:
    return {"heuristic": 1.0}


class FetchHttpxSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    retry: RetrySettings = Field(default_factory=RetrySettings)


class FetchCurlCffiSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    impersonate: str = "chrome120"
    http2: bool = True
    verify_ssl: bool = True
    retry: RetrySettings = Field(default_factory=RetrySettings)


class FetchAutoSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class FetchRateLimitSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    global_concurrency: int = 16
    per_host_concurrency: int = 2
    politeness_delay_ms: int = 0


class FetchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    backend: FetchBackendKey = "auto"
    user_agent: str = "serpsage-bot/3.0"
    timeout_s: float = 10.0
    follow_redirects: bool = True
    extra_headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    rate_limit: FetchRateLimitSettings = Field(default_factory=FetchRateLimitSettings)
    httpx: FetchHttpxSettings = Field(default_factory=FetchHttpxSettings)
    curl_cffi: FetchCurlCffiSettings = Field(default_factory=FetchCurlCffiSettings)
    auto: FetchAutoSettings = Field(default_factory=FetchAutoSettings)


class ChunkingSettings(Model):
    target_chars: int = 1200
    overlap_sentences: int = 1
    min_chunk_chars: int = 200
    max_sentence_chars: int = 600
    max_blocks: int = 120
    max_sentences: int = 400
    max_chunks: int = 80


class SelectSettings(Model):
    min_query_token_hits: int = 2
    early_bonus: float = 1.15
    min_chunk_score: float = 0.20


class ExtractorBasicSettings(Model):
    pass


class ExtractorMainContentSettings(Model):
    pass


class EnrichExtractorSettings(Model):
    backend: ExtractorBackendKey = "main_content"
    basic: ExtractorBasicSettings = Field(default_factory=ExtractorBasicSettings)
    main_content: ExtractorMainContentSettings = Field(
        default_factory=ExtractorMainContentSettings
    )


class EnrichSettings(Model):
    enabled: bool = True
    extractor: EnrichExtractorSettings = Field(default_factory=EnrichExtractorSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    select: SelectSettings = Field(default_factory=SelectSettings)
    depth_presets: dict[DepthKey, EnrichDepthPreset] = Field(
        default_factory=_default_depth_presets
    )


class HeuristicRankSettings(Model):
    unique_hit_weight: float = 6.0
    count_weight: float = 1.5
    intent_hit_weight: float = 5.0
    max_count_per_token: int = 5


class RankBm25Settings(Model):
    pass


class NormalizationSettings(Model):
    method: Literal["robust_sigmoid", "rank"] = "robust_sigmoid"
    temperature: float = 1.0
    min_items_for_sigmoid: int = 5
    flat_spread_eps: float = 1e-9
    z_clip: float = 8.0
    single_item_method: Literal["sigmoid_log1p", "exp", "fixed_0.5"] = "sigmoid_log1p"
    single_item_scale: float = 1.0


class RankBlendSettings(Model):
    providers: dict[RankBlendProviderKey, float] = Field(
        default_factory=_default_rank_blend_providers
    )


class RankSettings(Model):
    backend: RankBackendKey = "blend"
    blend: RankBlendSettings = Field(default_factory=RankBlendSettings)
    heuristic: HeuristicRankSettings = Field(default_factory=HeuristicRankSettings)
    bm25: RankBm25Settings = Field(default_factory=RankBm25Settings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)


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


class OverviewSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = True
    use_model: str = "gpt-4o-mini"
    models: list[OverviewModelSettings] = Field(
        default_factory=_default_overview_models
    )
    max_sources: int = 8
    max_chunks_per_source: int = 2
    max_chunk_chars: int = 900
    max_output_tokens: int = 600
    max_prompt_chars: int = 32_000
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: Literal["auto", "zh", "en"] = "auto"

    @model_validator(mode="after")
    def _validate_models(self) -> OverviewSettings:
        if not self.models:
            raise ValueError("overview.models must contain at least one model")

        names: set[str] = set()
        for idx, item in enumerate(self.models):
            if not item.name:
                raise ValueError(f"overview.models[{idx}].name must be non-empty")
            if item.name in names:
                raise ValueError(
                    f"duplicate overview model name `{item.name}` in overview.models"
                )
            names.add(item.name)

        if self.use_model not in names:
            raise ValueError(
                "overview.use_model must match one of overview.models[].name"
            )
        return self

    def resolve_model(self) -> OverviewModelSettings:
        for item in self.models:
            if item.name == self.use_model:
                return item
        raise ValueError(
            f"overview.use_model `{self.use_model}` does not exist in overview.models"
        )


class TelemetrySettings(Model):
    enabled: bool = False
    include_events: bool = False


class AppSettings(Model):
    http: HttpSettings = Field(default_factory=HttpSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    enrich: EnrichSettings = Field(default_factory=EnrichSettings)
    rank: RankSettings = Field(default_factory=RankSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    overview: OverviewSettings = Field(default_factory=OverviewSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    def get_profile(self, name: str) -> ProfileSettings:
        if name in self.pipeline.profiles:
            return self.pipeline.profiles[name]
        if self.pipeline.default_profile in self.pipeline.profiles:
            return self.pipeline.profiles[self.pipeline.default_profile]
        return ProfileSettings()

    def select_profile(
        self, *, query: str, explicit: str | None
    ) -> tuple[str, ProfileSettings]:
        if explicit:
            return explicit, self.get_profile(explicit)

        q = (query or "").lower()
        best_name: str | None = None
        best_score = -(10**9)
        for name, prof in self.pipeline.profiles.items():
            am = prof.auto_match
            if not am.enabled:
                continue
            hits = sum(1 for kw in am.keywords if kw and kw.lower() in q)
            if am.regex:
                for pat in am.regex:
                    if not pat:
                        continue
                    try:
                        if re.search(pat, q, re.IGNORECASE):
                            hits += 1
                    except re.error:
                        continue
            if hits <= 0:
                continue
            score = hits + int(am.priority)
            if score > best_score:
                best_score = score
                best_name = name

        chosen = best_name or self.pipeline.default_profile
        return chosen, self.get_profile(chosen)


__all__ = [
    "AppSettings",
    "CacheMySQLSettings",
    "CacheRedisSettings",
    "CacheSQLAlchemySettings",
    "CacheSettings",
    "ChunkingSettings",
    "EnrichDepthPreset",
    "EnrichSettings",
    "EnrichExtractorSettings",
    "FetchAutoSettings",
    "FetchCurlCffiSettings",
    "FetchHttpxSettings",
    "FetchRateLimitSettings",
    "FetchSettings",
    "HttpSettings",
    "HeuristicRankSettings",
    "NormalizationSettings",
    "OverviewModelBackendKey",
    "OverviewModelSettings",
    "OverviewSettings",
    "PipelineSettings",
    "ProfileSettings",
    "ProviderSettings",
    "RankBlendSettings",
    "RankSettings",
    "RetrySettings",
    "SearxngSettings",
    "TelemetrySettings",
]
