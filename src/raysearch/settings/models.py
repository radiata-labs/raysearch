from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from raysearch.models.components.tracking import TrackingLevel, normalize_tracking_level


class TrackingEmitterSettings(BaseModel):
    """Tracking emitter configuration (NOT a component config)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    queue_size: int = Field(default=2048, ge=1)
    minimum_level: TrackingLevel = "INFO"
    drop_noncritical_when_full: bool = True

    @field_validator("minimum_level", mode="before")
    @classmethod
    def _validate_level(cls, value: object) -> TrackingLevel:
        return normalize_tracking_level(value)


class MeteringEmitterSettings(BaseModel):
    """Metering emitter configuration (NOT a component config)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    queue_size: int = Field(default=2048, ge=1)


class TelemetrySettings(BaseModel):
    """Top-level telemetry configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    tracking: TrackingEmitterSettings = Field(default_factory=TrackingEmitterSettings)
    metering: MeteringEmitterSettings = Field(default_factory=MeteringEmitterSettings)


class SettingModel(BaseModel):
    model_config = ConfigDict(extra="allow")


ReportStyleKey = Literal["decision", "explainer", "execution"]


class FetchExtractSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_markdown_chars: int = 160_000
    min_text_chars: int = 220
    min_primary_chars: int = 220
    min_total_chars_with_secondary: int = 220
    include_secondary_content_default: bool = False
    collect_links_default: bool = False
    link_max_count: int = 800
    link_keep_hash: bool = False


class FetchAbstractSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_abstract_chars: int = 2000
    min_abstract_score: float = 0.20
    min_abstract_tokens: int = 4
    title_boost_alpha: float = 0.35


class FetchOverviewSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    use_model: str = "gpt-4.1-mini"
    max_abstract_chars: int = 900
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: str = "auto"


class FetchSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    extract: FetchExtractSettings = Field(default_factory=FetchExtractSettings)
    abstract: FetchAbstractSettings = Field(default_factory=FetchAbstractSettings)
    overview: FetchOverviewSettings = Field(default_factory=FetchOverviewSettings)


class SearchExpansionSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    llm_model: str = ""
    llm_timeout_s: float = 20.0

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchExpansionSettings:
        if float(self.llm_timeout_s) <= 0:
            raise ValueError("search.expansion.llm_timeout_s must be > 0")
        return self


class SearchModeSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_extra_queries: int = 0
    prefetch_multiplier: float = 1.0
    prefetch_max_urls: int = 16
    context_docs_limit: int = 0
    context_doc_min_chars: int = 0
    rank_by_context: bool = False

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchModeSettings:
        if int(self.max_extra_queries) < 0:
            raise ValueError("search mode max_extra_queries must be >= 0")
        if float(self.prefetch_multiplier) < 1.0:
            raise ValueError("search mode prefetch_multiplier must be >= 1.0")
        if int(self.prefetch_max_urls) <= 0:
            raise ValueError("search mode prefetch_max_urls must be > 0")
        if int(self.context_docs_limit) < 0:
            raise ValueError("search mode context_docs_limit must be >= 0")
        if int(self.context_doc_min_chars) < 0:
            raise ValueError("search mode context_doc_min_chars must be >= 0")
        if bool(self.rank_by_context) and int(self.context_docs_limit) <= 0:
            raise ValueError(
                "search mode context_docs_limit must be > 0 when rank_by_context=true"
            )
        if not bool(self.rank_by_context) and int(self.context_doc_min_chars) > 0:
            raise ValueError(
                "search mode context_doc_min_chars requires rank_by_context=true"
            )
        return self


def _default_search_fast_mode() -> SearchModeSettings:
    return SearchModeSettings(
        max_extra_queries=0,
        prefetch_multiplier=1.0,
        prefetch_max_urls=16,
        context_docs_limit=0,
        context_doc_min_chars=0,
        rank_by_context=False,
    )


def _default_search_auto_mode() -> SearchModeSettings:
    return SearchModeSettings(
        max_extra_queries=1,
        prefetch_multiplier=2.0,
        prefetch_max_urls=32,
        context_docs_limit=0,
        context_doc_min_chars=0,
        rank_by_context=False,
    )


def _default_search_deep_mode() -> SearchModeSettings:
    return SearchModeSettings(
        max_extra_queries=6,
        prefetch_multiplier=3.0,
        prefetch_max_urls=48,
        context_docs_limit=18,
        context_doc_min_chars=16,
        rank_by_context=True,
    )


class SearchModeProfilesSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    fast: SearchModeSettings = Field(default_factory=_default_search_fast_mode)
    auto: SearchModeSettings = Field(default_factory=_default_search_auto_mode)
    deep: SearchModeSettings = Field(default_factory=_default_search_deep_mode)


class SearchSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_results: int = 16
    select_engines: bool = False
    expansion: SearchExpansionSettings = Field(default_factory=SearchExpansionSettings)
    modes: SearchModeProfilesSettings = Field(
        default_factory=SearchModeProfilesSettings
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchSettings:
        if int(self.max_results) <= 0:
            raise ValueError("search.max_results must be > 0")
        return self


class AnswerStageSettings(SettingModel):
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


class AnswerSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    select_engines: bool = False
    plan: AnswerStageSettings = Field(default_factory=AnswerStageSettings)
    generate: AnswerGenerateSettings = Field(default_factory=AnswerGenerateSettings)


class ResearchModelsSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    plan: str = ""
    link_select: str = ""
    abstract_analyze: str = ""
    content_analyze: str = ""
    synthesize: str = ""
    markdown: str = ""


class ResearchModeSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    max_rounds: int = 5
    max_search_calls: int = 12
    max_fetch_calls: int = 24
    max_results_per_search: int = 6
    max_queries_per_round: int = 3
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


class ResearchSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    tool_max_attempts: int = 3
    llm_self_heal_retries: int = 2
    select_engines: bool = False
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


class RunnerSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    search_limit: int = Field(default=8, ge=1)
    fetch_limit: int = Field(default=24, ge=1)
    child_fetch_limit: int = Field(default=24, ge=1)
    queue_size: int = Field(default=256, ge=1)


class ComponentFamilySettings(SettingModel):
    default: str = ""

    @field_validator("default")
    @classmethod
    def _normalize_default(cls, value: str) -> str:
        return str(value).strip()


class CacheSettings(ComponentFamilySettings):
    pass


class CrawlSettings(ComponentFamilySettings):
    pass


class ExtractSettings(ComponentFamilySettings):
    pass


class HttpSettings(ComponentFamilySettings):
    pass


class LlmSettings(ComponentFamilySettings):
    pass


class ProviderSettings(ComponentFamilySettings):
    pass


class RankSettings(ComponentFamilySettings):
    pass


class RateLimitSettings(ComponentFamilySettings):
    pass


class TrackingSettings(ComponentFamilySettings):
    pass


class MeteringSettings(ComponentFamilySettings):
    pass


class ComponentSettings(SettingModel):
    cache: CacheSettings = Field(default_factory=CacheSettings)
    crawl: CrawlSettings = Field(default_factory=CrawlSettings)
    extract: ExtractSettings = Field(default_factory=ExtractSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    metering: MeteringSettings = Field(default_factory=MeteringSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    rank: RankSettings = Field(default_factory=RankSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)


class AppSettings(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    components: ComponentSettings = Field(default_factory=ComponentSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)

    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    runtime_env: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)


__all__ = [
    "AppSettings",
    "FetchAbstractSettings",
    "FetchExtractSettings",
    "FetchOverviewSettings",
    "FetchSettings",
    "AnswerGenerateSettings",
    "AnswerSettings",
    "AnswerStageSettings",
    "CacheSettings",
    "ComponentSettings",
    "CrawlSettings",
    "ExtractSettings",
    "HttpSettings",
    "LlmSettings",
    "MeteringEmitterSettings",
    "MeteringSettings",
    "ProviderSettings",
    "RankSettings",
    "RateLimitSettings",
    "SettingModel",
    "TelemetrySettings",
    "TrackingEmitterSettings",
    "TrackingSettings",
    "ReportStyleKey",
    "ResearchModeSettings",
    "ResearchModelsSettings",
    "ResearchSettings",
    "RunnerSettings",
    "SearchExpansionSettings",
    "SearchModeProfilesSettings",
    "SearchModeSettings",
    "SearchSettings",
]
