from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_FINAL_WEIGHT_SUM_TARGET = 1.0
_WEIGHT_SUM_EPS = 1e-6


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


ReportStyleKey = Literal["decision", "explainer", "execution"]


class ComponentInstanceSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    component: str
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("component")
    @classmethod
    def _validate_component(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("component must be non-empty")
        return token


class ComponentFamilySettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    default: str = "default"
    instances: dict[str, ComponentInstanceSettings] = Field(default_factory=dict)
    declared_instances: frozenset[str] = Field(
        default_factory=frozenset,
        exclude=True,
        repr=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_component_family(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        if "backend" in value:
            raise ValueError(
                "legacy component config format is no longer supported; "
                "use `default` and `instances`"
            )
        payload = dict(value)
        if "declared_instances" not in payload:
            raw_instances = payload.get("instances")
            payload["declared_instances"] = (
                frozenset(str(key) for key in raw_instances)
                if isinstance(raw_instances, dict)
                else frozenset()
            )
        return payload

    @field_validator("default")
    @classmethod
    def _validate_default(cls, value: str) -> str:
        token = str(value or "").strip()
        if not token:
            raise ValueError("default component instance id must be non-empty")
        return token


def _family_settings(
    *,
    default: str,
    instances: dict[str, dict[str, Any]],
) -> ComponentFamilySettings:
    return ComponentFamilySettings(
        default=default,
        instances={
            key: ComponentInstanceSettings(
                component=str(item.get("component") or ""),
                enabled=bool(item.get("enabled", True)),
                config=dict(item.get("config") or {}),
            )
            for key, item in instances.items()
        },
        declared_instances=frozenset(),
    )


def _default_http_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="httpx",
        instances={
            "httpx": {
                "component": "httpx",
                "config": {},
            }
        },
    )


def _default_provider_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="searxng",
        instances={
            "searxng": {
                "component": "searxng",
                "config": {},
            }
        },
    )


def _default_crawl_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="auto",
        instances={
            "auto": {
                "component": "auto",
                "config": {},
            },
            "curl_cffi": {
                "component": "curl_cffi",
                "config": {},
            },
            "playwright": {
                "component": "playwright",
                "config": {},
            },
        },
    )


def _default_extract_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="auto",
        instances={
            "auto": {
                "component": "auto",
                "config": {},
            },
            "html": {
                "component": "html",
                "config": {},
            },
            "pdf": {
                "component": "pdf",
                "config": {},
            },
        },
    )


def _default_rank_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="blend",
        instances={
            "blend": {
                "component": "blend",
                "config": {},
            },
            "heuristic": {
                "component": "heuristic",
                "config": {},
            },
            "tfidf": {
                "component": "tfidf",
                "config": {},
            },
            "bm25": {
                "component": "bm25",
                "config": {},
            },
            "cross_encoder": {
                "component": "cross_encoder",
                "config": {},
            },
        },
    )


def _default_cache_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="null",
        instances={
            "null": {
                "component": "null",
                "config": {},
            }
        },
    )


def _default_llm_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="router",
        instances={
            "router": {
                "component": "router",
                "config": {},
            },
            "default_model": {
                "component": "openai",
                "config": {
                    "name": "gpt-4.1-mini",
                    "model": "gpt-4.1-mini",
                },
            },
        },
    )


def _default_telemetry_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="null",
        instances={
            "null": {
                "component": "null_emitter",
                "config": {},
            }
        },
    )


def _default_rate_limit_settings() -> ComponentFamilySettings:
    return _family_settings(
        default="basic",
        instances={
            "basic": {
                "component": "basic",
                "config": {},
            }
        },
    )


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


class FetchOverviewSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    use_model: str = "gpt-4.1-mini"
    max_abstract_chars: int = 900
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: str = "auto"


class FetchSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    extract: FetchExtractSettings = Field(default_factory=FetchExtractSettings)
    abstract: FetchAbstractSettings = Field(default_factory=FetchAbstractSettings)
    overview: FetchOverviewSettings = Field(default_factory=FetchOverviewSettings)


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


class RunnerSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    search_limit: int = Field(default=8, ge=1)
    fetch_limit: int = Field(default=24, ge=1)
    child_fetch_limit: int = Field(default=24, ge=1)
    queue_size: int = Field(default=256, ge=1)


class AppSettings(Model):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    http: ComponentFamilySettings = Field(default_factory=_default_http_settings)
    provider: ComponentFamilySettings = Field(
        default_factory=_default_provider_settings
    )
    crawl: ComponentFamilySettings = Field(default_factory=_default_crawl_settings)
    extract: ComponentFamilySettings = Field(default_factory=_default_extract_settings)
    rank: ComponentFamilySettings = Field(default_factory=_default_rank_settings)
    llm: ComponentFamilySettings = Field(default_factory=_default_llm_settings)
    cache: ComponentFamilySettings = Field(default_factory=_default_cache_settings)
    telemetry: ComponentFamilySettings = Field(
        default_factory=_default_telemetry_settings
    )
    rate_limit: ComponentFamilySettings = Field(
        default_factory=_default_rate_limit_settings
    )
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    runner: RunnerSettings = Field(default_factory=RunnerSettings)
    runtime_env: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    @model_validator(mode="before")
    @classmethod
    def _merge_component_family_defaults(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for family_name, factory in _COMPONENT_FAMILY_DEFAULT_FACTORIES.items():
            payload[family_name] = _merge_component_family_payload(
                default_settings=factory(),
                raw_value=payload.get(family_name),
            )
        return payload


_COMPONENT_FAMILY_DEFAULT_FACTORIES = {
    "http": _default_http_settings,
    "provider": _default_provider_settings,
    "crawl": _default_crawl_settings,
    "extract": _default_extract_settings,
    "rank": _default_rank_settings,
    "llm": _default_llm_settings,
    "cache": _default_cache_settings,
    "telemetry": _default_telemetry_settings,
    "rate_limit": _default_rate_limit_settings,
}


def _merge_component_family_payload(
    *,
    default_settings: ComponentFamilySettings,
    raw_value: object,
) -> object:
    default_payload = default_settings.model_dump(
        mode="python",
        exclude={"declared_instances"},
    )
    default_instances = dict(default_payload.get("instances") or {})
    if raw_value is None:
        return {
            **default_payload,
            "declared_instances": frozenset(),
        }
    if isinstance(raw_value, ComponentFamilySettings):
        raw_payload = raw_value.model_dump(
            mode="python",
            exclude={"declared_instances"},
        )
        declared_instances = frozenset(str(key) for key in raw_value.declared_instances)
    elif isinstance(raw_value, dict):
        raw_payload = dict(raw_value)
        raw_instances = raw_payload.get("instances")
        declared_instances = (
            frozenset(str(key) for key in raw_instances)
            if isinstance(raw_instances, dict)
            else frozenset()
        )
    else:
        return raw_value
    merged_instances = dict(default_instances)
    merged_instances.update(dict(raw_payload.get("instances") or {}))
    return {
        **default_payload,
        **raw_payload,
        "instances": merged_instances,
        "declared_instances": declared_instances,
    }


__all__ = [
    "AppSettings",
    "FetchAbstractSettings",
    "FetchExtractSettings",
    "FetchOverviewSettings",
    "FetchSettings",
    "AnswerGenerateSettings",
    "AnswerSettings",
    "AnswerStageSettings",
    "ComponentFamilySettings",
    "ComponentInstanceSettings",
    "Model",
    "ReportStyleKey",
    "ResearchModeSettings",
    "ResearchModelsSettings",
    "ResearchSettings",
    "RunnerSettings",
    "SearchDeepSettings",
    "SearchSettings",
]
