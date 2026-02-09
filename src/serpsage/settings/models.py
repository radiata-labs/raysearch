from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


DepthKey = Literal["low", "medium", "high"]
RankProviderKey = Literal["heuristic", "bm25"]


class RetrySettings(Model):
    max_attempts: int = 3
    base_delay_ms: int = 200
    max_delay_ms: int = 2_000


class SearxngSettings(Model):
    base_url: str = "https://searxng.lycoreco.dpdns.org/search"
    api_key: str | None = None
    timeout_s: float = 20.0
    allow_redirects: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    retry: RetrySettings = Field(default_factory=RetrySettings)


class ProviderSettings(Model):
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
    # Helper keeps mypy happy about Literal dict keys.
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


def _default_rank_providers() -> dict[RankProviderKey, float]:
    return {"heuristic": 1.0}


class FetchSettings(Model):
    user_agent: str = "serpsage-bot/3.0"
    timeout_s: float = 10.0
    max_bytes: int = 2_000_000
    max_extracted_chars: int = 50_000
    allow_content_types: list[str] = Field(
        default_factory=lambda: ["text/html", "application/xhtml+xml", "text/plain"]
    )
    follow_redirects: bool = True
    retry: RetrySettings = Field(default_factory=RetrySettings)
    global_concurrency: int = 16
    per_host_concurrency: int = 2
    politeness_delay_ms: int = 0


class ChunkingSettings(Model):
    target_chars: int = 1200
    overlap_sentences: int = 1
    min_chunk_chars: int = 200
    max_sentence_chars: int = 600
    max_blocks: int = 120
    max_sentences: int = 400
    max_chunks: int = 80


class SelectSettings(Model):
    early_bonus: float = 1.15
    template_hard_drop_threshold: float = 0.95
    block_hard_drop_threshold: float = 0.90
    # Enrich-specific chunk threshold (do NOT couple to pipeline.min_score).
    min_chunk_score: float = 0.20
    # Smooth gating around min_chunk_score. Set to 0 to disable and use hard thresholding.
    score_soft_gate_tau: float = 0.07
    # How strongly template-like chunks are penalized in logit space.
    template_penalty_weight: float = 2.0


class EnrichExtractorSettings(Model):
    kind: Literal["basic", "main_content"] = "main_content"


class EnrichSettings(Model):
    enabled: bool = True
    extractor: EnrichExtractorSettings = Field(default_factory=EnrichExtractorSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    select: SelectSettings = Field(default_factory=SelectSettings)
    depth_presets: dict[Literal["low", "medium", "high"], EnrichDepthPreset] = Field(
        default_factory=_default_depth_presets
    )


class HeuristicRankSettings(Model):
    unique_hit_weight: float = 6.0
    count_weight: float = 1.5
    intent_hit_weight: float = 5.0
    phrase_bonus: float = 8.0
    min_token_len: int = 2
    max_count_per_token: int = 5


class NormalizationSettings(Model):
    method: Literal["robust_sigmoid", "rank"] = "robust_sigmoid"
    temperature: float = 1.0
    min_items_for_sigmoid: int = 5
    flat_spread_eps: float = 1e-9
    z_clip: float = 8.0
    single_item_method: Literal["sigmoid_log1p", "exp", "fixed_0.5"] = "sigmoid_log1p"
    single_item_scale: float = 1.0


class RankSettings(Model):
    providers: dict[Literal["heuristic", "bm25"], float] = Field(
        default_factory=_default_rank_providers
    )
    heuristic: HeuristicRankSettings = Field(default_factory=HeuristicRankSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)


class CacheSettings(Model):
    enabled: bool = False
    db_path: str = ".serpsage_cache.sqlite3"
    search_ttl_s: int = 600
    fetch_ttl_s: int = 86_400


class OpenAICompatSettings(Model):
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    model: str = "gpt-4o-mini"
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    headers: dict[str, str] = Field(default_factory=dict)


class OverviewSettings(Model):
    enabled: bool = True
    llm: OpenAICompatSettings = Field(default_factory=OpenAICompatSettings)
    max_sources: int = 8
    max_chunks_per_source: int = 2
    max_chunk_chars: int = 900
    max_output_tokens: int = 600
    max_prompt_chars: int = 32_000
    cache_ttl_s: int = 0
    self_heal_retries: int = 1
    force_language: Literal["auto", "zh", "en"] = "auto"
    schema_strict: bool = True


class TelemetrySettings(Model):
    enabled: bool = False
    include_events: bool = False


class AppSettings(Model):
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
        # fallback to a default instance
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
    "CacheSettings",
    "ChunkingSettings",
    "EnrichDepthPreset",
    "EnrichSettings",
    "FetchSettings",
    "HeuristicRankSettings",
    "NormalizationSettings",
    "OpenAICompatSettings",
    "OverviewSettings",
    "PipelineSettings",
    "ProfileSettings",
    "ProviderSettings",
    "RetrySettings",
    "SearxngSettings",
    "TelemetrySettings",
]
