from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DepthKey = Literal["low", "medium", "high"]

DEFAULT_FETCH_USER_AGENT = "serpsage-bot/4.0"
DEFAULT_FOLLOW_REDIRECTS = True
DEFAULT_FETCH_VERSION = "v3"

RATE_LIMIT_GLOBAL_CONCURRENCY = 24
RATE_LIMIT_PER_HOST_CONCURRENCY = 4
RATE_LIMIT_POLITENESS_DELAY_MS = 0

PLAYWRIGHT_ENABLED = True
PLAYWRIGHT_HEADLESS = True
PLAYWRIGHT_JS_CONCURRENCY = 4
PLAYWRIGHT_NAV_TIMEOUT_MS = 2_500
PLAYWRIGHT_WAIT_NETWORK_IDLE_MS = 220
PLAYWRIGHT_BLOCK_RESOURCES = True

FETCH_BLOCKED_MARKERS: tuple[str, ...] = (
    "cloudflare",
    "just a moment",
    "verify you are human",
    "captcha",
    "access denied",
    "blocked",
    "please enable javascript",
    "security check",
    "checking your browser",
)


@dataclass(frozen=True, slots=True)
class FetchTuningProfile:
    http_timeout_s: float
    retry_attempts: int
    retry_delay_ms: int
    hedge_delay_ms: int
    min_text_chars: int
    min_content_score: float
    script_ratio_threshold: float
    max_render_pages: int
    max_base_attempts: int
    html_byte_budget: int
    text_byte_budget: int
    pdf_byte_budget: int
    binary_byte_budget: int
    html_early_accept_chars: int
    html_early_accept_score: float
    allow_speculative_render: bool


@dataclass(frozen=True, slots=True)
class ExtractTuningProfile:
    min_plain_chars: int
    quality_threshold: float
    max_html_chars: int
    max_markdown_chars: int
    fallback_parallel: bool
    trafilatura_timeout_s: float
    readability_enabled: bool
    trafilatura_enabled: bool


@dataclass(frozen=True, slots=True)
class ChunkTuningProfile:
    target_chars: int
    overlap_segments: int
    min_chunk_chars: int
    max_sentence_chars: int
    max_markdown_chars: int
    max_segments: int
    max_chunks: int
    query_prefilter_window: int
    min_query_token_hits: int
    min_chunk_score: float
    early_bonus: float


@dataclass(frozen=True, slots=True)
class DeadlineProfile:
    step_timeout_s: float
    page_timeout_s: float


_FETCH_PROFILES: dict[DepthKey, FetchTuningProfile] = {
    "low": FetchTuningProfile(
        http_timeout_s=0.9,
        retry_attempts=1,
        retry_delay_ms=60,
        hedge_delay_ms=100,
        min_text_chars=180,
        min_content_score=0.36,
        script_ratio_threshold=0.38,
        max_render_pages=0,
        max_base_attempts=2,
        html_byte_budget=900_000,
        text_byte_budget=450_000,
        pdf_byte_budget=12_000_000,
        binary_byte_budget=2_000_000,
        html_early_accept_chars=850,
        html_early_accept_score=0.66,
        allow_speculative_render=False,
    ),
    "medium": FetchTuningProfile(
        http_timeout_s=1.6,
        retry_attempts=1,
        retry_delay_ms=80,
        hedge_delay_ms=95,
        min_text_chars=220,
        min_content_score=0.42,
        script_ratio_threshold=0.35,
        max_render_pages=2,
        max_base_attempts=3,
        html_byte_budget=1_500_000,
        text_byte_budget=700_000,
        pdf_byte_budget=16_000_000,
        binary_byte_budget=3_000_000,
        html_early_accept_chars=1_100,
        html_early_accept_score=0.70,
        allow_speculative_render=True,
    ),
    "high": FetchTuningProfile(
        http_timeout_s=2.5,
        retry_attempts=2,
        retry_delay_ms=90,
        hedge_delay_ms=80,
        min_text_chars=260,
        min_content_score=0.40,
        script_ratio_threshold=0.33,
        max_render_pages=6,
        max_base_attempts=4,
        html_byte_budget=2_300_000,
        text_byte_budget=1_000_000,
        pdf_byte_budget=24_000_000,
        binary_byte_budget=4_000_000,
        html_early_accept_chars=1_350,
        html_early_accept_score=0.68,
        allow_speculative_render=True,
    ),
}

_EXTRACT_PROFILES: dict[DepthKey, ExtractTuningProfile] = {
    "low": ExtractTuningProfile(
        min_plain_chars=180,
        quality_threshold=0.42,
        max_html_chars=1_200_000,
        max_markdown_chars=110_000,
        fallback_parallel=True,
        trafilatura_timeout_s=0.45,
        readability_enabled=True,
        trafilatura_enabled=True,
    ),
    "medium": ExtractTuningProfile(
        min_plain_chars=220,
        quality_threshold=0.48,
        max_html_chars=1_800_000,
        max_markdown_chars=160_000,
        fallback_parallel=True,
        trafilatura_timeout_s=0.70,
        readability_enabled=True,
        trafilatura_enabled=True,
    ),
    "high": ExtractTuningProfile(
        min_plain_chars=260,
        quality_threshold=0.52,
        max_html_chars=2_600_000,
        max_markdown_chars=220_000,
        fallback_parallel=True,
        trafilatura_timeout_s=1.0,
        readability_enabled=True,
        trafilatura_enabled=True,
    ),
}

_CHUNK_PROFILES: dict[DepthKey, ChunkTuningProfile] = {
    "low": ChunkTuningProfile(
        target_chars=860,
        overlap_segments=1,
        min_chunk_chars=120,
        max_sentence_chars=520,
        max_markdown_chars=80_000,
        max_segments=220,
        max_chunks=24,
        query_prefilter_window=180,
        min_query_token_hits=1,
        min_chunk_score=0.18,
        early_bonus=1.12,
    ),
    "medium": ChunkTuningProfile(
        target_chars=1_050,
        overlap_segments=1,
        min_chunk_chars=160,
        max_sentence_chars=600,
        max_markdown_chars=140_000,
        max_segments=420,
        max_chunks=42,
        query_prefilter_window=320,
        min_query_token_hits=2,
        min_chunk_score=0.20,
        early_bonus=1.15,
    ),
    "high": ChunkTuningProfile(
        target_chars=1_220,
        overlap_segments=2,
        min_chunk_chars=190,
        max_sentence_chars=680,
        max_markdown_chars=220_000,
        max_segments=720,
        max_chunks=72,
        query_prefilter_window=540,
        min_query_token_hits=2,
        min_chunk_score=0.18,
        early_bonus=1.16,
    ),
}

_DEADLINE_PROFILES: dict[DepthKey, DeadlineProfile] = {
    "low": DeadlineProfile(step_timeout_s=1.2, page_timeout_s=0.9),
    "medium": DeadlineProfile(step_timeout_s=2.0, page_timeout_s=1.6),
    "high": DeadlineProfile(step_timeout_s=4.0, page_timeout_s=2.5),
}


def normalize_depth(depth: str | None) -> DepthKey:
    if depth == "low":
        return "low"
    if depth == "high":
        return "high"
    return "medium"


def fetch_profile_for_depth(depth: str | None) -> FetchTuningProfile:
    return _FETCH_PROFILES[normalize_depth(depth)]


def extract_profile_for_depth(depth: str | None) -> ExtractTuningProfile:
    return _EXTRACT_PROFILES[normalize_depth(depth)]


def chunk_profile_for_depth(depth: str | None) -> ChunkTuningProfile:
    return _CHUNK_PROFILES[normalize_depth(depth)]


def deadline_profile_for_depth(depth: str | None) -> DeadlineProfile:
    return _DEADLINE_PROFILES[normalize_depth(depth)]


__all__ = [
    "ChunkTuningProfile",
    "DEFAULT_FETCH_USER_AGENT",
    "DEFAULT_FETCH_VERSION",
    "DEFAULT_FOLLOW_REDIRECTS",
    "DeadlineProfile",
    "ExtractTuningProfile",
    "FETCH_BLOCKED_MARKERS",
    "FetchTuningProfile",
    "PLAYWRIGHT_BLOCK_RESOURCES",
    "PLAYWRIGHT_ENABLED",
    "PLAYWRIGHT_HEADLESS",
    "PLAYWRIGHT_JS_CONCURRENCY",
    "PLAYWRIGHT_NAV_TIMEOUT_MS",
    "PLAYWRIGHT_WAIT_NETWORK_IDLE_MS",
    "RATE_LIMIT_GLOBAL_CONCURRENCY",
    "RATE_LIMIT_PER_HOST_CONCURRENCY",
    "RATE_LIMIT_POLITENESS_DELAY_MS",
    "chunk_profile_for_depth",
    "deadline_profile_for_depth",
    "extract_profile_for_depth",
    "fetch_profile_for_depth",
    "normalize_depth",
]
