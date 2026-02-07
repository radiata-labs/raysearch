from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from .client import SearxngClient
from .config import SearchConfig, SearchContextConfig
from .models import SearchContext, SearchResult
from .scorer import ScoringEngine
from .tools import (
    canonical_site,
    compile_patterns,
    domain_bonus,
    fuzzy_normalize,
    hybrid_similarity,
    strip_title_tails,
)
from .utils import TextUtils
from .web import WebEnricher

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SearchDepth = Literal["simple", "low", "medium", "high"]


class SearchPipeline:
    """High-level search pipeline: profile selection + result processing + rendering."""

    client: SearxngClient
    config: SearchConfig

    def __init__(
        self,
        config: SearchConfig,
        *,
        client: SearxngClient | None = None,
        page_fetcher: Callable[[str], str] | None = None,
        scorer: ScoringEngine | None = None,
        web_enricher: WebEnricher | None = None,
    ) -> None:
        self.config = config
        self.client = client or SearxngClient(config)
        self.scorer = scorer or ScoringEngine(
            score_norm_cfg=self.config.score_normalization
        )
        self.web_enricher = web_enricher or WebEnricher(
            self.config.web_enrichment,
            user_agent=self.config.searxng.user_agent,
            fetcher=page_fetcher,
            score_normalization=self.config.score_normalization,
        )

    def search_raw(
        self, query: str, *, params: Mapping[str, object] | None = None
    ) -> dict:
        """Fetch raw results from SearxNG (no processing)."""

        return self.client.search(query, params=params)

    def search_context(
        self,
        query: str,
        depth: SearchDepth = "simple",
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
        max_chunk_chars: int = 800,
    ) -> SearchContext:
        """Search and build LLM-friendly context."""

        raw = self.search_raw(query, params=params)
        return self.build_context(
            raw,
            query,
            depth,
            profile=profile,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
            show_source_domain=show_source_domain,
            show_source_url=show_source_url,
            show_source_engine=show_source_engine,
            fuzzy_threshold=fuzzy_threshold,
            chunk_target_chars=chunk_target_chars,
            chunk_overlap_sentences=chunk_overlap_sentences,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        )

    def search_json(
        self,
        query: str,
        depth: SearchDepth = "simple",
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        fuzzy_threshold: float | None = None,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> dict[str, object]:
        """Search and return JSON data only."""
        raw = self.search_raw(query, params=params)
        query = TextUtils.clean_whitespace(query)
        if not query:
            return {}

        processed = self.process_response(
            raw,
            query,
            depth,
            profile=profile,
            max_results=max_results,
            fuzzy_threshold=fuzzy_threshold,
            chunk_target_chars=chunk_target_chars,
            chunk_overlap_sentences=chunk_overlap_sentences,
            min_chunk_chars=min_chunk_chars,
        )
        return {
            "query": query,
            "depth": depth,
            "number_of_results": len(processed),
            "results": [item.model_dump() for item in processed],
        }

    def search_markdown(
        self,
        query: str,
        depth: SearchDepth = "simple",
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
        max_chunk_chars: int = 800,
    ) -> str:
        """Search and return markdown only."""

        raw = self.search_raw(query, params=params)
        return self.build_context(
            raw,
            query,
            depth,
            profile=profile,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
            show_source_domain=show_source_domain,
            show_source_url=show_source_url,
            show_source_engine=show_source_engine,
            fuzzy_threshold=fuzzy_threshold,
            chunk_target_chars=chunk_target_chars,
            chunk_overlap_sentences=chunk_overlap_sentences,
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
        ).markdown

    def build_context(
        self,
        searxng_response: Mapping[str, object],
        user_query: str,
        depth: SearchDepth = "simple",
        *,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
        max_chunk_chars: int = 800,
    ) -> SearchContext:
        """Build context from a raw SearxNG response (no HTTP)."""

        query = TextUtils.clean_whitespace(user_query)
        if not query:
            return SearchContext(
                query="",
                results=[],
                json_data={},
                markdown=self._render_empty("用户问题为空。"),
            )

        processed = self.process_response(
            searxng_response,
            query,
            depth,
            profile=profile,
            max_results=max_results,
            fuzzy_threshold=fuzzy_threshold,
            chunk_target_chars=chunk_target_chars,
            chunk_overlap_sentences=chunk_overlap_sentences,
            min_chunk_chars=min_chunk_chars,
        )

        json_data = {
            "query": query,
            "depth": depth,
            "number_of_results": len(processed),
            "results": [item.model_dump() for item in processed],
        }
        markdown = self._render_markdown(
            query,
            processed,
            max_snippet_chars=max_snippet_chars,
            max_chunk_chars=max_chunk_chars,
            show_source_domain=show_source_domain,
            show_source_url=show_source_url,
            show_source_engine=show_source_engine,
        )

        return SearchContext(
            query=query, results=processed, json_data=json_data, markdown=markdown
        )

    def process_response(
        self,
        response: Mapping[str, object],
        query: str,
        depth: SearchDepth = "simple",
        *,
        profile: str | None = None,
        max_results: int = 16,
        fuzzy_threshold: float | None = None,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> list[SearchResult]:
        """Process a raw SearXNG response into ranked results (no HTTP)."""

        self._validate_depth(depth)
        if chunk_target_chars is not None and chunk_target_chars <= 0:
            raise ValueError("chunk_target_chars must be > 0")
        if chunk_overlap_sentences is not None and chunk_overlap_sentences < 0:
            raise ValueError("chunk_overlap_sentences must be >= 0")

        env_profile = os.getenv("SEARCH_PROFILE")
        explicit_profile = profile or env_profile
        profile_name = self._select_profile_name(query, explicit_profile)
        config = self._get_profile_config(profile_name)
        if fuzzy_threshold is not None:
            config = config.with_overrides(fuzzy_threshold=fuzzy_threshold)

        results_data = (response or {}).get("results", [])
        raw_results = list(results_data if isinstance(results_data, list) else [])

        processed = self._run_processing(query, config, raw_results)[:max_results]
        if depth != "simple" and processed and self.config.web_enrichment.enabled:
            preset = self.config.web_enrichment.depth_presets.get(depth)
            if preset is None:
                logger.warning(
                    "No depth preset configured for '%s'; skipping web enrichment.",
                    depth,
                )
                return processed

            query_tokens = [t for t in TextUtils.tokenize(query) if len(t) >= 2]
            intent_tokens = self._extract_intent_tokens(query, config)
            self.web_enricher.enrich_results(
                processed,
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
                context_config=config,
                ranking_config=config.ranking,
                preset=preset,
                chunk_target_chars=chunk_target_chars,
                chunk_overlap_sentences=chunk_overlap_sentences,
                min_chunk_chars=min_chunk_chars,
            )
        return processed

    def _select_profile_name(self, query: str, explicit_profile: str | None) -> str:
        """Select a profile based on explicit input or keyword rules.

        Explicit profile must exist, otherwise raise.
        """

        if explicit_profile:
            if not self.config.has_profile(explicit_profile):
                raise ValueError(f"Profile not found: {explicit_profile}")
            return explicit_profile

        query_text = (query or "").lower()
        best_profile: str | None = None
        best_score = -1

        for profile_name, profile_config in self.config.profiles.items():
            auto_match = profile_config.auto_match
            if not auto_match.enabled:
                continue
            hits = self._count_keyword_hits(query_text, auto_match.keywords)
            hits += self._count_regex_hits(query_text, auto_match.regex)
            if hits <= 0:
                continue
            score = hits + auto_match.priority
            if score > best_score:
                best_score = score
                best_profile = profile_name

        if best_profile:
            return best_profile

        return self.config.default_profile

    def _get_profile_config(self, name: str) -> SearchContextConfig:
        config = self.config.get_profile(name)
        if config is not None:
            return config

        fallback_name = self.config.default_profile
        fallback = self.config.get_profile(fallback_name)
        if fallback is not None:
            logger.warning(
                "Profile '%s' not found; fallback to '%s'.",
                name,
                fallback_name,
            )
            return fallback

        logger.warning("Profile '%s' not found in config; using defaults.", name)
        return SearchContextConfig()

    @staticmethod
    def _count_keyword_hits(query: str, keywords: Iterable[str]) -> int:
        hits = 0
        for keyword in keywords:
            if keyword and keyword.lower() in query:
                hits += 1
        return hits

    @staticmethod
    def _count_regex_hits(query: str, patterns: Iterable[str]) -> int:
        hits = 0
        for pattern in patterns:
            if not pattern:
                continue
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    hits += 1
            except re.error:
                logger.warning("Invalid auto profile regex: %s", pattern)
        return hits

    @staticmethod
    def _validate_depth(depth: SearchDepth) -> None:
        if depth not in {"simple", "low", "medium", "high"}:
            raise ValueError(f"Invalid depth: {depth}")

    def _run_processing(
        self,
        query: str,
        config: SearchContextConfig,
        raw_results: list[Mapping[str, object]],
    ) -> list[SearchResult]:
        normalized = [self._normalize_result(result) for result in raw_results]

        noise_extensions = {ext.lower().lstrip(".") for ext in config.noise_extensions}
        title_tail_patterns = compile_patterns(config.title_tail_patterns)
        domain_groups = config.domain_groups or {}

        query_tokens = [t for t in TextUtils.tokenize(query) if len(t) >= 2]
        intent_tokens = self._extract_intent_tokens(query, config)

        normalized = [
            result
            for result in normalized
            if self._is_not_noise(result, config, noise_extensions)
        ]
        if not normalized:
            return []

        normalized = [
            result
            for result in normalized
            if self._is_relevant(result, config, query_tokens, intent_tokens)
        ]
        if not normalized:
            return []

        normalized = self._dedupe_exact(normalized)
        normalized = self._dedupe_fuzzy(
            normalized,
            config.fuzzy_threshold,
            config=config,
            title_tail_patterns=title_tail_patterns,
            domain_groups=domain_groups,
        )

        return self.scorer.rank_results(
            normalized,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
            context_config=config,
            ranking_config=config.ranking,
            score_norm_cfg=self.config.score_normalization,
        )

    def _render_markdown(
        self,
        query: str,
        results: list[SearchResult],
        *,
        max_snippet_chars: int = 1000,
        max_chunk_chars: int = 800,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
    ) -> str:
        if not results:
            return self._render_empty("无有用搜索结果(已过滤无关与噪音)。")

        lines: list[str] = [
            "# 网络搜索结果 (SearXNG)",
            "",
            "## 用户问题",
            query,
            "",
            "## 搜索结果",
        ]

        for index, result in enumerate(results, 1):
            sid = f"S{index}"
            title = result.title or "(no-title)"
            snippet = result.snippet
            if max_snippet_chars and len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars].rstrip() + "..."

            lines.append(f"### [{sid}] {title}")

            if show_source_domain and result.domain:
                lines.append(f"- 来源: {result.domain}")

            if show_source_url and result.url:
                lines.append(f"- 链接: {result.url}")

            if result.published_date:
                lines.append(f"- 时间: {result.published_date}")

            if snippet:
                lines.append(f"- 内容: {snippet}")

            if result.page.chunks:
                lines.append("- 页面片段:")
                for _, chunk in enumerate(result.page.chunks, 1):
                    rendered = chunk.text
                    if max_chunk_chars and len(rendered) > max_chunk_chars:
                        rendered = rendered[:max_chunk_chars].rstrip() + "..."
                    score_str = f"(score={chunk.score:.2f}) "
                    lines.append(f"  - {score_str}{rendered}")

            if result.hit_keywords:
                lines.append(f"- 命中: {', '.join(result.hit_keywords[:12])}")

            if show_source_engine and result.engine:
                lines.append(f"- 引擎: {result.engine}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_empty(reason: str) -> str:
        return "\n".join(
            [
                "# 网络搜索结果 (SearXNG)",
                "",
                "## 搜索结果",
                f"_{reason}_",
                "",
            ]
        )

    def _normalize_result(self, raw: Mapping[str, object]) -> SearchResult:
        url = str(raw.get("url") or "").strip()
        title_raw = str(raw.get("title") or "").strip()

        snippet_raw = raw.get("snippet")
        if snippet_raw is None:
            snippet_raw = raw.get("content")
        if snippet_raw is None:
            snippet_raw = raw.get("description")

        snippet_text = str(snippet_raw or "").strip()
        published = raw.get("publishedDate")
        published_str = "" if published in (None, "null") else str(published).strip()
        engine = str(raw.get("engine") or "").strip()

        title = TextUtils.clean_whitespace(TextUtils.strip_html(title_raw))
        snippet = TextUtils.clean_whitespace(TextUtils.strip_html(snippet_text))
        domain = self._extract_domain(url)

        return SearchResult(
            url=url,
            title=title,
            snippet=snippet,
            domain=domain,
            published_date=published_str,
            engine=engine,
            raw=dict(raw),
        )

    @staticmethod
    def _extract_domain(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc
        if not host and parsed.path:
            host = parsed.path.split("/")[0]
        return host.lower()

    def _is_not_noise(
        self,
        result: SearchResult,
        config: SearchContextConfig,
        noise_extensions: set[str],
    ) -> bool:
        title = result.title.strip()
        snippet = result.snippet.strip()
        url = result.url.strip()
        domain = result.domain.strip()
        blob = f"{title} {snippet} {url} {domain}".lower()

        if not title and not snippet:
            return False

        if self._is_noise_extension(url, noise_extensions):
            return False

        for word in config.noise_words:
            if word.lower() in blob:
                return False

        return not (len(title) < 2 and len(snippet) < 40)

    def _is_relevant(
        self,
        result: SearchResult,
        config: SearchContextConfig,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> bool:
        title = result.title.lower()
        snippet = result.snippet.lower()
        domain = result.domain.lower()
        blob = f"{title} {snippet}"

        core_hits = 0
        score = 0

        for token in query_tokens:
            if token in title:
                score += 10
                core_hits += 1
            elif token in snippet:
                score += 5
                core_hits += 1

        intent_hits = 0
        for token in intent_tokens:
            if token in blob:
                score += 4
                intent_hits += 1

        score += domain_bonus(domain, config.domain_bonus)

        if core_hits == 0:
            return False

        if score < config.ranking.min_relevance_score:
            return False

        return not (
            intent_tokens
            and intent_hits == 0
            and score < config.ranking.min_intent_score
        )

    @staticmethod
    def _dedupe_exact(results: list[SearchResult]) -> list[SearchResult]:
        seen: set[tuple[str, str]] = set()
        output: list[SearchResult] = []
        for result in results:
            title = TextUtils.clean_whitespace(result.title).lower()
            snippet = TextUtils.clean_whitespace(result.snippet).lower()
            fingerprint = (title[:140], snippet[:260])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            output.append(result)
        return output

    def _dedupe_fuzzy(
        self,
        results: list[SearchResult],
        threshold: float,
        *,
        config: SearchContextConfig,
        title_tail_patterns: list[re.Pattern[str]],
        domain_groups: Mapping[str, tuple[str, ...]],
    ) -> list[SearchResult]:
        kept: list[SearchResult] = []

        for candidate in sorted(
            results,
            key=lambda r: self._quality_score(r, config),
            reverse=True,
        ):
            if not kept:
                kept.append(candidate)
                continue

            candidate_key = self._fuzzy_key(candidate, title_tail_patterns)
            candidate_site = canonical_site(candidate.domain, dict(domain_groups))

            duplicate_index = -1
            for index, existing in enumerate(kept):
                existing_key = self._fuzzy_key(existing, title_tail_patterns)
                existing_site = canonical_site(existing.domain, dict(domain_groups))

                threshold_value = (
                    threshold
                    if candidate_site == existing_site
                    else min(0.94, threshold + 0.06)
                )

                if (
                    hybrid_similarity(candidate_key, existing_key) >= threshold_value
                ):
                    duplicate_index = index
                    break

            if duplicate_index < 0:
                kept.append(candidate)
            elif self._quality_score(candidate, config) > self._quality_score(
                kept[duplicate_index], config
            ):
                kept[duplicate_index] = candidate

        kept_ids = {id(item) for item in kept}
        return [result for result in results if id(result) in kept_ids]

    def _fuzzy_key(
        self, result: SearchResult, title_tail_patterns: list[re.Pattern[str]]
    ) -> str:
        title = result.title.strip()
        snippet = result.snippet.strip()

        base = fuzzy_normalize(strip_title_tails(title, title_tail_patterns))
        if len(base) < 8 and snippet:
            base = f"{base} {fuzzy_normalize(snippet[:240])}".strip()
        return base

    def _quality_score(
        self,
        result: SearchResult,
        config: SearchContextConfig,
    ) -> int:
        bonus = domain_bonus(result.domain, config.domain_bonus)
        return (
            bonus * 1000 + min(len(result.snippet), 1200) + min(len(result.title), 220)
        )

    @staticmethod
    def _extract_intent_tokens(query: str, config: SearchContextConfig) -> list[str]:
        lowered = query.lower()
        return [term for term in config.intent_terms if term.lower() in lowered]


    @staticmethod
    def _is_noise_extension(url: str, noise_extensions: set[str]) -> bool:
        if not noise_extensions or not url:
            return False
        path = urlparse(url).path.lower()
        return any(path.endswith(f".{ext}") for ext in noise_extensions)


__all__ = ["SearchPipeline"]
