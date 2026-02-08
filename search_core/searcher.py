"""Search pipeline (sync + async) built on top of SearxNG.

`Searcher` and `AsyncSearcher` orchestrate:
1) fetching raw JSON results from SearxNG
2) profile selection (auto match or explicit)
3) normalization, filtering, dedupe, ranking
4) optional web page crawling + chunk scoring enrichment
5) rendering output (JSON/markdown/context)
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlparse

from search_core.client import AsyncSearxngClient, SearxngClient
from search_core.config import SearchConfig, SearchContextConfig
from search_core.enrich import AsyncWebEnricher, WebEnricher
from search_core.models import SearchContext, SearchResult
from search_core.scorer import ScoringEngine
from search_core.text import TextUtils
from search_core.tools import (
    canonical_site,
    compile_patterns,
    extract_intent_tokens,
    fuzzy_normalize,
    has_noise_word,
    hybrid_similarity,
    strip_title_tails,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

SearchDepth = Literal["simple", "low", "medium", "high"]


class _SearcherBase:
    """Shared, non-I/O pipeline logic for both sync and async variants.

    Concrete pipelines are responsible for performing I/O:
    - Searcher: sync HTTP + sync web enrichment
    - AsyncSearcher: async HTTP + async web enrichment
    """

    config: SearchConfig
    scorer: ScoringEngine

    def __init__(
        self,
        config: SearchConfig,
        *,
        scorer: ScoringEngine | None = None,
    ) -> None:
        self.config = config
        self.scorer = scorer or ScoringEngine(self.config.scoring)

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

    def _validate_depth(self, depth: SearchDepth) -> None:
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
        intent_tokens = extract_intent_tokens(query, config)

        normalized = [
            result
            for result in normalized
            if self._is_not_noise(result, config, noise_extensions)
        ]
        if not normalized:
            return []

        normalized = [
            result for result in normalized if self._is_relevant(result, query_tokens)
        ]
        if not normalized:
            return []

        normalized = self._dedupe_exact(normalized)
        normalized = self._dedupe_fuzzy(
            normalized,
            config.fuzzy_threshold,
            title_tail_patterns=title_tail_patterns,
            domain_groups=domain_groups,
        )

        return self._rank_results(
            normalized,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )

    def _rerank_with_page_scores(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> list[SearchResult]:
        """Blend snippet score with page score and rerank.

        Fixed weights (no config):
        - snippet: 0.4
        - page: 0.6
        """
        if not results:
            return []

        page_docs: list[str] = []
        has_any_page = False
        for r in results:
            if r.page and r.page.chunks:
                doc = TextUtils.clean_whitespace(
                    " ".join(c.text for c in r.page.chunks)
                )
                page_docs.append(doc)
                if doc:
                    has_any_page = True
            else:
                page_docs.append("")

        if not has_any_page:
            return results

        page_scored = self.scorer.score(
            page_docs,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        page_scores = [float(s) for s, _ in page_scored]

        combined_raw: list[float] = []
        for i, r in enumerate(results):
            snippet_s = float(r.score)
            page_s = float(page_scores[i]) if i < len(page_scores) else 0.0
            combined_raw.append(0.4 * snippet_s + 0.6 * page_s)

        combined_norm = self.scorer.normalize_scores(combined_raw)
        if combined_norm and max(combined_norm) <= 0.0 and max(combined_raw) > 0.0:
            combined_norm = [0.5 for _ in combined_norm]

        for i, r in enumerate(results):
            if i < len(combined_norm):
                r.score = float(combined_norm[i])

        return sorted(results, key=lambda r: float(r.score), reverse=True)

    def _rank_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> list[SearchResult]:
        if not results:
            return []

        docs = [TextUtils.clean_whitespace(f"{r.title} {r.snippet}") for r in results]
        scored = self.scorer.score(
            docs,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        scores = [float(s) for s, _ in scored]

        hit_keywords: list[list[str]] = []
        for r in results:
            title = (r.title or "").lower()
            snippet = (r.snippet or "").lower()
            hits: list[str] = [
                token for token in query_tokens if token in title or token in snippet
            ]
            hit_keywords.append(TextUtils.unique_preserve_order(hits))

        ranked_pairs = sorted(
            zip(scores, results, hit_keywords, strict=False),
            key=lambda t: t[0],
            reverse=True,
        )
        ranked = [r for _, r, _ in ranked_pairs]
        ranked_hits = [hits for _, _, hits in ranked_pairs]
        ranked_scores = [float(s) for s, _, _ in ranked_pairs]

        for i, result in enumerate(ranked):
            result.score = float(ranked_scores[i])
            result.hit_keywords = ranked_hits[i]

        return ranked

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

    def _render_empty(self, reason: str) -> str:
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

        if any(
            urlparse(url).path.lower().endswith(f".{ext}") for ext in noise_extensions
        ):
            return False

        if has_noise_word(blob, config):
            return False

        return not (len(title) < 2 and len(snippet) < 40)

    def _is_relevant(
        self,
        result: SearchResult,
        query_tokens: list[str],
    ) -> bool:
        title = result.title.lower()
        snippet = result.snippet.lower()
        core_hits = 0
        for token in query_tokens:
            if token in title or token in snippet:
                core_hits += 1

        return core_hits != 0

    def _dedupe_fuzzy(
        self,
        results: list[SearchResult],
        threshold: float,
        *,
        title_tail_patterns: list[re.Pattern[str]],
        domain_groups: Mapping[str, tuple[str, ...]],
    ) -> list[SearchResult]:
        kept: list[SearchResult] = []

        for candidate in sorted(
            results,
            key=self._quality_score,
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

                if hybrid_similarity(candidate_key, existing_key) >= threshold_value:
                    duplicate_index = index
                    break

            if duplicate_index < 0:
                kept.append(candidate)
            elif self._quality_score(candidate) > self._quality_score(
                kept[duplicate_index]
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
    ) -> int:
        return min(len(result.snippet), 1200) + min(len(result.title), 220)

    def _dedupe_exact(self, results: list[SearchResult]) -> list[SearchResult]:
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

    def _extract_domain(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc
        if not host and parsed.path:
            host = parsed.path.split("/")[0]
        return host.lower()


class Searcher(_SearcherBase):
    """High-level search pipeline: profile selection + result processing + rendering."""

    client: SearxngClient

    def __init__(
        self,
        config: SearchConfig,
        *,
        client: SearxngClient | None = None,
        page_fetcher: Callable[[str], bytes | str] | None = None,
        scorer: ScoringEngine | None = None,
        web_enricher: WebEnricher | None = None,
    ) -> None:
        super().__init__(config, scorer=scorer)
        self.client = client or SearxngClient(self.config)
        self.web_enricher = web_enricher or WebEnricher(
            self.config.web_enrichment,
            user_agent=self.config.searxng.user_agent,
            fetcher=page_fetcher,
            scorer=self.scorer,
            min_score=float(self.config.score_filter.min_score),
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

        processed = self._run_processing(query, config, raw_results)
        if not processed:
            return []

        query_tokens = [t for t in TextUtils.tokenize(query) if len(t) >= 2]
        intent_tokens = extract_intent_tokens(query, config)

        if depth != "simple" and self.config.web_enrichment.enabled and query_tokens:
            preset = self.config.web_enrichment.depth_presets.get(depth)
            if preset is None:
                logger.warning(
                    "No depth preset configured for '%s'; skipping web enrichment.",
                    depth,
                )
            else:
                self.web_enricher.enrich_results(
                    processed,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    context_config=config,
                    preset=preset,
                    chunk_target_chars=chunk_target_chars,
                    chunk_overlap_sentences=chunk_overlap_sentences,
                    min_chunk_chars=min_chunk_chars,
                )

                # If pages were fetched successfully for some results, let page quality
                # influence the final ordering (page is weighted higher than snippet).
                processed = self._rerank_with_page_scores(
                    processed,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                )

        # Apply the global score floor only once at the end so page scores can
        # "rescue" weaker snippet results.
        min_score = float(self.config.score_filter.min_score)
        processed = [
            r for r in processed if float(r.score) > 0.0 and float(r.score) >= min_score
        ]

        return processed[:max_results]


class AsyncSearcher(_SearcherBase):
    """Async pipeline variant: async SearxNG search + async web enrichment."""

    client: AsyncSearxngClient
    web_enricher: AsyncWebEnricher

    def __init__(
        self,
        config: SearchConfig,
        *,
        client: AsyncSearxngClient | None = None,
        apage_fetcher: Callable[[str], Awaitable[bytes | str]] | None = None,
        scorer: ScoringEngine | None = None,
        web_enricher: AsyncWebEnricher | None = None,
    ) -> None:
        super().__init__(config, scorer=scorer)
        self.client = client or AsyncSearxngClient(self.config)
        self.web_enricher = web_enricher or AsyncWebEnricher(
            self.config.web_enrichment,
            user_agent=self.config.searxng.user_agent,
            afetcher=apage_fetcher,
            scorer=self.scorer,
            min_score=float(self.config.score_filter.min_score),
        )

    async def aclose(self) -> None:
        """Close async resources owned by this instance."""
        await self.client.aclose()
        await self.web_enricher.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()

    async def asearch_raw(
        self, query: str, *, params: Mapping[str, object] | None = None
    ) -> dict:
        """Async fetch raw results from SearxNG (no processing)."""
        return await self.client.asearch(query, params=params)

    async def asearch_json(
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
        """Async search and return JSON data only."""
        raw = await self.asearch_raw(query, params=params)
        query = TextUtils.clean_whitespace(query)
        if not query:
            return {}

        processed = await self.aprocess_response(
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

    async def asearch_markdown(
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
        """Async search and return markdown only."""
        raw = await self.asearch_raw(query, params=params)
        ctx = await self.abuild_context(
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
        return ctx.markdown

    async def asearch_context(
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
        """Async search and build LLM-friendly context."""
        raw = await self.asearch_raw(query, params=params)
        return await self.abuild_context(
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

    async def abuild_context(
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
        """Async build context from a raw SearxNG response (no SearxNG HTTP in here)."""
        query = TextUtils.clean_whitespace(user_query)
        if not query:
            return SearchContext(
                query="",
                results=[],
                json_data={},
                markdown=self._render_empty("用户问题为空。"),
            )

        processed = await self.aprocess_response(
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

    async def aprocess_response(
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
        """Async process a raw SearxNG response into ranked results."""
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

        processed = self._run_processing(query, config, raw_results)
        if not processed:
            return []

        query_tokens = [t for t in TextUtils.tokenize(query) if len(t) >= 2]
        intent_tokens = extract_intent_tokens(query, config)

        if depth != "simple" and self.config.web_enrichment.enabled and query_tokens:
            preset = self.config.web_enrichment.depth_presets.get(depth)
            if preset is None:
                logger.warning(
                    "No depth preset configured for '%s'; skipping web enrichment.",
                    depth,
                )
            else:
                await self.web_enricher.aenrich_results(
                    processed,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    context_config=config,
                    preset=preset,
                    chunk_target_chars=chunk_target_chars,
                    chunk_overlap_sentences=chunk_overlap_sentences,
                    min_chunk_chars=min_chunk_chars,
                )

                processed = self._rerank_with_page_scores(
                    processed,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                )

        min_score = float(self.config.score_filter.min_score)
        processed = [
            r for r in processed if float(r.score) > 0.0 and float(r.score) >= min_score
        ]

        return processed[:max_results]


__all__ = ["Searcher", "AsyncSearcher"]
