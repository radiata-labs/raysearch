from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from .client import SearxngClient
from .config import RANKING_STRATEGIES, SearchConfig, SearchContextConfig
from .models import SearchContext, SearchResult
from .utils import PUNCTUATION_RE, TextUtils
from .web import chunk_text, fetch_url, html_to_text, score_chunk

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None
    BM25_AVAILABLE = False

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
    ) -> None:
        self.config = config
        self.client = client or SearxngClient(config)
        self._page_fetcher = page_fetcher

    def search_raw(
        self, query: str, *, params: Mapping[str, object] | None = None
    ) -> dict:
        """Fetch raw results from SearxNG (no processing)."""

        return self.client.search(query, params=params)

    def search_context(
        self,
        query: str,
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        depth: SearchDepth = "simple",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        max_chunk_chars: int = 800,
    ) -> SearchContext:
        """Search and build LLM-friendly context."""

        raw = self.search_raw(query, params=params)
        return self.build_context(
            raw,
            query,
            profile=profile,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
            show_source_domain=show_source_domain,
            show_source_url=show_source_url,
            show_source_engine=show_source_engine,
            fuzzy_threshold=fuzzy_threshold,
            depth=depth,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_chunk_chars=max_chunk_chars,
        )

    def search_json(
        self,
        query: str,
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        fuzzy_threshold: float | None = None,
        depth: SearchDepth = "simple",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
    ) -> dict[str, object]:
        """Search and return JSON data only."""
        raw = self.search_raw(query, params=params)
        query = TextUtils.clean_whitespace(query)
        if not query:
            return {}

        processed = self.process_response(
            raw,
            query,
            profile=profile,
            max_results=max_results,
            fuzzy_threshold=fuzzy_threshold,
            depth=depth,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
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
        *,
        params: Mapping[str, object] | None = None,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        depth: SearchDepth = "simple",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
        max_chunk_chars: int = 800,
    ) -> str:
        """Search and return markdown only."""

        raw = self.search_raw(query, params=params)
        return self.build_context(
            raw,
            query,
            profile=profile,
            max_results=max_results,
            max_snippet_chars=max_snippet_chars,
            show_source_domain=show_source_domain,
            show_source_url=show_source_url,
            show_source_engine=show_source_engine,
            fuzzy_threshold=fuzzy_threshold,
            depth=depth,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            max_chunk_chars=max_chunk_chars,
        ).markdown

    def build_context(
        self,
        searxng_response: Mapping[str, object],
        user_query: str,
        *,
        profile: str | None = None,
        max_results: int = 16,
        max_snippet_chars: int = 1000,
        show_source_domain: bool = True,
        show_source_url: bool = False,
        show_source_engine: bool = False,
        fuzzy_threshold: float | None = None,
        depth: SearchDepth = "simple",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
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
            profile=profile,
            max_results=max_results,
            fuzzy_threshold=fuzzy_threshold,
            depth=depth,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
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
        *,
        profile: str | None = None,
        max_results: int = 16,
        fuzzy_threshold: float | None = None,
        depth: SearchDepth = "simple",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
    ) -> list[SearchResult]:
        """Process a raw SearXNG response into ranked results (no HTTP)."""

        self._validate_depth(depth)
        if chunk_overlap >= chunk_chars:
            raise ValueError("chunk_overlap must be < chunk_chars")

        env_profile = os.getenv("SEARCH_PROFILE")
        explicit_profile = profile or env_profile
        profile_name = self._select_profile_name(query, explicit_profile)
        config = self._get_profile_config(profile_name)
        if fuzzy_threshold is not None:
            config = config.with_overrides(fuzzy_threshold=fuzzy_threshold)

        results_data = (response or {}).get("results", [])
        raw_results = list(results_data if isinstance(results_data, list) else [])

        processed = self._run_processing(query, config, raw_results)[:max_results]
        if depth != "simple" and processed:
            self._enrich_results_with_crawl(
                processed,
                query,
                config,
                depth=depth,
                chunk_chars=chunk_chars,
                chunk_overlap=chunk_overlap,
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

    @staticmethod
    def _depth_settings(depth: SearchDepth) -> tuple[float, int, int, int]:
        """Return (ratio, min_pages, max_pages, top_chunks_per_page)."""

        if depth == "simple":
            return (0.0, 0, 0, 0)
        if depth == "low":
            return (0.25, 1, 3, 2)
        if depth == "medium":
            return (0.50, 2, 6, 3)
        # high
        return (0.75, 3, 10, 5)

    def _enrich_results_with_crawl(
        self,
        results: list[SearchResult],
        query: str,
        config: SearchContextConfig,
        *,
        depth: SearchDepth,
        chunk_chars: int,
        chunk_overlap: int,
    ) -> None:
        """Fetch top pages and attach best-matching page chunks to results (in-place)."""

        ratio, min_pages, max_pages, top_k = self._depth_settings(depth)
        if ratio <= 0 or top_k <= 0:
            return

        n = len(results)
        target = int(math.ceil(n * ratio))
        m = max(min_pages, min(max_pages, target))
        m = min(m, n)
        if m <= 0:
            return

        query_tokens = self._split_query_tokens(query)
        intent_tokens = self._extract_intent_tokens(query, config)
        if not query_tokens:
            return

        max_workers = min(6, m)

        def crawl_one(  # noqa: PLR0911
            item: SearchResult,
        ) -> tuple[SearchResult, list[str], list[float], str | None]:
            url = (item.url or "").strip()
            if not url:
                return (item, [], [], "empty url")

            try:
                if self._page_fetcher is not None:
                    html = self._page_fetcher(url)
                    if not html:
                        return (item, [], [], "empty response")
                    page_text = html_to_text(html)
                else:
                    fetched = fetch_url(
                        url,
                        timeout=10.0,
                        max_bytes=2_000_000,
                        user_agent=self.config.searxng.user_agent,
                    )
                    if fetched.error:
                        return (item, [], [], fetched.error)
                    page_text = html_to_text(fetched.text)

                if not page_text:
                    return (item, [], [], "no text extracted")

                page_text = page_text[:50_000]
                chunks = chunk_text(
                    page_text,
                    chunk_chars=chunk_chars,
                    overlap=chunk_overlap,
                )
                if not chunks:
                    return (item, [], [], "no chunks")

                scored: list[tuple[float, str]] = []
                for ch in chunks:
                    s = score_chunk(
                        ch,
                        query_tokens=query_tokens,
                        intent_tokens=intent_tokens,
                    )
                    if s > 0:
                        scored.append((s, ch))

                if not scored:
                    return (item, [], [], "no matching chunks")

                scored.sort(key=lambda t: t[0], reverse=True)
                top = scored[:top_k]
                return (item, [c for _, c in top], [float(s) for s, _ in top], None)
            except Exception as exc:  # noqa: BLE001
                return (item, [], [], str(exc))

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for item in results[:m]:
                futures[ex.submit(crawl_one, item)] = item

            for fut in as_completed(futures):
                item, chunks, scores, err = fut.result()
                item.page_chunks = chunks
                item.page_chunk_scores = scores
                item.page_crawl_error = err

    def _run_processing(
        self,
        query: str,
        config: SearchContextConfig,
        raw_results: list[Mapping[str, object]],
    ) -> list[SearchResult]:
        normalized = [self._normalize_result(result) for result in raw_results]

        noise_extensions = {ext.lower().lstrip(".") for ext in config.noise_extensions}
        title_tail_patterns = self._compile_title_patterns(config.title_tail_patterns)
        domain_groups = config.domain_groups or {}

        query_tokens = self._split_query_tokens(query)
        intent_tokens = self._extract_intent_tokens(query, config)
        ranking_strategy = self._normalize_ranking_strategy(config.ranking.strategy)

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

        return self._rank_results(
            normalized,
            query=query,
            config=config,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
            ranking_strategy=ranking_strategy,
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

            if result.page_chunks:
                lines.append("- 页面片段:")
                for j, chunk in enumerate(result.page_chunks, 1):
                    rendered = chunk
                    if max_chunk_chars and len(rendered) > max_chunk_chars:
                        rendered = rendered[:max_chunk_chars].rstrip() + "..."
                    score_str = ""
                    if j - 1 < len(result.page_chunk_scores):
                        score_str = f"(score={result.page_chunk_scores[j - 1]:.2f}) "
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

        score += self._domain_bonus(domain, config)

        if core_hits == 0:
            return False

        if score < config.ranking.min_relevance_score:
            return False

        return not (
            intent_tokens
            and intent_hits == 0
            and score < config.ranking.min_intent_score
        )

    def _rank_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        config: SearchContextConfig,
        query_tokens: list[str],
        intent_tokens: list[str],
        ranking_strategy: str,
    ) -> list[SearchResult]:
        bm25_scaled: list[float] | None = None
        if ranking_strategy in {"bm25", "hybrid"}:
            bm25_scores = self._compute_bm25_scores(results, query)
            bm25_scaled = self._normalize_scores(bm25_scores)

        heuristic_scores = [
            self._heuristic_score(result, config, query_tokens, intent_tokens)
            for result in results
        ]
        max_heuristic = max(heuristic_scores) if heuristic_scores else 1.0

        for index, result in enumerate(results):
            heuristic_score = heuristic_scores[index]
            if ranking_strategy == "bm25" and bm25_scaled is not None:
                result.score = bm25_scaled[index] * max_heuristic
            elif ranking_strategy == "hybrid" and bm25_scaled is not None:
                bm25_weight, heuristic_weight = config.ranking.normalized_weights()
                result.score = (
                    bm25_scaled[index] * bm25_weight * max_heuristic
                    + heuristic_score * heuristic_weight
                )
            else:
                result.score = heuristic_score

        return sorted(results, key=lambda item: item.score, reverse=True)

    def _heuristic_score(
        self,
        result: SearchResult,
        config: SearchContextConfig,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> float:
        title = result.title.lower()
        snippet = result.snippet.lower()
        domain = result.domain.lower()

        score = 0
        hits: list[str] = []

        for token in query_tokens:
            if token in title:
                score += 12
                hits.append(token)
            elif token in snippet:
                score += 6
                hits.append(token)

        blob = f"{title} {snippet}"
        for token in intent_tokens:
            if token in blob:
                score += 5

        score += self._domain_bonus(domain, config) * 2

        if len(result.snippet.strip()) < 60:
            score -= 2

        result.hit_keywords = TextUtils.unique_preserve_order(hits)
        return float(score)

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
            candidate_site = self._canonical_site(candidate.domain, domain_groups)

            duplicate_index = -1
            for index, existing in enumerate(kept):
                existing_key = self._fuzzy_key(existing, title_tail_patterns)
                existing_site = self._canonical_site(existing.domain, domain_groups)

                threshold_value = (
                    threshold
                    if candidate_site == existing_site
                    else min(0.94, threshold + 0.06)
                )

                if (
                    self._hybrid_similarity(candidate_key, existing_key)
                    >= threshold_value
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

        base = self._fuzzy_normalize(
            self._strip_title_tails(title, title_tail_patterns)
        )
        if len(base) < 8 and snippet:
            base = f"{base} {self._fuzzy_normalize(snippet[:240])}".strip()
        return base

    def _quality_score(
        self,
        result: SearchResult,
        config: SearchContextConfig,
    ) -> int:
        bonus = self._domain_bonus(result.domain, config)
        return (
            bonus * 1000 + min(len(result.snippet), 1200) + min(len(result.title), 220)
        )

    @staticmethod
    def _canonical_site(
        domain: str, domain_groups: Mapping[str, tuple[str, ...]]
    ) -> str:
        normalized = (domain or "").lower()
        if not normalized:
            return "other"

        if domain_groups:
            for group, needles in domain_groups.items():
                if any(needle in normalized for needle in needles):
                    return group
            return normalized

        return normalized

    def _fuzzy_normalize(self, text: str) -> str:
        lowered = (text or "").lower()
        lowered = PUNCTUATION_RE.sub(" ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _hybrid_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        seq = SequenceMatcher(None, a, b).ratio()
        jac = TextUtils.jaccard(
            TextUtils.char_ngrams(a, 2), TextUtils.char_ngrams(b, 2)
        )
        return max(seq * 0.95, jac)

    @staticmethod
    def _split_query_tokens(query: str) -> list[str]:
        cleaned = query.strip()
        if not cleaned:
            return []

        parts = re.split(r"[\s,，、|]+", cleaned)
        parts = [part.strip().lower() for part in parts if part.strip()]
        tokens = [part for part in parts if len(part) >= 2]

        if len(tokens) < 2:
            cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
            grams: list[str] = []
            for run in cjk_runs:
                if len(run) <= 3:
                    grams.append(run)
                else:
                    grams.extend(TextUtils.ngrams(run, 2))
                    grams.extend(TextUtils.ngrams(run, 3))
            tokens.extend([gram.lower() for gram in grams])

        return TextUtils.unique_preserve_order(tokens)

    @staticmethod
    def _extract_intent_tokens(query: str, config: SearchContextConfig) -> list[str]:
        lowered = query.lower()
        return [term for term in config.intent_terms if term.lower() in lowered]

    @staticmethod
    def _domain_bonus(domain: str, config: SearchContextConfig | None) -> int:
        if not config:
            return 0
        normalized = (domain or "").lower()
        if not normalized:
            return 0
        if normalized in config.domain_bonus:
            return config.domain_bonus[normalized]
        for key, value in config.domain_bonus.items():
            if normalized.endswith(key):
                return value
        return 0

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _normalize_ranking_strategy(self, strategy: str) -> str:
        normalized = (strategy or "heuristic").lower()
        if normalized not in RANKING_STRATEGIES:
            logger.warning(
                "Unknown ranking strategy '%s'; fallback to heuristic.", normalized
            )
            return "heuristic"
        if normalized in {"bm25", "hybrid"} and not BM25_AVAILABLE:
            logger.warning("BM25 not available; fallback to heuristic.")
            return "heuristic"
        return normalized

    @staticmethod
    def _compile_title_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            if not pattern:
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning("Invalid title_tail_patterns regex: %s", pattern)
        return compiled

    @staticmethod
    def _strip_title_tails(title: str, patterns: list[re.Pattern[str]]) -> str:
        cleaned = title
        for pattern in patterns:
            cleaned = pattern.sub("", cleaned).strip()
        return cleaned

    @staticmethod
    def _is_noise_extension(url: str, noise_extensions: set[str]) -> bool:
        if not noise_extensions or not url:
            return False
        path = urlparse(url).path.lower()
        return any(path.endswith(f".{ext}") for ext in noise_extensions)

    def _compute_bm25_scores(
        self, results: list[SearchResult], query: str
    ) -> list[float]:
        if not BM25_AVAILABLE or BM25Okapi is None:
            return [0.0 for _ in results]

        corpus = [TextUtils.tokenize(f"{r.title} {r.snippet}") for r in results]
        if not corpus:
            return [0.0 for _ in results]

        query_tokens = TextUtils.tokenize(query)
        if not query_tokens:
            return [0.0 for _ in results]

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores]


__all__ = ["SearchPipeline"]
