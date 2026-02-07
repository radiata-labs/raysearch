from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

from . import scoring as scoring_mod
from .config import ScoreNormalizationConfig, WebEnrichmentConfig
from .models import SearchResult
from .tools import domain_bonus
from .utils import TextUtils

if TYPE_CHECKING:
    from .config import RankingConfig, SearchContextConfig

logger = logging.getLogger(__name__)


class ScoreProvider(Protocol):
    name: str

    def score(self, texts: list[str], *, query: str, **ctx: object) -> list[float]: ...


class BM25Provider:
    name = "bm25"

    def score(self, texts: list[str], *, query: str, **ctx: object) -> list[float]:
        return scoring_mod.bm25_scores(texts, query=query)


class VectorProvider:
    """Optional provider for vector similarity. Default behavior is "no signal"."""

    name = "vector"

    def __init__(self, scorer: Callable[..., list[float]] | None = None) -> None:
        self._scorer = scorer

    def score(self, texts: list[str], *, query: str, **ctx: object) -> list[float]:
        if not texts:
            return []
        if self._scorer is None:
            return [0.0 for _ in texts]
        try:
            out = self._scorer(texts, query=query, **ctx)
        except Exception:  # noqa: BLE001
            logger.exception("VectorProvider failed; returning zeros")
            return [0.0 for _ in texts]
        if not isinstance(out, list) or len(out) != len(texts):
            return [0.0 for _ in texts]
        return [float(x) for x in out]


_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_DATE_JP_RE = re.compile(r"\d{4}\u5e74\d{1,2}\u6708")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SEP_RE = re.compile(r"[\|/>\u00bb]")


class ScoringEngine:
    def __init__(
        self,
        *,
        score_norm_cfg: ScoreNormalizationConfig | None = None,
        vector_provider: ScoreProvider | None = None,
    ) -> None:
        self._norm_cfg = score_norm_cfg or ScoreNormalizationConfig()
        self._bm25 = BM25Provider()
        self._vector = vector_provider

    def normalize_chunk_scores(self, raw_scores: list[float]) -> list[float]:
        return scoring_mod.normalize_scores(raw_scores, self._norm_cfg)

    def _effective_weights(self, ranking_cfg: RankingConfig) -> dict[str, float]:
        weights = dict(ranking_cfg.weights_map())
        weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
        if not weights:
            return {"heuristic": 1.0}

        # Downgrade gracefully when BM25 isn't available.
        if weights.get("bm25") and not scoring_mod.BM25_AVAILABLE:
            weights.pop("bm25", None)

        total = sum(weights.values())
        if total <= 0:
            return {"heuristic": 1.0}
        return {k: float(v) / total for k, v in weights.items()}

    def rank_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        context_config: SearchContextConfig,
        ranking_config: RankingConfig,
        score_norm_cfg: ScoreNormalizationConfig | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        weights = self._effective_weights(ranking_config)
        heur_scores, hit_keywords = self._heuristic_result_scores(
            results,
            context_config=context_config,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )

        # Optional providers.
        providers: dict[str, list[float]] = {}
        if weights.get("bm25") and scoring_mod.BM25_AVAILABLE:
            docs = [f"{r.title} {r.snippet}" for r in results]
            providers["bm25"] = self._bm25.score(docs, query=query)

        if weights.get("vector") and self._vector is not None:
            docs = [f"{r.title} {r.snippet}" for r in results]
            providers["vector"] = self._vector.score(docs, query=query)

        raw_scores = self._blend_generic(
            heuristic=heur_scores,
            weights=weights,
            others=providers,
        )

        ranked_pairs = sorted(
            zip(raw_scores, results, hit_keywords, strict=False),
            key=lambda t: t[0],
            reverse=True,
        )
        ranked = [r for _, r, _ in ranked_pairs]
        ranked_hits = [hits for _, _, hits in ranked_pairs]
        ranked_raw = [float(s) for s, _, _ in ranked_pairs]

        norm_cfg = score_norm_cfg or self._norm_cfg
        norm = scoring_mod.normalize_scores(ranked_raw, norm_cfg)
        for i, result in enumerate(ranked):
            result.score = float(norm[i])
            result.hit_keywords = ranked_hits[i]

        return ranked

    def _blend_generic(
        self,
        *,
        heuristic: list[float],
        weights: dict[str, float],
        others: dict[str, list[float]],
    ) -> list[float]:
        n = len(heuristic)
        if n == 0:
            return []

        heur_w = float(weights.get("heuristic", 0.0))
        max_heur = max(heuristic) if heuristic else 1.0

        # Start with heuristic component.
        out = [float(h) * heur_w for h in heuristic]

        for name, scores in others.items():
            w = float(weights.get(name, 0.0))
            if w <= 0:
                continue
            if len(scores) != n:
                continue
            scaled = scoring_mod.rank_scale([float(s) for s in scores])
            for i in range(n):
                out[i] += float(scaled[i]) * w * float(max_heur)

        return out

    def _heuristic_result_scores(
        self,
        results: list[SearchResult],
        *,
        context_config: SearchContextConfig,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> tuple[list[float], list[list[str]]]:
        scores: list[float] = []
        all_hits: list[list[str]] = []
        for result in results:
            title = (result.title or "").lower()
            snippet = (result.snippet or "").lower()
            dom = (result.domain or "").lower()

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

            score += domain_bonus(dom, context_config.domain_bonus) * 2

            if len((result.snippet or "").strip()) < 60:
                score -= 2

            scores.append(float(score))
            all_hits.append(TextUtils.unique_preserve_order(hits))

        return scores, all_hits

    # ---- Chunk scoring (moved from WebEnricher) ----
    def score_and_filter_chunks(
        self,
        chunks: list[str],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        domain: str | None,
        context_config: SearchContextConfig,
        ranking_config: RankingConfig,
        web_config: WebEnrichmentConfig,
        title_patterns: list[re.Pattern[str]],
    ) -> list[tuple[float, str]]:
        if not chunks:
            return []

        weights = self._effective_weights(ranking_config)
        query_has_cjk = bool(_CJK_CHAR_RE.search(query))
        min_query_hits = max(0, int(web_config.scoring.min_query_hits))

        filtered: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks):
            if self._hard_drop_chunk(
                chunk,
                context_config=context_config,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                web_config=web_config,
            ):
                continue
            if (
                min_query_hits > 0
                and self._query_hit_count(chunk, query_tokens=query_tokens)
                < min_query_hits
            ):
                continue
            filtered.append((idx, chunk))

        if not filtered:
            return []

        dom_bonus = domain_bonus(domain, context_config.domain_bonus)

        # Heuristic component (always computed).
        heuristic_scores: list[float] = []
        for idx, chunk in filtered:
            heuristic_scores.append(
                self._heuristic_chunk_score(
                    chunk,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    domain_bonus=dom_bonus,
                    position=idx,
                    web_config=web_config,
                )
            )

        others: dict[str, list[float]] = {}
        if weights.get("bm25") and scoring_mod.BM25_AVAILABLE:
            docs = [c for _, c in filtered]
            others["bm25"] = self._bm25.score(docs, query=query)

        if weights.get("vector") and self._vector is not None:
            docs = [c for _, c in filtered]
            others["vector"] = self._vector.score(docs, query=query)

        final_scores = self._blend_generic(
            heuristic=heuristic_scores,
            weights=weights,
            others=others,
        )

        # NOTE: Chunk thresholds are independent from RankingConfig.min_* (result-level scale).
        min_score = max(
            float(web_config.scoring.min_chunk_score),
            float(web_config.scoring.min_final_score),
        )
        intent_missing_penalty = float(web_config.scoring.intent_missing_penalty)
        tpl_w = float(web_config.scoring.template_penalty_weight)
        tpl_b = float(web_config.scoring.template_penalty_bias)
        tpl_hard = float(web_config.scoring.template_hard_drop_threshold)

        scored: list[tuple[float, str]] = []
        for (_, chunk), score in zip(filtered, final_scores, strict=False):
            tpl = self.template_score(
                chunk,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="chunk",
            )
            if tpl >= tpl_hard:
                continue

            final = float(score) * (1.0 - tpl * tpl_w) - tpl * tpl_b

            if intent_tokens and not self._has_any_token(chunk, intent_tokens):
                final -= intent_missing_penalty

            if final < min_score:
                continue

            scored.append((float(final), chunk))

        return scored

    def _heuristic_chunk_score(
        self,
        chunk: str,
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        domain_bonus: int,
        position: int,
        web_config: WebEnrichmentConfig,
    ) -> float:
        if not chunk:
            return 0.0

        lowered = chunk.lower()
        hits = [t for t in query_tokens if t and t.lower() in lowered]
        if not hits:
            return 0.0

        unique_hits = set(hits)
        score = 0.0
        score += 6.0 * len(unique_hits)
        score += 1.5 * sum(min(lowered.count(t.lower()), 5) for t in unique_hits)
        score += 5.0 * sum(1 for t in intent_tokens if t and t.lower() in lowered)

        normalized_query = TextUtils.normalize_text(query)
        if normalized_query and normalized_query in TextUtils.normalize_text(chunk):
            score += float(web_config.scoring.phrase_bonus)

        score += float(domain_bonus)

        if len(chunk) < 80:
            score *= 0.85

        early_bonus = float(web_config.scoring.early_bonus)
        if early_bonus > 1.0:
            score *= early_bonus ** (-position)

        return float(score)

    @staticmethod
    def _has_any_token(text: str, tokens: list[str]) -> bool:
        lowered = (text or "").lower()
        return any(t and t.lower() in lowered for t in tokens)

    @staticmethod
    def _query_hit_count(chunk: str, *, query_tokens: list[str]) -> int:
        lowered = (chunk or "").lower()
        return sum(1 for t in set(query_tokens) if t and t.lower() in lowered)

    @staticmethod
    def _has_noise_word(text: str, context_config: SearchContextConfig) -> bool:
        lowered = TextUtils.normalize_text(text)
        for word in context_config.noise_words:
            if word and word.lower() in lowered:
                return True
        return False

    def _chunk_stats(self, chunk: str) -> tuple[int, float, int, float, float]:
        tokens = TextUtils.tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(_CJK_CHAR_RE.findall(chunk))
        cjk_ratio = cjk_count / max(1, len(chunk))
        return token_count, unique_ratio, digits, digits_ratio, cjk_ratio

    def _hard_drop_chunk(
        self,
        chunk: str,
        *,
        context_config: SearchContextConfig,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
        web_config: WebEnrichmentConfig,
    ) -> bool:
        if not chunk:
            return True

        normalized = TextUtils.normalize_text(chunk)
        if not normalized:
            return True

        if self._has_noise_word(chunk, context_config):
            return True

        token_count, unique_ratio, digits, digits_ratio, cjk_ratio = self._chunk_stats(
            chunk
        )

        if digits >= 18 and digits_ratio > 0.2:
            return True
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.06:
            return True

        tpl = self.template_score(
            chunk,
            query_has_cjk=query_has_cjk,
            title_patterns=title_patterns,
            mode="chunk",
        )
        return tpl >= float(web_config.scoring.template_hard_drop_threshold)

    def template_score(
        self,
        text: str,
        *,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
        mode: Literal["block", "chunk"],
    ) -> float:
        if not text:
            return 1.0

        lowered = TextUtils.normalize_text(text)
        if not lowered:
            return 1.0

        kw_strong = {
            "archive",
            "archives",
            "sitemap",
            "privacy",
            "terms",
            "cookie",
            "cookies",
            "\u30a2\u30fc\u30ab\u30a4\u30d6",
            "\u30b5\u30a4\u30c8\u30de\u30c3\u30d7",
            "\u30d7\u30e9\u30a4\u30d0\u30b7\u30fc",
            "\u5229\u7528\u898f\u7d04",
            "\u6708\u3092\u9078\u629e",
            "\u7ad9\u70b9\u5730\u56fe",
            "\u9690\u79c1",
            "\u6761\u6b3e",
        }
        kw_weak = {
            "next",
            "previous",
            "older",
            "newer",
            "page",
            "pages",
            "category",
            "categories",
            "tag",
            "tags",
            "\u4e0a\u4e00\u9875",
            "\u4e0b\u4e00\u9875",
            "\u4e00\u89a7",
            "\u76ee\u5f55",
        }

        kw_hits = 0.0
        for kw in kw_strong:
            if kw in lowered:
                kw_hits += 0.35
        for kw in kw_weak:
            if kw in lowered:
                kw_hits += 0.15
        kw_component = min(1.0, kw_hits)

        date_component = min(
            1.0,
            0.25 * len(_DATE_JP_RE.findall(text)) + 0.15 * len(_YEAR_RE.findall(text)),
        )

        token_count, unique_ratio, digits, digits_ratio, cjk_ratio = self._chunk_stats(
            text
        )
        uniq_component = 0.0
        if token_count >= 8 and unique_ratio < 0.45:
            uniq_component = min(1.0, (0.45 - unique_ratio) / 0.45)

        digit_component = 0.0
        if digits_ratio > 0.18:
            digit_component = min(1.0, (digits_ratio - 0.18) / 0.32)

        sep_component = 0.0
        sep_ratio = len(_SEP_RE.findall(text)) / max(1, len(text))
        if mode == "block" and sep_ratio > 0.03:
            sep_component = min(1.0, (sep_ratio - 0.03) / 0.10)
        elif mode == "chunk" and sep_ratio > 0.05:
            sep_component = min(1.0, (sep_ratio - 0.05) / 0.10)

        list_component = 0.0
        parts = [p for p in re.split(r"[\s\|/>\u00bb]+", lowered) if p]
        if len(parts) >= 20:
            short = sum(1 for p in parts if len(p) <= 3)
            if short / len(parts) > 0.6:
                list_component = 0.8

        lang_component = 0.0
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.10:
            lang_component = 0.35

        title_component = 0.0
        for pattern in title_patterns:
            if pattern.search(text):
                title_component = 0.6
                break

        tpl = 0.0
        for comp in (
            kw_component,
            date_component,
            digit_component,
            sep_component,
            uniq_component,
            list_component,
            lang_component,
            title_component,
        ):
            comp = max(0.0, min(1.0, float(comp)))
            tpl = 1.0 - (1.0 - tpl) * (1.0 - comp)

        if mode == "block" and len(lowered) <= 64 and kw_component >= 0.35:
            tpl = max(tpl, 0.9)

        return float(max(0.0, min(1.0, tpl)))


__all__ = ["ScoreProvider", "BM25Provider", "VectorProvider", "ScoringEngine"]
