from __future__ import annotations

import bisect
import heapq
import math
import re
import statistics
from collections import Counter
from typing_extensions import override

from serpsage.components.base import ComponentMeta
from serpsage.components.rank.base import HeuristicRankSettings, RankerBase, RankMode
from serpsage.load import register_component
from serpsage.utils import normalize_text

_HEURISTIC_META = ComponentMeta(
    family="rank",
    name="heuristic",
    version="1.0.0",
    summary="Rule-based text ranker.",
    provides=("rank.heuristic_engine",),
    config_model=HeuristicRankSettings,
    config_optional=True,
)


@register_component(meta=_HEURISTIC_META)
class HeuristicRanker(RankerBase[HeuristicRankSettings]):
    meta = _HEURISTIC_META
    _ASCII_TOKEN_RE = re.compile(r"^[a-z0-9_]+$")
    _WORD_CHAR_CLASS = r"[a-z0-9_]"
    _DEFAULT_LOCAL_IDF_BOOST = 0.35
    _DECILE_SIGMA = 2.5631031310892007

    def __init__(self) -> None:
        self._ascii_pattern_cache: dict[str, re.Pattern[str]] = {}

    @override
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        _ = self._resolve_mode(mode, supported=("retrieve",))
        cfg = self.config
        normalized_query_tokens = self._normalize_query_tokens(query_tokens)
        if not normalized_query_tokens:
            return [0.0 for _ in texts]
        unique_query_tokens = self._unique_preserve_order(normalized_query_tokens)
        normalized_texts = [normalize_text(text) for text in texts]
        token_hits_per_text = [
            self._collect_token_hits(text, unique_query_tokens)
            for text in normalized_texts
        ]
        doc_freqs = self._compute_doc_freqs(token_hits_per_text, unique_query_tokens)
        query_terms = self._build_query_terms(
            ordered_query_tokens=normalized_query_tokens,
            doc_freqs=doc_freqs,
            num_docs=len(normalized_texts),
            local_idf_boost=float(
                getattr(cfg, "local_idf_boost", self._DEFAULT_LOCAL_IDF_BOOST)
            ),
        )
        if not query_terms:
            return [0.0 for _ in texts]
        total_query_weight = sum(weight for _, weight in query_terms)
        if total_query_weight <= 0.0:
            return [0.0 for _ in texts]
        normalized_query = " ".join(normalized_query_tokens)
        query_bigrams = self._build_query_ngrams(normalized_query_tokens, 2)
        query_trigrams = self._build_query_ngrams(normalized_query_tokens, 3)
        max_count = max(1, int(cfg.max_count_per_token))
        max_count_log = math.log1p(float(max_count))
        unique_hit_weight = float(cfg.unique_hit_weight)
        count_weight = float(cfg.count_weight)
        early_bonus = float(cfg.early_bonus)
        raw_scores: list[float] = []
        for normalized_text, token_hits in zip(
            normalized_texts, token_hits_per_text, strict=False
        ):
            if not normalized_text or not token_hits:
                raw_scores.append(0.0)
                continue
            weighted_coverage_sum = 0.0
            tf_quality_sum = 0.0
            positions_per_term: list[list[int]] = []
            for token, weight in query_terms:
                positions = token_hits.get(token, [])
                if not positions:
                    continue
                weighted_coverage_sum += weight
                positions_per_term.append(positions)
                capped_count = min(len(positions), max_count)
                tf_quality_sum += (
                    math.log1p(float(capped_count)) / max_count_log
                ) * weight
            query_coverage = weighted_coverage_sum / total_query_weight
            if query_coverage <= 0.0:
                raw_scores.append(0.0)
                continue
            query_tf_quality = tf_quality_sum / total_query_weight
            query_tf_quality = min(1.0, max(0.0, float(query_tf_quality)))
            query_proximity = self._compute_min_cover_proximity(positions_per_term)
            in_order_ratio = self._compute_in_order_match_ratio(
                normalized_query_tokens,
                token_hits,
            )
            full_phrase_hit = (
                1.0
                if normalized_query and normalized_text.find(normalized_query) >= 0
                else 0.0
            )
            bigram_ratio = self._compute_ngram_hit_ratio(normalized_text, query_bigrams)
            trigram_ratio = self._compute_ngram_hit_ratio(
                normalized_text, query_trigrams
            )
            lexical_core = (
                0.50 * query_coverage
                + 0.20 * query_tf_quality
                + 0.15 * query_proximity
                + 0.15 * in_order_ratio
            )
            phrase_quality = (
                0.60 * full_phrase_hit + 0.25 * bigram_ratio + 0.15 * trigram_ratio
            )
            base = (unique_hit_weight * lexical_core) + (count_weight * phrase_quality)
            base *= math.sqrt(max(0.0, query_coverage))
            early_position_score = self._compute_weighted_first_position_score(
                query_terms,
                token_hits,
                len(normalized_text),
            )
            early_gain = 1.0 + max(0.0, early_bonus - 1.0) * (early_position_score**2)
            raw_scores.append(max(0.0, float(base * early_gain)))
        return self._normalize_scores(raw_scores)

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        cfg = self.config
        cleaned = [max(0.0, self._safe_float(s)) for s in (scores or [])]
        if not cleaned:
            return []
        out = [0.0 for _ in cleaned]
        pos_idx = [i for i, x in enumerate(cleaned) if x > 0.0]
        if not pos_idx:
            return out
        if len(pos_idx) == 1:
            out[pos_idx[0]] = 0.5
            return out
        pos_vals = [cleaned[i] for i in pos_idx]
        log_vals = [math.log1p(x) for x in pos_vals]
        if len(log_vals) == 1:
            out[pos_idx[0]] = 0.5
            return out
        temp = max(1e-6, self._safe_float(getattr(cfg, "temperature", 1.0)) or 1.0)
        min_items = max(2, int(getattr(cfg, "min_items_for_sigmoid", 4)))
        flat_eps = max(0.0, self._safe_float(getattr(cfg, "flat_spread_eps", 1e-6)))
        z_clip = max(0.0, self._safe_float(getattr(cfg, "z_clip", 4.0)))
        sorted_vals = sorted(log_vals)
        spread = sorted_vals[-1] - sorted_vals[0]
        if spread <= flat_eps:
            pos_norm = [0.5 for _ in log_vals]
        elif len(log_vals) < min_items:
            pos_norm = self._rank_scales(log_vals)
        else:
            median = statistics.median(sorted_vals)
            mad = statistics.median([abs(x - median) for x in log_vals])
            p10 = self._quantile_sorted(sorted_vals, 0.10)
            p90 = self._quantile_sorted(sorted_vals, 0.90)
            sigma_from_mad = 1.4826 * mad
            sigma_from_decile = (p90 - p10) / self._DECILE_SIGMA if p90 > p10 else 0.0
            scale = max(sigma_from_mad, sigma_from_decile, 1e-12)
            rank_part = self._rank_scales(log_vals)
            pos_norm = []
            for i, x in enumerate(log_vals):
                z = (x - median) / (scale * temp)
                if z_clip > 0.0:
                    z = max(-z_clip, min(z_clip, z))
                sig_part = 1.0 / (1.0 + math.exp(-z))
                if p90 - p10 <= flat_eps:
                    density_part = 0.5
                else:
                    density_part = (x - p10) / (p90 - p10)
                    density_part = min(1.0, max(0.0, density_part))
                score = 0.55 * sig_part + 0.30 * rank_part[i] + 0.15 * density_part
                pos_norm.append(min(1.0, max(0.0, float(score))))
        for idx, value in zip(pos_idx, pos_norm, strict=False):
            out[idx] = min(1.0, max(0.0, self._safe_float(value)))
        return out

    def _rank_scales(self, scores: list[float]) -> list[float]:
        cleaned = [self._safe_float(s) for s in (scores or [])]
        n = len(cleaned)
        if n == 0:
            return []
        if n == 1:
            return [0.5 if cleaned[0] > 0.0 else 0.0]
        if max(cleaned) - min(cleaned) < 1e-12:
            return [0.5 for _ in cleaned]
        ordered = sorted(enumerate(cleaned), key=lambda item: item[1], reverse=True)
        out = [0.0 for _ in cleaned]
        i = 0
        while i < n:
            j = i + 1
            while j < n and ordered[j][1] == ordered[i][1]:
                j += 1
            avg_pos = (i + j - 1) / 2.0
            percentile = 1.0 - (avg_pos / (n - 1))
            for k in range(i, j):
                out[ordered[k][0]] = float(percentile)
            i = j
        return out

    def _safe_float(self, value: object) -> float:
        try:
            x = float(value if isinstance(value, (int, float, str)) else 0.0)
        except Exception:
            return 0.0
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x

    def _normalize_query_tokens(self, tokens: list[str]) -> list[str]:
        return [
            normalized for tok in tokens or [] if (normalized := normalize_text(tok))
        ]

    def _unique_preserve_order(self, tokens: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out

    def _collect_token_hits(
        self,
        normalized_text: str,
        unique_query_tokens: list[str],
    ) -> dict[str, list[int]]:
        if not normalized_text:
            return {}
        hits: dict[str, list[int]] = {}
        for tok in unique_query_tokens:
            positions = self._find_occurrences(normalized_text, tok)
            if positions:
                hits[tok] = positions
        return hits

    def _find_occurrences(self, text: str, token: str) -> list[int]:
        if not text or not token:
            return []
        if self._ASCII_TOKEN_RE.fullmatch(token):
            pattern = self._ascii_pattern_cache.get(token)
            if pattern is None:
                pattern = re.compile(
                    rf"(?<!{self._WORD_CHAR_CLASS}){re.escape(token)}"
                    rf"(?!{self._WORD_CHAR_CLASS})"
                )
                self._ascii_pattern_cache[token] = pattern
            return [m.start() for m in pattern.finditer(text)]
        positions: list[int] = []
        cursor = 0
        while True:
            pos = text.find(token, cursor)
            if pos < 0:
                break
            positions.append(pos)
            cursor = pos + 1
        return positions

    def _compute_doc_freqs(
        self,
        token_hits_per_text: list[dict[str, list[int]]],
        unique_tokens: list[str],
    ) -> dict[str, int]:
        doc_freqs = dict.fromkeys(unique_tokens, 0)
        for token_hits in token_hits_per_text:
            for tok in unique_tokens:
                if token_hits.get(tok):
                    doc_freqs[tok] += 1
        return doc_freqs

    def _build_query_terms(
        self,
        *,
        ordered_query_tokens: list[str],
        doc_freqs: dict[str, int],
        num_docs: int,
        local_idf_boost: float,
    ) -> list[tuple[str, float]]:
        if not ordered_query_tokens:
            return []
        counts = Counter(ordered_query_tokens)
        unique_tokens = self._unique_preserve_order(ordered_query_tokens)
        raw_idfs: dict[str, float] = {}
        for tok in unique_tokens:
            df = float(doc_freqs.get(tok, 0))
            raw_idfs[tok] = math.log1p((float(num_docs) + 1.0) / (df + 1.0))
        max_raw_idf = max(raw_idfs.values(), default=1.0)
        terms: list[tuple[str, float]] = []
        for tok in unique_tokens:
            repeat_weight = 1.0 + min(1.0, float(counts[tok] - 1))
            idf_norm = raw_idfs[tok] / max_raw_idf if max_raw_idf > 0.0 else 0.0
            shape_weight = 1.0
            if any(ch.isdigit() for ch in tok):
                shape_weight += 0.10
            elif self._ASCII_TOKEN_RE.fullmatch(tok) and (len(tok) >= 12 or "_" in tok):
                shape_weight += 0.05
            weight = repeat_weight * (1.0 + local_idf_boost * idf_norm) * shape_weight
            terms.append((tok, weight))
        return terms

    def _build_query_ngrams(self, tokens: list[str], n: int) -> list[str]:
        if n <= 0 or len(tokens) < n:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n]).strip()
            if not gram or gram in seen:
                continue
            seen.add(gram)
            out.append(gram)
        return out

    def _compute_ngram_hit_ratio(self, text: str, ngrams: list[str]) -> float:
        if not text or not ngrams:
            return 0.0
        hits = sum(1 for gram in ngrams if text.find(gram) >= 0)
        return float(hits) / float(len(ngrams))

    def _compute_in_order_match_ratio(
        self,
        ordered_query_tokens: list[str],
        token_hits: dict[str, list[int]],
    ) -> float:
        if not ordered_query_tokens:
            return 0.0
        matched = 0
        current_pos = -1
        for tok in ordered_query_tokens:
            positions = token_hits.get(tok)
            if not positions:
                continue
            next_idx = bisect.bisect_right(positions, current_pos)
            if next_idx >= len(positions):
                continue
            current_pos = positions[next_idx]
            matched += 1
        return float(matched) / float(len(ordered_query_tokens))

    def _compute_min_cover_proximity(
        self, positions_per_term: list[list[int]]
    ) -> float:
        non_empty = [positions for positions in positions_per_term if positions]
        num_terms = len(non_empty)
        if num_terms == 0:
            return 0.0
        if num_terms == 1:
            return 0.5
        heap: list[tuple[int, int, int]] = []
        current_max = -1
        for term_idx, positions in enumerate(non_empty):
            pos = positions[0]
            heap.append((pos, term_idx, 0))
            current_max = max(current_max, pos)
        heapq.heapify(heap)
        best_span = float("inf")
        while heap:
            current_min, term_idx, pos_idx = heapq.heappop(heap)
            best_span = min(best_span, float(current_max - current_min))
            next_idx = pos_idx + 1
            if next_idx >= len(non_empty[term_idx]):
                break
            next_pos = non_empty[term_idx][next_idx]
            current_max = max(current_max, next_pos)
            heapq.heappush(heap, (next_pos, term_idx, next_idx))
        if best_span == float("inf"):
            return 0.5
        ideal_span = float(num_terms - 1)
        excess_span = max(0.0, best_span - ideal_span)
        softness = max(4.0, float(num_terms) * 2.0)
        return 1.0 / (1.0 + excess_span / softness)

    def _compute_weighted_first_position_score(
        self,
        query_terms: list[tuple[str, float]],
        token_hits: dict[str, list[int]],
        text_length: int,
    ) -> float:
        if not query_terms or text_length <= 1:
            return 1.0
        weighted_pos_sum = 0.0
        weighted_hit_sum = 0.0
        for token, weight in query_terms:
            positions = token_hits.get(token)
            if not positions:
                continue
            weighted_pos_sum += float(positions[0]) * weight
            weighted_hit_sum += weight
        if weighted_hit_sum <= 0.0:
            return 0.0
        avg_pos = weighted_pos_sum / weighted_hit_sum
        pos_ratio = avg_pos / max(1.0, float(text_length - 1))
        pos_ratio = min(1.0, max(0.0, pos_ratio))
        return 1.0 - pos_ratio

    def _quantile_sorted(self, sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        if len(sorted_vals) == 1:
            return float(sorted_vals[0])
        q = min(1.0, max(0.0, q))
        pos = q * (len(sorted_vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(sorted_vals[lo])
        frac = pos - lo
        return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


__all__ = ["HeuristicRanker"]
