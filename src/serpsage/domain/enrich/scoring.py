from __future__ import annotations

import math
from typing import TYPE_CHECKING

from serpsage.core.tuning import chunk_profile_for_depth
from serpsage.text.normalize import normalize_text
from serpsage.text.similarity import is_duplicate_text

if TYPE_CHECKING:
    from serpsage.contracts.services import RankerBase
    from serpsage.settings.models import AppSettings, ProfileSettings


class EnrichScoringMixin:
    settings: AppSettings
    ranker: RankerBase

    def __init__(self, *, settings: AppSettings, ranker: RankerBase) -> None:
        self.settings = settings
        self.ranker = ranker

    async def score_chunks(
        self,
        *,
        chunks: list[str],
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        profile: ProfileSettings,
        depth: str = "medium",
    ) -> tuple[list[tuple[float, str]], dict[str, int | float]]:
        cfg = chunk_profile_for_depth(depth)

        stats: dict[str, int | float] = {
            "chunks_raw_count": int(len(chunks)),
        }

        chunks = self._filter_chunks(
            chunks,
            query_tokens=query_tokens,
            profile=profile,
            min_query_token_hits=int(cfg.min_query_token_hits),
        )

        stats["chunks_after_filter"] = int(len(chunks))

        scores = await self.ranker.score_texts(
            texts=chunks,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )

        if scores:
            stats["scores_min"] = float(min(scores))
            stats["scores_max"] = float(max(scores))

        scored, post_stats = self._post_process_chunk_scores(
            chunks,
            base_scores=scores,
            early_bonus=float(cfg.early_bonus),
            dedupe_threshold=float(profile.fuzzy_threshold),
            min_score=float(cfg.min_chunk_score),
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        stats.update(post_stats)
        stats["chunks_candidates"] = int(len(chunks))
        return scored, stats

    def _filter_chunks(
        self,
        chunks: list[str],
        *,
        query_tokens: list[str],
        profile: ProfileSettings,
        min_query_token_hits: int,
    ) -> list[str]:
        filtered: list[str] = []
        for chunk in chunks:
            lowered = normalize_text(chunk)
            has_noise = False
            for noise_word in profile.noise_words:
                nl = noise_word.lower()
                if nl and nl in lowered:
                    has_noise = True
                    break
            if has_noise:
                continue
            hit: int = 0
            for t in query_tokens:
                tl = t.lower()
                if tl in lowered:
                    hit += 1
            if hit >= int(min_query_token_hits):
                filtered.append(chunk)
        return filtered

    def _get_hit_location(
        self,
        chunk: str,
        *,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> float:
        lowered = normalize_text(chunk)
        query_hit_poses = []
        for t in query_tokens or []:
            tl = t.lower()
            pos = lowered.find(tl)
            if pos >= 0:
                query_hit_poses.append(pos)
        if not query_hit_poses:
            return 1.0
        first_hit_pos = float(min(query_hit_poses) / len(lowered))
        if intent_tokens:
            intent_hit_poses = []
            for t in intent_tokens:
                tl = t.lower()
                pos = lowered.find(tl)
                if pos >= 0:
                    intent_hit_poses.append(pos)
            if intent_hit_poses:
                first_intent_hit_pos = float(min(intent_hit_poses) / len(lowered))
                return min(first_hit_pos, first_intent_hit_pos)
        return first_hit_pos

    def _post_process_chunk_scores(
        self,
        chunks: list[str],
        *,
        base_scores: list[float],
        early_bonus: float,
        dedupe_threshold: float,
        min_score: float,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> tuple[list[tuple[float, str]], dict[str, int]]:
        stats: dict[str, int] = {"drop_min_score": 0, "drop_duplicate": 0}

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        def logit(p: float, eps: float = 1e-6) -> float:
            p = max(eps, min(1.0 - eps, float(p)))
            return math.log(p / (1.0 - p))

        def process_location(loc: float, k: float = 2.5) -> float:
            return 0.5 * math.tanh(k * loc - k / 2.0) + 0.25

        scored: list[tuple[float, str]] = []
        for chunk, base_score in zip(chunks, base_scores, strict=False):
            scored.append((base_score, chunk))

        scored.sort(key=lambda t: t[0], reverse=True)

        kept: list[tuple[float, str]] = []
        th = float(dedupe_threshold)
        for score, chunk in scored:
            if float(score) <= 0.0:
                continue
            if float(score) < float(min_score):
                stats["drop_min_score"] += 1
                continue
            if is_duplicate_text(chunk, [c for _, c in kept], threshold=th):
                stats["drop_duplicate"] += 1
                continue
            kept.append((score, chunk))
        return kept, stats


__all__ = ["EnrichScoringMixin"]
