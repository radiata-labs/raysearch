from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.response import PageChunk
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.chunking import (
    chunk_segments,
    markdown_to_segments,
    prefilter_segments_by_tokens,
)
from serpsage.text.normalize import normalize_text
from serpsage.text.similarity import is_duplicate_text

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ProfileSettings


class FetchRankStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_chunk_rank"

    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        chunks_request = ctx.chunks_request
        span.set_attr("has_chunks", bool(chunks_request is not None))
        if chunks_request is None:
            return ctx
        query = chunks_request.query

        if ctx.extracted is None:
            ctx.page.error = ctx.page.error or "missing extracted content"
            return ctx
        if not (ctx.extracted.plain_text or "").strip():
            ctx.page.error = ctx.page.error or "no content extracted"
            return ctx

        top_k = int(
            chunks_request.top_k_chunks
            if chunks_request.top_k_chunks is not None
            else self.settings.fetch.chunk.default_top_k
        )
        chunks, timing_ms, error = await self._chunk_and_score(
            markdown=ctx.extracted.markdown,
            plain_text=ctx.extracted.plain_text,
            query=query,
            query_tokens=ctx.chunk_query_tokens or [],
            intent_tokens=ctx.chunk_intent_tokens or [],
            profile=ctx.profile,
            top_k=top_k,
        )
        max_chars = chunks_request.max_chars
        if max_chars is not None and max_chars > 0:
            for chunk in chunks:
                if len(chunk.text) > max_chars:
                    chunk.text = chunk.text[:max_chars].rstrip() + "..."
        ctx.chunks = chunks
        ctx.page.chunks = chunks
        ctx.page.timing_ms.update(timing_ms)
        if error:
            ctx.page.error = error
        span.set_attr("top_k_chunks", int(top_k))
        span.set_attr("chunks_kept", int(len(chunks)))
        return ctx

    async def _chunk_and_score(
        self,
        *,
        markdown: str,
        plain_text: str,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        profile: ProfileSettings | None,
        top_k: int | None = None,
    ) -> tuple[list[PageChunk], dict[str, int], str | None]:
        chunk_cfg = self.settings.fetch.chunk
        profile = profile or self.settings.get_profile(
            self.settings.search.default_profile
        )

        t0 = time.monotonic()
        segments = markdown_to_segments(
            markdown or plain_text,
            max_markdown_chars=int(chunk_cfg.max_markdown_chars),
            max_segments=int(chunk_cfg.max_segments),
            max_sentence_chars=int(chunk_cfg.max_sentence_chars),
        )
        filtered_segments = prefilter_segments_by_tokens(
            segments,
            query_tokens=query_tokens or [],
            min_hits=int(chunk_cfg.min_query_token_hits),
            max_segments=int(chunk_cfg.query_prefilter_window),
        )
        chunks = chunk_segments(
            filtered_segments,
            target_chars=int(chunk_cfg.target_chars),
            overlap_segments=int(chunk_cfg.overlap_segments),
            min_chunk_chars=int(chunk_cfg.min_chunk_chars),
        )
        if len(chunks) > int(chunk_cfg.max_chunks):
            chunks = chunks[: int(chunk_cfg.max_chunks)]
        chunk_ms = int((time.monotonic() - t0) * 1000)
        if not chunks:
            return [], {"chunk_ms": chunk_ms, "score_ms": 0}, "no chunks"

        kept_chunks = self._filter_noise_chunks(
            chunks=chunks,
            query_tokens=query_tokens,
            profile=profile,
            min_query_token_hits=int(chunk_cfg.min_query_token_hits),
        )
        if not kept_chunks:
            return [], {"chunk_ms": chunk_ms, "score_ms": 0}, "no matching chunks"

        t1 = time.monotonic()
        scores = await self._ranker.score_texts(
            texts=kept_chunks,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        score_ms = int((time.monotonic() - t1) * 1000)

        scored = self._post_process_chunk_scores(
            chunks=kept_chunks,
            scores=scores,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
            min_score=float(chunk_cfg.min_chunk_score),
            dedupe_threshold=float(profile.fuzzy_threshold),
            early_bonus=float(chunk_cfg.early_bonus),
        )
        if not scored:
            return (
                [],
                {"chunk_ms": chunk_ms, "score_ms": score_ms},
                "no matching chunks",
            )

        limit = int(top_k or int(chunk_cfg.default_top_k))
        top = scored[: max(1, limit)]
        page_chunks = [PageChunk(text=txt, score=float(score)) for score, txt in top]
        return page_chunks, {"chunk_ms": chunk_ms, "score_ms": score_ms}, None

    def _filter_noise_chunks(
        self,
        *,
        chunks: list[str],
        query_tokens: list[str],
        profile: ProfileSettings,
        min_query_token_hits: int,
    ) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            lowered = normalize_text(chunk)
            if not lowered:
                continue
            if any(
                w and normalize_text(w) in lowered for w in (profile.noise_words or [])
            ):
                continue
            if not query_tokens:
                out.append(chunk)
                continue
            hits = 0
            for tok in query_tokens:
                if tok and tok.lower() in lowered:
                    hits += 1
            if hits >= max(1, int(min_query_token_hits)):
                out.append(chunk)
        return out

    def _post_process_chunk_scores(
        self,
        *,
        chunks: list[str],
        scores: list[float],
        query_tokens: list[str],
        intent_tokens: list[str],
        min_score: float,
        dedupe_threshold: float,
        early_bonus: float,
    ) -> list[tuple[float, str]]:
        scored: list[tuple[float, str]] = []
        for idx, chunk in enumerate(chunks):
            base = float(scores[idx]) if idx < len(scores) else 0.0
            loc = self._get_hit_location(
                chunk=chunk,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
            )
            bonus = 1.0 + max(0.0, float(early_bonus) - 1.0) * ((1.0 - loc) ** 2)
            score = base * bonus
            if score >= float(min_score):
                scored.append((float(score), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)

        kept: list[tuple[float, str]] = []
        for score, chunk in scored:
            if is_duplicate_text(
                chunk,
                [c for _, c in kept],
                threshold=float(dedupe_threshold),
            ):
                continue
            kept.append((score, chunk))
        return kept

    def _get_hit_location(
        self,
        *,
        chunk: str,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> float:
        lowered = normalize_text(chunk)
        if not lowered:
            return 1.0
        poses: list[int] = []
        for token in query_tokens or []:
            pos = lowered.find((token or "").lower())
            if pos >= 0:
                poses.append(pos)
        for token in intent_tokens or []:
            pos = lowered.find((token or "").lower())
            if pos >= 0:
                poses.append(pos)
        if not poses:
            return 1.0
        return float(min(poses)) / float(max(1, len(lowered)))


__all__ = ["FetchRankStep"]
