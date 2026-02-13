from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import anyio
from anyio import to_thread

from serpsage.app.response import PageChunk, PageEnrichment, ResultItem
from serpsage.core.tuning import (
    chunk_profile_for_depth,
    deadline_profile_for_depth,
    normalize_depth,
)
from serpsage.core.workunit import WorkUnit
from serpsage.domain.enrich.scoring import EnrichScoringMixin
from serpsage.text.chunking import (
    chunk_segments,
    markdown_to_segments,
    prefilter_segments_by_tokens,
)

if TYPE_CHECKING:
    from serpsage.contracts.services import ExtractorBase, FetcherBase, RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ProfileSettings


class Enricher(WorkUnit):
    def __init__(
        self,
        *,
        rt: Runtime,
        fetcher: FetcherBase,
        extractor: ExtractorBase,
        ranker: RankerBase,
    ) -> None:
        super().__init__(rt=rt)
        self._fetcher = fetcher
        self._extractor = extractor
        self._scoring = EnrichScoringMixin(settings=self.settings, ranker=ranker)
        self.bind_deps(fetcher, extractor, ranker)

    async def enrich_one(
        self,
        *,
        result: ResultItem,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        profile: ProfileSettings,
        top_k: int,
        depth: Literal["low", "medium", "high"] = "medium",
        rank_index: int = 0,
        step_deadline_ts: float | None = None,
        page_timeout_s: float | None = None,
    ) -> PageEnrichment:
        url = (result.url or "").strip()
        if not url:
            return PageEnrichment(chunks=[], markdown="", error="empty url")

        depth_key = normalize_depth(depth)
        chunk_cfg = chunk_profile_for_depth(depth_key)
        with self.span("enrich.one", url=url, domain=result.domain) as sp:
            timing_ms: dict[str, int] = {}
            warnings: list[str] = []
            started = time.monotonic()
            try:
                timeout_s = self._resolve_page_timeout(
                    depth=depth_key,
                    step_deadline_ts=step_deadline_ts,
                    page_timeout_s=page_timeout_s,
                )
                if timeout_s <= 0:
                    return PageEnrichment(
                        chunks=[],
                        markdown="",
                        timing_ms={"total_ms": 0},
                        warnings=["deadline exceeded before start"],
                        error="deadline exceeded",
                    )

                with anyio.fail_after(timeout_s):
                    t0 = time.monotonic()
                    fetch = await self._fetcher.afetch(
                        url=url,
                        timeout_s=timeout_s,
                        allow_render=True,
                        depth=depth_key,
                        rank_index=rank_index,
                    )
                    timing_ms["fetch_ms"] = int((time.monotonic() - t0) * 1000)
                    sp.set_attr("fetch_mode", str(fetch.fetch_mode))
                    sp.set_attr("fetch_content_kind", str(fetch.content_kind))
                    if fetch.attempt_chain:
                        warnings.append(
                            f"attempt_chain:{'->'.join(fetch.attempt_chain)}"
                        )

                    t1 = time.monotonic()
                    extracted = await to_thread.run_sync(
                        lambda: self._extractor.extract(
                            url=url,
                            content=fetch.content,
                            content_type=fetch.content_type,
                        )
                    )
                    timing_ms["extract_ms"] = int((time.monotonic() - t1) * 1000)

                    markdown = (extracted.markdown or "").strip()
                    plain_text = (extracted.plain_text or "").strip()
                    warnings.extend(extracted.warnings or [])
                    if not plain_text:
                        timing_ms["total_ms"] = int((time.monotonic() - started) * 1000)
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            timing_ms=timing_ms,
                            warnings=warnings,
                            error="no content extracted",
                        )

                    t2 = time.monotonic()
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
                    timing_ms["chunk_ms"] = int((time.monotonic() - t2) * 1000)
                    sp.set_attr("segments", int(len(segments)))
                    sp.set_attr("segments_filtered", int(len(filtered_segments)))
                    sp.set_attr("chunks_total", int(len(chunks)))

                    if not chunks:
                        timing_ms["total_ms"] = int((time.monotonic() - started) * 1000)
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            timing_ms=timing_ms,
                            warnings=warnings,
                            error="no chunks",
                        )

                    t3 = time.monotonic()
                    scored, score_stats = await self._scoring.score_chunks(
                        chunks=chunks,
                        query=query,
                        query_tokens=query_tokens,
                        intent_tokens=intent_tokens,
                        profile=profile,
                        depth=depth_key,
                    )
                    timing_ms["score_ms"] = int((time.monotonic() - t3) * 1000)
                    for key, val in score_stats.items():
                        sp.set_attr(key, val)
                    if not scored:
                        timing_ms["total_ms"] = int((time.monotonic() - started) * 1000)
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            timing_ms=timing_ms,
                            warnings=warnings,
                            error="no matching chunks",
                        )

                    top = scored[: int(top_k)]
                    timing_ms["total_ms"] = int((time.monotonic() - started) * 1000)
                    sp.set_attr("chunks_kept", int(len(top)))
                    sp.set_attr("markdown_chars", int(len(markdown)))
                    sp.set_attr("plain_text_chars", int(len(plain_text)))
                    sp.set_attr("extractor_used", str(extracted.extractor_used))
                    sp.set_attr("extract_quality", float(extracted.quality_score))
                    for key, val in timing_ms.items():
                        sp.set_attr(key, int(val))

                    return PageEnrichment(
                        chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                        markdown=markdown,
                        content_kind=extracted.content_kind,
                        fetch_mode=fetch.fetch_mode,
                        timing_ms=timing_ms,
                        warnings=warnings,
                        error=None,
                    )
            except TimeoutError:
                sp.set_attr("timeout", True)
                return PageEnrichment(
                    chunks=[],
                    markdown="",
                    timing_ms={"total_ms": int((time.monotonic() - started) * 1000)},
                    warnings=warnings,
                    error="timeout",
                )
            except Exception as exc:  # noqa: BLE001
                sp.set_attr("error_type", type(exc).__name__)
                return PageEnrichment(
                    chunks=[],
                    markdown="",
                    timing_ms={"total_ms": int((time.monotonic() - started) * 1000)},
                    warnings=warnings,
                    error=str(exc),
                )

    def _resolve_page_timeout(
        self,
        *,
        depth: Literal["low", "medium", "high"],
        step_deadline_ts: float | None,
        page_timeout_s: float | None,
    ) -> float:
        cfg = deadline_profile_for_depth(depth)
        effective = (
            float(page_timeout_s)
            if page_timeout_s is not None
            else float(cfg.page_timeout_s)
        )
        if step_deadline_ts is not None:
            remain = float(step_deadline_ts - time.monotonic())
            effective = min(effective, remain)
        return max(0.0, effective)


__all__ = ["Enricher"]
