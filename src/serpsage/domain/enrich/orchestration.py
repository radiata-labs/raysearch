from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import anyio
from anyio import to_thread

from serpsage.app.response import PageChunk, PageEnrichment, ResultItem
from serpsage.core.workunit import WorkUnit
from serpsage.domain.enrich.scoring import EnrichScoringMixin
from serpsage.text.chunking import chunk_segments, markdown_to_segments

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

        with self.span("enrich.one", url=url, domain=result.domain) as sp:
            stats: dict[str, int | float] = {}
            started = time.monotonic()
            try:
                timeout_s = self._resolve_page_timeout(
                    depth=depth,
                    step_deadline_ts=step_deadline_ts,
                    page_timeout_s=page_timeout_s,
                )
                if timeout_s <= 0:
                    return PageEnrichment(
                        chunks=[],
                        markdown="",
                        error="deadline exceeded",
                    )

                with anyio.fail_after(timeout_s):
                    t0 = time.monotonic()
                    fetch = await self._fetcher.afetch(
                        url=url,
                        timeout_s=timeout_s,
                        allow_render=True,
                        depth=depth,
                        rank_index=rank_index,
                    )
                    stats["fetch_ms"] = int((time.monotonic() - t0) * 1000)
                    sp.set_attr("fetch_mode", str(fetch.fetch_mode))
                    sp.set_attr("fetch_content_kind", str(fetch.content_kind))

                    t1 = time.monotonic()
                    extracted = await to_thread.run_sync(
                        lambda: self._extractor.extract(
                            url=url,
                            content=fetch.content,
                            content_type=fetch.content_type,
                        )
                    )
                    stats["extract_ms"] = int((time.monotonic() - t1) * 1000)

                    markdown = (extracted.markdown or "").strip()
                    plain_text = (extracted.plain_text or "").strip()
                    if not plain_text:
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            error="no content extracted",
                        )

                    ck = self.settings.enrich.chunking
                    t2 = time.monotonic()
                    segments = markdown_to_segments(
                        markdown or plain_text,
                        max_markdown_chars=int(ck.max_markdown_chars),
                        max_segments=int(ck.max_segments),
                        max_sentence_chars=int(ck.max_sentence_chars),
                    )
                    stats["segments"] = int(len(segments))

                    chunks = chunk_segments(
                        segments,
                        target_chars=int(ck.target_chars),
                        overlap_segments=int(ck.overlap_segments),
                        min_chunk_chars=int(ck.min_chunk_chars),
                    )
                    stats["chunk_ms"] = int((time.monotonic() - t2) * 1000)
                    stats["chunks_total"] = int(len(chunks))
                    if not chunks:
                        for k, v in stats.items():
                            sp.set_attr(k, v)
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            error="no chunks",
                        )
                    if len(chunks) > int(ck.max_chunks):
                        chunks = chunks[: int(ck.max_chunks)]

                    t3 = time.monotonic()
                    scored, score_stats = await self._scoring.score_chunks(
                        chunks=chunks,
                        query=query,
                        query_tokens=query_tokens,
                        intent_tokens=intent_tokens,
                        profile=profile,
                    )
                    stats["score_ms"] = int((time.monotonic() - t3) * 1000)
                    stats.update(score_stats)
                    if not scored:
                        for k, v in stats.items():
                            sp.set_attr(k, v)
                        return PageEnrichment(
                            chunks=[],
                            markdown=markdown,
                            content_kind=extracted.content_kind,
                            fetch_mode=fetch.fetch_mode,
                            error="no matching chunks",
                        )

                    top = scored[: int(top_k)]
                    stats["chunks_kept"] = int(len(top))
                    stats["markdown_chars"] = int(len(markdown))
                    stats["plain_text_chars"] = int(len(plain_text))
                    stats["total_ms"] = int((time.monotonic() - started) * 1000)
                    for k, v in stats.items():
                        sp.set_attr(k, v)

                    return PageEnrichment(
                        chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                        markdown=markdown,
                        content_kind=extracted.content_kind,
                        fetch_mode=fetch.fetch_mode,
                        error=None,
                    )
            except TimeoutError:
                sp.set_attr("timeout", True)
                return PageEnrichment(chunks=[], markdown="", error="timeout")
            except Exception as exc:  # noqa: BLE001
                sp.set_attr("error_type", type(exc).__name__)
                return PageEnrichment(chunks=[], markdown="", error=str(exc))

    def _resolve_page_timeout(
        self,
        *,
        depth: Literal["low", "medium", "high"],
        step_deadline_ts: float | None,
        page_timeout_s: float | None,
    ) -> float:
        latency_budgets = self.settings.enrich.latency_budgets or {}
        cfg = latency_budgets.get(depth) or latency_budgets.get("medium")
        depth_timeout = float(getattr(cfg, "page_timeout_s", 1.6) or 1.6)
        effective = (
            float(page_timeout_s) if page_timeout_s is not None else depth_timeout
        )
        if step_deadline_ts is not None:
            remain = float(step_deadline_ts - time.monotonic())
            effective = min(effective, remain)
        return max(0.0, effective)


__all__ = ["Enricher"]
