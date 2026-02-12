from __future__ import annotations

from typing import TYPE_CHECKING

from anyio import to_thread

from serpsage.app.response import PageChunk, PageEnrichment, ResultItem
from serpsage.core.workunit import WorkUnit
from serpsage.domain.enrich.scoring import EnrichScoringMixin
from serpsage.text.chunking import chunk_sentences, split_sentences

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
        profile: ProfileSettings,
        top_k: int,
    ) -> PageEnrichment:
        url = (result.url or "").strip()
        if not url:
            return PageEnrichment(chunks=[], error="empty url")

        with self.span("enrich.one", url=url, domain=result.domain) as sp:
            stats: dict[str, int | float] = {}
            try:
                fetch = await self._fetcher.afetch(url=url)
                extracted = await to_thread.run_sync(
                    lambda: self._extractor.extract(
                        url=url, content=fetch.content, content_type=fetch.content_type
                    )
                )
                blocks = list(extracted.blocks or [])
                stats["blocks_total"] = int(len(blocks))
                if not blocks:
                    sp.set_attr("blocks_total", 0)
                    return PageEnrichment(chunks=[], error="no blocks extracted")

                kept, block_stats = self._scoring.filter_blocks(
                    blocks, profile=profile, query=query
                )
                stats.update(block_stats)
                if not kept:
                    for k, v in stats.items():
                        sp.set_attr(k, v)
                    return PageEnrichment(chunks=[], error="no blocks after filtering")

                text_for_chunking = "\n\n".join(kept)
                sents = split_sentences(
                    text_for_chunking,
                    max_sentence_chars=int(
                        self.settings.enrich.chunking.max_sentence_chars
                    ),
                )
                stats["sentences"] = int(len(sents))
                if len(sents) > int(self.settings.enrich.chunking.max_sentences):
                    sents = sents[: int(self.settings.enrich.chunking.max_sentences)]

                chunks = chunk_sentences(
                    sents,
                    target_chars=int(self.settings.enrich.chunking.target_chars),
                    overlap_sentences=int(
                        self.settings.enrich.chunking.overlap_sentences
                    ),
                    min_chunk_chars=int(self.settings.enrich.chunking.min_chunk_chars),
                )
                stats["chunks_total"] = int(len(chunks))
                if not chunks:
                    for k, v in stats.items():
                        sp.set_attr(k, v)
                    return PageEnrichment(chunks=[], error="no chunks")
                if len(chunks) > int(self.settings.enrich.chunking.max_chunks):
                    chunks = chunks[: int(self.settings.enrich.chunking.max_chunks)]

                scored, score_stats = await self._scoring.score_chunks(
                    chunks=chunks,
                    query=query,
                    query_tokens=query_tokens,
                    profile=profile,
                )
                stats.update(score_stats)
                if not scored:
                    for k, v in stats.items():
                        sp.set_attr(k, v)
                    return PageEnrichment(chunks=[], error="no matching chunks")

                top = scored[: int(top_k)]
                stats["chunks_kept"] = int(len(top))
                for k, v in stats.items():
                    sp.set_attr(k, v)
                return PageEnrichment(
                    chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                    error=None,
                )
            except Exception as exc:  # noqa: BLE001
                sp.set_attr("error_type", type(exc).__name__)
                return PageEnrichment(chunks=[], error=str(exc))


__all__ = ["Enricher"]
