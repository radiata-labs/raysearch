from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal

import anyio

from search_core.config import ScoringConfig
from search_core.crawler import AsyncWebCrawler, WebCrawler
from search_core.models import PageChunk, PageEnrichment
from search_core.scorer import ScoringEngine
from search_core.text import TextUtils
from search_core.tools import compile_patterns, has_noise_word, is_duplicate_text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from search_core.config import (
        SearchContextConfig,
        WebChunkingConfig,
        WebDepthPreset,
        WebEnrichmentConfig,
    )
    from search_core.models import SearchResult

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"([\u3002\uFF01\uFF1F!?;\uFF1B.\n])")
_LONG_SENT_SPLIT_RE = re.compile(r"([,\uFF0C\u3001\t ])")


class _WebEnricherBase:
    def __init__(
        self,
        config: WebEnrichmentConfig,
        *,
        user_agent: str,
        scorer: ScoringEngine | None = None,
        min_score: float = 0.5,
    ) -> None:
        self.config = config
        self._min_score = float(min_score)
        self.scorer = scorer or ScoringEngine(ScoringConfig())
        self._user_agent = user_agent

    def _make_webcrawler(
        self,
        *,
        fetcher: Callable[[str], bytes | str] | None = None,
    ) -> WebCrawler:
        return WebCrawler(
            fetch_cfg=self.config.fetch,
            user_agent=self._user_agent,
            fetcher=fetcher,
        )

    def _make_async_webcrawler(
        self,
        *,
        afetcher: Callable[[str], Awaitable[bytes | str]] | None = None,
    ) -> AsyncWebCrawler:
        return AsyncWebCrawler(
            fetch_cfg=self.config.fetch,
            user_agent=self._user_agent,
            afetcher=afetcher,
        )

    def filter_blocks(
        self,
        blocks: list[str],
        *,
        context_config: SearchContextConfig,
        query_has_cjk: bool,
    ) -> list[str]:
        if not blocks:
            return []

        title_patterns = compile_patterns(context_config.title_tail_patterns)

        kept: list[str] = []
        for block in blocks:
            if has_noise_word(block, context_config):
                continue
            t = self.template_score(
                block,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="block",
            )
            if t >= float(self.config.select.block_hard_drop_threshold):
                continue
            kept.append(block)
            if len(kept) >= int(self.config.chunking.max_blocks):
                break
        return kept

    def split_sentences(
        self, text: str, *, max_sentence_chars: int | None = None
    ) -> list[str]:
        cleaned = TextUtils.clean_whitespace(text or "")
        if not cleaned:
            return []

        max_len = max_sentence_chars or self.config.chunking.max_sentence_chars
        parts = _SENTENCE_BOUNDARY_RE.split(cleaned)

        sentences: list[str] = []
        buf = ""
        for part in parts:
            if not part:
                continue
            buf += part
            if _SENTENCE_BOUNDARY_RE.fullmatch(part):
                s = buf.strip()
                if s:
                    sentences.extend(self._split_long_sentence(s, max_len))
                buf = ""
        tail = buf.strip()
        if tail:
            sentences.extend(self._split_long_sentence(tail, max_len))
        return [s for s in sentences if s]

    def _split_long_sentence(self, sentence: str, max_len: int) -> list[str]:
        if max_len <= 0 or len(sentence) <= max_len:
            return [sentence]

        parts = _LONG_SENT_SPLIT_RE.split(sentence)
        out: list[str] = []
        buf = ""
        for part in parts:
            if not part:
                continue
            if len(buf) + len(part) > max_len and buf.strip():
                out.append(buf.strip())
                buf = ""
            buf += part
        if buf.strip():
            out.append(buf.strip())
        return out or [sentence]

    def chunk_sentences(
        self,
        sentences: list[str],
        *,
        chunking: WebChunkingConfig,
    ) -> list[str]:
        if not sentences:
            return []

        target = max(1, int(chunking.target_chars))
        overlap = max(0, int(chunking.overlap_sentences))

        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0

        def flush() -> None:
            nonlocal cur, cur_len
            if not cur:
                return
            chunk = TextUtils.clean_whitespace(" ".join(cur))
            if len(chunk) >= chunking.min_chunk_chars:
                chunks.append(chunk)
            if overlap > 0:
                cur = cur[-overlap:]
                cur_len = sum(len(s) + 1 for s in cur)
            else:
                cur = []
                cur_len = 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if cur and cur_len + len(s) + 1 > target:
                flush()
            cur.append(s)
            cur_len += len(s) + 1
        flush()
        return chunks

    def score_chunks(
        self,
        chunks: list[str],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        domain: str | None,
        context_config: SearchContextConfig,
    ) -> list[tuple[float, str]]:
        _ = domain  # Domain bonus is intentionally not part of scoring.
        if not chunks:
            return []

        title_patterns = compile_patterns(context_config.title_tail_patterns)
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))

        filtered: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks):
            if self._hard_drop_chunk(
                chunk,
                context_config=context_config,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
            ):
                continue
            filtered.append((idx, chunk))

        if not filtered:
            return []

        base = self.scorer.score(
            [c for _, c in filtered],
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        base_scores = [float(s) for s, _ in base]
        return self._post_process_chunk_scores(
            filtered,
            base_scores=base_scores,
            early_bonus=float(self.config.select.early_bonus),
            dedupe_threshold=float(context_config.fuzzy_threshold),
        )

    def _post_process_chunk_scores(
        self,
        filtered_chunks: list[tuple[int, str]],
        *,
        base_scores: list[float],
        early_bonus: float,
        dedupe_threshold: float,
    ) -> list[tuple[float, str]]:
        scored: list[tuple[float, str]] = []
        for (pos, chunk), base_score in zip(filtered_chunks, base_scores, strict=False):
            final = float(base_score)
            if early_bonus > 1.0:
                final *= early_bonus ** (-pos)
            final = max(0.0, min(1.0, float(final)))
            scored.append((final, chunk))

        scored.sort(key=lambda t: t[0], reverse=True)

        kept: list[tuple[float, str]] = []
        th = float(dedupe_threshold)
        for score, chunk in scored:
            if float(score) <= 0.0:
                continue
            if float(score) < float(self._min_score):
                continue
            if is_duplicate_text(chunk, [c for _, c in kept], threshold=th):
                continue
            kept.append((score, chunk))

        return kept

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
            "アーカイブ",
            "サイトマップ",
            "プライバシー",
            "利用規約",
            "月を選択",
            "站点地图",
            "隐私",
            "条款",
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
            "上一页",
            "下一页",
            "一览",
            "目录",
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
            0.25 * len(re.findall(r"\d{4}年\d{1,2}月", text))
            + 0.15 * len(re.findall(r"\b(?:19|20)\d{2}\b", text)),
        )

        token_count, unique_ratio, _, digits_ratio, cjk_ratio = self._chunk_stats(text)
        uniq_component = 0.0
        if token_count >= 8 and unique_ratio < 0.45:
            uniq_component = min(1.0, (0.45 - unique_ratio) / 0.45)

        digit_component = 0.0
        if digits_ratio > 0.18:
            digit_component = min(1.0, (digits_ratio - 0.18) / 0.32)

        sep_component = 0.0
        sep_ratio = len(re.findall(r"[\|/>\u00bb]", text)) / max(1, len(text))
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

    def _hard_drop_chunk(
        self,
        chunk: str,
        *,
        context_config: SearchContextConfig,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
    ) -> bool:
        if not chunk:
            return True

        normalized = TextUtils.normalize_text(chunk)
        if not normalized:
            return True

        if has_noise_word(chunk, context_config):
            return True

        _, _, digits, digits_ratio, cjk_ratio = self._chunk_stats(chunk)
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
        return tpl >= float(self.config.select.template_hard_drop_threshold)

    def _chunk_stats(self, chunk: str) -> tuple[int, float, int, float, float]:
        tokens = TextUtils.tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", chunk))
        cjk_ratio = cjk_count / max(1, len(chunk))
        return token_count, unique_ratio, digits, digits_ratio, cjk_ratio


class WebEnricher(_WebEnricherBase):
    def __init__(
        self,
        config: WebEnrichmentConfig,
        *,
        user_agent: str,
        fetcher: Callable[[str], bytes | str] | None = None,
        scorer: ScoringEngine | None = None,
        min_score: float = 0.5,
    ) -> None:
        super().__init__(
            config,
            user_agent=user_agent,
            scorer=scorer,
            min_score=min_score,
        )
        self.crawler = self._make_webcrawler(fetcher=fetcher)

    def enrich_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        context_config: SearchContextConfig,
        preset: WebDepthPreset,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if not results:
            return
        if not query_tokens:
            return

        n = len(results)
        target = int(math.ceil(n * float(preset.pages_ratio)))
        m = max(int(preset.min_pages), min(int(preset.max_pages), target))
        m = min(m, n)
        if m <= 0:
            return

        chunking = self.config.chunking
        if chunk_target_chars is not None:
            chunking = chunking.with_overrides(target_chars=int(chunk_target_chars))
        if chunk_overlap_sentences is not None:
            chunking = chunking.with_overrides(
                overlap_sentences=int(chunk_overlap_sentences)
            )
        if min_chunk_chars is not None:
            chunking = chunking.with_overrides(min_chunk_chars=int(min_chunk_chars))

        top_k = int(preset.top_chunks_per_page)
        max_workers = min(int(self.config.fetch.max_workers), m)

        def crawl_one(  # noqa: PLR0911
            item: SearchResult,
        ) -> tuple[SearchResult, list[str], list[float], str | None]:
            url = (item.url or "").strip()
            if not url:
                return (item, [], [], "empty url")

            try:
                crawl = self.crawler.fetch_blocks(url)
                if crawl.error:
                    return (item, [], [], crawl.error)
                blocks = crawl.blocks

                if not blocks:
                    return (item, [], [], "no blocks extracted")

                query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))
                kept_blocks = self.filter_blocks(
                    blocks,
                    context_config=context_config,
                    query_has_cjk=query_has_cjk,
                )
                if not kept_blocks:
                    return (item, [], [], "no blocks after filtering")

                text_for_chunking = "\n\n".join(kept_blocks)

                sents = self.split_sentences(text_for_chunking)
                if len(sents) > int(chunking.max_sentences):
                    sents = sents[: int(chunking.max_sentences)]
                chunks = self.chunk_sentences(sents, chunking=chunking)
                if not chunks:
                    return (item, [], [], "no chunks")
                if len(chunks) > int(chunking.max_chunks):
                    chunks = chunks[: int(chunking.max_chunks)]

                scored = self.score_chunks(
                    chunks,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    domain=item.domain,
                    context_config=context_config,
                )
                if not scored:
                    return (item, [], [], "no matching chunks")

                top: list[tuple[float, str]] = []
                for score, chunk in scored:
                    top.append((score, chunk))
                    if len(top) >= top_k:
                        break
                return (item, [c for _, c in top], [float(s) for s, _ in top], None)
            except Exception as exc:  # noqa: BLE001
                return (item, [], [], str(exc))

        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for item in results[:m]:
                futures[ex.submit(crawl_one, item)] = item
            for fut in as_completed(futures):
                item, chunks, scores, err = fut.result()
                item.page = PageEnrichment(
                    chunks=[
                        PageChunk(text=chunks[i], score=float(scores[i]))
                        for i in range(min(len(chunks), len(scores)))
                    ],
                    error=err,
                )


class AsyncWebEnricher(_WebEnricherBase):
    def __init__(
        self,
        config: WebEnrichmentConfig,
        *,
        user_agent: str,
        afetcher: Callable[[str], Awaitable[bytes | str]] | None = None,
        scorer: ScoringEngine | None = None,
        min_score: float = 0.5,
    ) -> None:
        super().__init__(
            config,
            user_agent=user_agent,
            scorer=scorer,
            min_score=min_score,
        )
        self.crawler = self._make_async_webcrawler(afetcher=afetcher)

    async def aenrich_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        context_config: SearchContextConfig,
        preset: WebDepthPreset,
        chunk_target_chars: int | None = None,
        chunk_overlap_sentences: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        if not results:
            return
        if not query_tokens:
            return

        n = len(results)
        target = int(math.ceil(n * float(preset.pages_ratio)))
        m = max(int(preset.min_pages), min(int(preset.max_pages), target))
        m = min(m, n)
        if m <= 0:
            return

        chunking = self.config.chunking
        if chunk_target_chars is not None:
            chunking = chunking.with_overrides(target_chars=int(chunk_target_chars))
        if chunk_overlap_sentences is not None:
            chunking = chunking.with_overrides(
                overlap_sentences=int(chunk_overlap_sentences)
            )
        if min_chunk_chars is not None:
            chunking = chunking.with_overrides(min_chunk_chars=int(min_chunk_chars))

        top_k = int(preset.top_chunks_per_page)
        max_workers = min(int(self.config.fetch.max_workers), m)

        sem = anyio.Semaphore(max_workers)
        cancelled_exc = anyio.get_cancelled_exc_class()

        async def crawl_one(item: SearchResult) -> None:  # noqa: PLR0915
            await sem.acquire()
            try:
                url = (item.url or "").strip()
                if not url:
                    item.page = PageEnrichment(chunks=[], error="empty url")
                    return

                crawl = await self.crawler.afetch_blocks(url)
                if crawl.error:
                    item.page = PageEnrichment(chunks=[], error=crawl.error)
                    return
                blocks = crawl.blocks
                if not blocks:
                    item.page = PageEnrichment(chunks=[], error="no blocks extracted")
                    return

                query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))
                kept_blocks = self.filter_blocks(
                    blocks,
                    context_config=context_config,
                    query_has_cjk=query_has_cjk,
                )
                if not kept_blocks:
                    item.page = PageEnrichment(
                        chunks=[],
                        error="no blocks after filtering",
                    )
                    return

                text_for_chunking = "\n\n".join(kept_blocks)
                sents = self.split_sentences(text_for_chunking)
                if len(sents) > int(chunking.max_sentences):
                    sents = sents[: int(chunking.max_sentences)]
                chunks = self.chunk_sentences(sents, chunking=chunking)
                if not chunks:
                    item.page = PageEnrichment(chunks=[], error="no chunks")
                    return
                if len(chunks) > int(chunking.max_chunks):
                    chunks = chunks[: int(chunking.max_chunks)]

                scored = self.score_chunks(
                    chunks,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    domain=item.domain,
                    context_config=context_config,
                )
                if not scored:
                    item.page = PageEnrichment(chunks=[], error="no matching chunks")
                    return

                top: list[tuple[float, str]] = []
                for score, chunk in scored:
                    top.append((float(score), chunk))
                    if len(top) >= top_k:
                        break

                item.page = PageEnrichment(
                    chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                    error=None,
                )
            except cancelled_exc:
                raise
            except Exception as exc:  # noqa: BLE001
                item.page = PageEnrichment(chunks=[], error=str(exc))
            finally:
                sem.release()

        async with anyio.create_task_group() as tg:
            for item in results[:m]:
                tg.start_soon(crawl_one, item)

    async def aclose(self) -> None:
        await self.crawler.aclose()


__all__ = ["WebEnricher", "AsyncWebEnricher"]
