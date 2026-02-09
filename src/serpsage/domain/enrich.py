from __future__ import annotations

import re
from typing import TYPE_CHECKING

from anyio import to_thread

from serpsage.app.response import PageChunk, PageEnrichment, ResultItem
from serpsage.contracts.base import WorkUnit
from serpsage.text.chunking import chunk_sentences, split_sentences
from serpsage.text.normalize import normalize_text
from serpsage.text.similarity import is_duplicate_text
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens

if TYPE_CHECKING:
    from serpsage.contracts.protocols import Extractor, Fetcher, Ranker
    from serpsage.settings.models import ProfileSettings


class Enricher(WorkUnit):
    def __init__(
        self, *, rt, fetcher: Fetcher, extractor: Extractor, ranker: Ranker
    ) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._fetcher = fetcher
        self._extractor = extractor
        self._ranker = ranker

    async def enrich_one(  # noqa: PLR0911
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

        try:
            fetch = await self._fetcher.afetch(url=url)
            extracted = await to_thread.run_sync(
                lambda: self._extractor.extract(
                    url=url, content=fetch.content, content_type=fetch.content_type
                )
            )
            blocks = list(extracted.blocks or [])
            if not blocks:
                return PageEnrichment(chunks=[], error="no blocks extracted")

            kept = self._filter_blocks(blocks, profile=profile, query=query)
            if not kept:
                return PageEnrichment(chunks=[], error="no blocks after filtering")

            text_for_chunking = "\n\n".join(kept)
            sents = split_sentences(
                text_for_chunking,
                max_sentence_chars=int(
                    self.settings.enrich.chunking.max_sentence_chars
                ),
            )
            if len(sents) > int(self.settings.enrich.chunking.max_sentences):
                sents = sents[: int(self.settings.enrich.chunking.max_sentences)]

            chunks = chunk_sentences(
                sents,
                target_chars=int(self.settings.enrich.chunking.target_chars),
                overlap_sentences=int(self.settings.enrich.chunking.overlap_sentences),
                min_chunk_chars=int(self.settings.enrich.chunking.min_chunk_chars),
            )
            if not chunks:
                return PageEnrichment(chunks=[], error="no chunks")
            if len(chunks) > int(self.settings.enrich.chunking.max_chunks):
                chunks = chunks[: int(self.settings.enrich.chunking.max_chunks)]

            scored = self._score_chunks(
                chunks=chunks,
                query=query,
                query_tokens=query_tokens,
                profile=profile,
            )
            if not scored:
                return PageEnrichment(chunks=[], error="no matching chunks")

            top = scored[: int(top_k)]
            return PageEnrichment(
                chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            return PageEnrichment(chunks=[], error=str(exc))

    def _filter_blocks(
        self, blocks: list[str], *, profile: ProfileSettings, query: str
    ) -> list[str]:
        noise_words = list(profile.noise_words or [])
        title_patterns = self._compile_patterns(list(profile.title_tail_patterns or []))
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))

        kept: list[str] = []
        for block in blocks:
            if self._has_noise_word(block, noise_words):
                continue
            tpl = self._template_score(
                block,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="block",
            )
            if tpl >= float(self.settings.enrich.select.block_hard_drop_threshold):
                continue
            kept.append(block)
            if len(kept) >= int(self.settings.enrich.chunking.max_blocks):
                break
        return kept

    def _score_chunks(
        self,
        *,
        chunks: list[str],
        query: str,
        query_tokens: list[str],
        profile: ProfileSettings,
    ) -> list[tuple[float, str]]:
        title_patterns = self._compile_patterns(list(profile.title_tail_patterns or []))
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))
        intent_tokens = extract_intent_tokens(query, list(profile.intent_terms or []))

        filtered: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks):
            if self._hard_drop_chunk(
                chunk,
                profile=profile,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
            ):
                continue
            filtered.append((idx, chunk))
        if not filtered:
            return []

        raw_scores = self._ranker.score_texts(
            texts=[c for _, c in filtered],
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        norm = self._ranker.normalize(scores=raw_scores)
        if norm and max(norm) <= 0.0 and max(raw_scores) > 0.0:
            norm = [0.5 for _ in norm]

        return self._post_process_chunk_scores(
            filtered,
            base_scores=norm,
            early_bonus=float(self.settings.enrich.select.early_bonus),
            dedupe_threshold=float(profile.fuzzy_threshold),
            min_score=float(self.settings.pipeline.min_score),
        )

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern[str]]:
        out: list[re.Pattern[str]] = []
        for p in patterns:
            if not p:
                continue
            try:
                out.append(re.compile(p, re.IGNORECASE))
            except re.error:
                continue
        return out

    def _has_noise_word(self, text: str, noise_words: list[str]) -> bool:
        lowered = normalize_text(text)
        for w in noise_words or []:
            wl = normalize_text(w)
            if wl and wl in lowered:
                return True
        return False

    def _chunk_stats(self, chunk: str) -> tuple[int, float, int, float, float]:
        tokens = tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", chunk))
        cjk_ratio = cjk_count / max(1, len(chunk))
        return token_count, unique_ratio, digits, digits_ratio, cjk_ratio

    def _template_score(
        self,
        text: str,
        *,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
        mode: str,
    ) -> float:
        if not text:
            return 1.0
        lowered = normalize_text(text)
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
        }

        kw_hits = 0.0
        for kw in kw_strong:
            if kw in lowered:
                kw_hits += 0.35
        for kw in kw_weak:
            if kw in lowered:
                kw_hits += 0.15
        kw_component = min(1.0, kw_hits)

        date_component = min(1.0, 0.15 * len(re.findall(r"\b(?:19|20)\d{2}\b", text)))

        token_count, unique_ratio, _, digits_ratio, cjk_ratio = self._chunk_stats(text)
        uniq_component = 0.0
        if token_count >= 8 and unique_ratio < 0.45:
            uniq_component = min(1.0, (0.45 - unique_ratio) / 0.45)

        digit_component = 0.0
        if digits_ratio > 0.18:
            digit_component = min(1.0, (digits_ratio - 0.18) / 0.32)

        sep_ratio = len(re.findall(r"[\|/>\u00bb]", text)) / max(1, len(text))
        sep_component = 0.0
        if mode == "block" and sep_ratio > 0.03:
            sep_component = min(1.0, (sep_ratio - 0.03) / 0.10)
        elif mode != "block" and sep_ratio > 0.05:
            sep_component = min(1.0, (sep_ratio - 0.05) / 0.10)

        lang_component = 0.0
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.10:
            lang_component = 0.35

        title_component = 0.0
        for pat in title_patterns:
            if pat.search(text):
                title_component = 0.6
                break

        tpl = 0.0
        for comp in (
            kw_component,
            date_component,
            digit_component,
            sep_component,
            uniq_component,
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
        profile: ProfileSettings,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
    ) -> bool:
        if not chunk:
            return True
        normalized = normalize_text(chunk)
        if not normalized:
            return True
        if self._has_noise_word(chunk, list(profile.noise_words or [])):
            return True

        _, _, digits, digits_ratio, cjk_ratio = self._chunk_stats(chunk)
        if digits >= 18 and digits_ratio > 0.2:
            return True
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.06:
            return True

        tpl = self._template_score(
            chunk,
            query_has_cjk=query_has_cjk,
            title_patterns=title_patterns,
            mode="chunk",
        )
        return tpl >= float(self.settings.enrich.select.template_hard_drop_threshold)

    def _post_process_chunk_scores(
        self,
        filtered_chunks: list[tuple[int, str]],
        *,
        base_scores: list[float],
        early_bonus: float,
        dedupe_threshold: float,
        min_score: float,
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
            if float(score) < float(min_score):
                continue
            if is_duplicate_text(chunk, [c for _, c in kept], threshold=th):
                continue
            kept.append((score, chunk))
        return kept


__all__ = ["Enricher"]
