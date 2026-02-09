from __future__ import annotations

import math
import re
from typing import Any

import anyio

from serpsage.app.container import Container
from serpsage.app.response import PageChunk, PageEnrichment, ResultItem
from serpsage.pipeline.steps import StepContext
from serpsage.text.chunking import chunk_sentences, split_sentences
from serpsage.text.normalize import clean_whitespace, normalize_text
from serpsage.text.similarity import is_duplicate_text
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens


class EnrichStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.enrich")
        try:
            depth = ctx.request.depth
            if depth == "simple":
                return ctx
            if not ctx.settings.enrich.enabled:
                return ctx
            if not ctx.results:
                return ctx

            preset = ctx.settings.enrich.depth_presets.get(depth)  # type: ignore[index]
            if preset is None:
                return ctx

            n = len(ctx.results)
            target = int(math.ceil(n * float(preset.pages_ratio)))
            m = max(int(preset.min_pages), min(int(preset.max_pages), target))
            m = min(m, n)
            if m <= 0:
                return ctx

            # Only enrich the current top results.
            work = ctx.results[:m]
            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or ctx.settings.get_profile(ctx.settings.pipeline.default_profile)
            intent_tokens = extract_intent_tokens(query, list(getattr(profile, "intent_terms", []) or []))
            top_k = int(preset.top_chunks_per_page)

            async with anyio.create_task_group() as tg:
                for r in work:
                    tg.start_soon(
                        self._enrich_one,
                        r,
                        query,
                        query_tokens,
                        intent_tokens,
                        profile,
                        top_k,
                    )
            return ctx
        finally:
            span.end()

    async def _enrich_one(
        self,
        r: ResultItem,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        profile: Any,
        top_k: int,
    ) -> None:
        url = (r.url or "").strip()
        if not url:
            r.page = PageEnrichment(chunks=[], error="empty url")
            return

        try:
            fetch = await self._c.fetcher.afetch(url=url)
            extracted = await anyio.to_thread.run_sync(
                lambda: self._c.extractor.extract(
                    url=url, content=fetch.content, content_type=fetch.content_type
                )
            )
            blocks = list(extracted.blocks or [])
            if not blocks:
                r.page = PageEnrichment(chunks=[], error="no blocks extracted")
                return

            kept = _filter_blocks(
                blocks,
                profile=profile,
                max_blocks=int(self._c.settings.enrich.chunking.max_blocks),
                block_hard_drop_threshold=float(self._c.settings.enrich.select.block_hard_drop_threshold),
                query=query,
            )
            if not kept:
                r.page = PageEnrichment(chunks=[], error="no blocks after filtering")
                return

            text_for_chunking = "\n\n".join(kept)
            sents = split_sentences(
                text_for_chunking,
                max_sentence_chars=int(self._c.settings.enrich.chunking.max_sentence_chars),
            )
            if len(sents) > int(self._c.settings.enrich.chunking.max_sentences):
                sents = sents[: int(self._c.settings.enrich.chunking.max_sentences)]

            chunks = chunk_sentences(
                sents,
                target_chars=int(self._c.settings.enrich.chunking.target_chars),
                overlap_sentences=int(self._c.settings.enrich.chunking.overlap_sentences),
                min_chunk_chars=int(self._c.settings.enrich.chunking.min_chunk_chars),
            )
            if not chunks:
                r.page = PageEnrichment(chunks=[], error="no chunks")
                return
            if len(chunks) > int(self._c.settings.enrich.chunking.max_chunks):
                chunks = chunks[: int(self._c.settings.enrich.chunking.max_chunks)]

            title_patterns = _compile_patterns(getattr(profile, "title_tail_patterns", []) or [])
            query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))

            filtered: list[tuple[int, str]] = []
            for idx, chunk in enumerate(chunks):
                if _hard_drop_chunk(
                    chunk,
                    profile=profile,
                    query_has_cjk=query_has_cjk,
                    title_patterns=title_patterns,
                    tpl_threshold=float(self._c.settings.enrich.select.template_hard_drop_threshold),
                ):
                    continue
                filtered.append((idx, chunk))
            if not filtered:
                r.page = PageEnrichment(chunks=[], error="no matching chunks")
                return

            raw_scores = self._c.ranker.score_texts(
                texts=[c for _, c in filtered],
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
            )
            norm = self._c.ranker.normalize(scores=raw_scores)
            if norm and max(norm) <= 0.0 and max(raw_scores) > 0.0:
                norm = [0.5 for _ in norm]

            scored = _post_process_chunk_scores(
                filtered,
                base_scores=norm,
                early_bonus=float(self._c.settings.enrich.select.early_bonus),
                dedupe_threshold=float(getattr(profile, "fuzzy_threshold", 0.88)),
                min_score=float(self._c.settings.pipeline.min_score),
            )
            if not scored:
                r.page = PageEnrichment(chunks=[], error="no matching chunks")
                return

            top = scored[: int(top_k)]
            r.page = PageEnrichment(
                chunks=[PageChunk(text=c, score=float(s)) for s, c in top],
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
            r.page = PageEnrichment(chunks=[], error=str(exc))


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    out: list[re.Pattern[str]] = []
    for p in patterns:
        if not p:
            continue
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return out


def _has_noise_word(text: str, noise_words: list[str]) -> bool:
    lowered = normalize_text(text)
    for w in noise_words or []:
        wl = normalize_text(w)
        if wl and wl in lowered:
            return True
    return False


def _filter_blocks(
    blocks: list[str],
    *,
    profile: Any,
    max_blocks: int,
    block_hard_drop_threshold: float,
    query: str,
) -> list[str]:
    if not blocks:
        return []
    noise_words = list(getattr(profile, "noise_words", []) or [])
    title_patterns = _compile_patterns(list(getattr(profile, "title_tail_patterns", []) or []))
    query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))

    kept: list[str] = []
    for block in blocks:
        if _has_noise_word(block, noise_words):
            continue
        tpl = _template_score(block, query_has_cjk=query_has_cjk, title_patterns=title_patterns, mode="block")
        if tpl >= float(block_hard_drop_threshold):
            continue
        kept.append(block)
        if len(kept) >= int(max_blocks):
            break
    return kept


def _chunk_stats(chunk: str) -> tuple[int, float, int, float, float]:
    tokens = tokenize(chunk)
    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
    digits = sum(ch.isdigit() for ch in chunk)
    digits_ratio = digits / max(1, len(chunk))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", chunk))
    cjk_ratio = cjk_count / max(1, len(chunk))
    return token_count, unique_ratio, digits, digits_ratio, cjk_ratio


def _template_score(
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

    # A small, non-garbled set. Users can extend via profile noise_words/title_tail_patterns.
    kw_strong = {"archive", "archives", "sitemap", "privacy", "terms", "cookie", "cookies"}
    kw_weak = {"next", "previous", "older", "newer", "page", "pages", "category", "categories", "tag", "tags"}

    kw_hits = 0.0
    for kw in kw_strong:
        if kw in lowered:
            kw_hits += 0.35
    for kw in kw_weak:
        if kw in lowered:
            kw_hits += 0.15
    kw_component = min(1.0, kw_hits)

    date_component = min(1.0, 0.15 * len(re.findall(r"\b(?:19|20)\d{2}\b", text)))

    token_count, unique_ratio, _, digits_ratio, cjk_ratio = _chunk_stats(text)
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
    for comp in (kw_component, date_component, digit_component, sep_component, uniq_component, lang_component, title_component):
        comp = max(0.0, min(1.0, float(comp)))
        tpl = 1.0 - (1.0 - tpl) * (1.0 - comp)

    if mode == "block" and len(lowered) <= 64 and kw_component >= 0.35:
        tpl = max(tpl, 0.9)

    return float(max(0.0, min(1.0, tpl)))


def _hard_drop_chunk(
    chunk: str,
    *,
    profile: Any,
    query_has_cjk: bool,
    title_patterns: list[re.Pattern[str]],
    tpl_threshold: float,
) -> bool:
    if not chunk:
        return True
    normalized = normalize_text(chunk)
    if not normalized:
        return True
    if _has_noise_word(chunk, list(getattr(profile, "noise_words", []) or [])):
        return True

    _, _, digits, digits_ratio, cjk_ratio = _chunk_stats(chunk)
    if digits >= 18 and digits_ratio > 0.2:
        return True
    if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.06:
        return True

    tpl = _template_score(chunk, query_has_cjk=query_has_cjk, title_patterns=title_patterns, mode="chunk")
    return tpl >= float(tpl_threshold)


def _post_process_chunk_scores(
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


__all__ = ["EnrichStep"]
