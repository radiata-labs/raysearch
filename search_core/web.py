from __future__ import annotations

import html as html_mod
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Literal
from typing_extensions import override
from urllib.parse import urlparse

import requests

from .config import ScoringConfig
from .models import PageChunk, PageEnrichment
from .scorer import ScoringEngine
from .tools import compile_patterns, is_duplicate_text
from .utils import TextUtils

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import (
        SearchContextConfig,
        WebChunkingConfig,
        WebDepthPreset,
        WebEnrichmentConfig,
    )
    from .models import SearchResult

logger = logging.getLogger(__name__)

ContentKind = Literal["html", "text"]


@dataclass(frozen=True)
class FetchResult:
    text: str
    error: str | None = None
    content_type: str | None = None


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._buf: list[str] = []
        self._skip_stack: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = (tag or "").lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_stack.append(lowered)

    @override
    def handle_endtag(self, tag: str) -> None:
        lowered = (tag or "").lower()
        if self._skip_stack and self._skip_stack[-1] == lowered:
            self._skip_stack.pop()
        if lowered in {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self._buf.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if data:
            self._buf.append(data)

    @override
    def handle_entityref(self, name: str) -> None:
        if self._skip_stack:
            return
        self._buf.append(f"&{name};")

    @override
    def handle_charref(self, name: str) -> None:
        if self._skip_stack:
            return
        self._buf.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self._buf)


_WS_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"([\u3002\uFF01\uFF1F!?;\uFF1B.\n])")
_LONG_SENT_SPLIT_RE = re.compile(r"([,\uFF0C\u3001\t ])")


class WebEnricher:
    def __init__(
        self,
        config: WebEnrichmentConfig,
        *,
        user_agent: str,
        fetcher: Callable[[str], str] | None = None,
        scorer: ScoringEngine | None = None,
    ) -> None:
        self.config = config
        self.user_agent = user_agent
        self._fetcher = fetcher
        self.scorer = scorer or ScoringEngine(ScoringConfig())

    def fetch(self, url: str) -> FetchResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return FetchResult(text="", error=f"unsupported scheme: {parsed.scheme}")

        headers = {"User-Agent": self.user_agent}
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=self.config.fetch.timeout,
                stream=True,
            )  # noqa: S113
        except requests.RequestException as exc:
            return FetchResult(text="", error=str(exc))

        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            return FetchResult(
                text="",
                error=str(exc),
                content_type=resp.headers.get("content-type"),
            )

        content_type = resp.headers.get("content-type") or ""
        if content_type:
            ct = content_type.lower()
            if self.config.fetch.allow_content_types and not any(
                allowed in ct for allowed in self.config.fetch.allow_content_types
            ):
                return FetchResult(
                    text="",
                    error=f"unsupported content-type: {content_type}",
                    content_type=content_type,
                )

        chunks: list[bytes] = []
        total = 0
        try:
            for part in resp.iter_content(chunk_size=64 * 1024):
                if not part:
                    continue
                total += len(part)
                if total > self.config.fetch.max_bytes:
                    return FetchResult(
                        text="",
                        error=f"exceeded max_bytes={self.config.fetch.max_bytes}",
                        content_type=content_type,
                    )
                chunks.append(part)
        finally:
            resp.close()

        data = b"".join(chunks)
        encoding = resp.encoding or "utf-8"
        try:
            text = data.decode(encoding, errors="replace")
        except LookupError:
            text = data.decode("utf-8", errors="replace")
        return FetchResult(text=text, error=None, content_type=content_type)

    def extract_text(self, raw: str, *, kind: ContentKind) -> str:
        if not raw:
            return ""
        if kind == "text":
            text = TextUtils.clean_whitespace(raw)
        else:
            parser = VisibleTextParser()
            try:
                parser.feed(raw)
                parser.close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to parse HTML")
            text = html_mod.unescape(parser.get_text())
            text = _WS_RE.sub(" ", text).strip()

        if (
            self.config.fetch.max_extracted_chars
            and len(text) > self.config.fetch.max_extracted_chars
        ):
            text = text[: self.config.fetch.max_extracted_chars]
        return text

    def extract_blocks(self, raw: str, *, kind: ContentKind) -> list[str]:
        """Extract paragraph-ish blocks.

        This is intentionally low-tech: the goal is to keep boilerplate/nav as small
        blocks so it can be filtered before sentence chunking.
        """

        if not raw:
            return []

        if kind == "text":
            text = raw
        else:
            parser = VisibleTextParser()
            try:
                parser.feed(raw)
                parser.close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to parse HTML")
            text = html_mod.unescape(parser.get_text())

        # Preserve newlines for block segmentation; only normalize spaces/tabs.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [TextUtils.clean_whitespace(line) for line in text.split("\n")]
        return [line for line in lines if line]

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
            if self._has_noise_word(block, context_config):
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
        min_query_hits = max(0, int(self.config.select.min_query_hits))

        filtered: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks):
            if self._hard_drop_chunk(
                chunk,
                context_config=context_config,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
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

        base = self.scorer.score(
            [c for _, c in filtered],
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        base_scores = [float(s) for s, _ in base]

        min_score = max(
            float(self.config.select.min_chunk_score),
            float(self.config.select.min_final_score),
        )
        intent_missing_penalty = float(self.config.select.intent_missing_penalty)
        tpl_w = float(self.config.select.template_penalty_weight)
        tpl_b = float(self.config.select.template_penalty_bias)
        tpl_hard = float(self.config.select.template_hard_drop_threshold)
        early_bonus = float(self.config.select.early_bonus)

        scored: list[tuple[float, str]] = []
        for (pos, chunk), base_score in zip(filtered, base_scores, strict=False):
            tpl = self.template_score(
                chunk,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="chunk",
            )
            if tpl >= tpl_hard:
                continue

            final = float(base_score)
            if early_bonus > 1.0:
                final *= early_bonus ** (-pos)
            final = final * (1.0 - tpl * tpl_w) - tpl * tpl_b

            if intent_tokens and not self._has_any_token(chunk, intent_tokens):
                final -= intent_missing_penalty

            final = max(0.0, min(1.0, float(final)))
            if final < min_score:
                continue

            scored.append((final, chunk))

        return scored

    @staticmethod
    def _has_noise_word(text: str, context_config: SearchContextConfig) -> bool:
        lowered = TextUtils.normalize_text(text)
        for word in context_config.noise_words:
            if word and word.lower() in lowered:
                return True
        return False

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

    @staticmethod
    def _has_any_token(text: str, tokens: list[str]) -> bool:
        lowered = (text or "").lower()
        return any(t and t.lower() in lowered for t in tokens)

    @staticmethod
    def _query_hit_count(chunk: str, *, query_tokens: list[str]) -> int:
        lowered = (chunk or "").lower()
        return sum(1 for t in set(query_tokens) if t and t.lower() in lowered)

    @staticmethod
    def _chunk_stats(chunk: str) -> tuple[int, float, int, float, float]:
        tokens = TextUtils.tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", chunk))
        cjk_ratio = cjk_count / max(1, len(chunk))
        return token_count, unique_ratio, digits, digits_ratio, cjk_ratio

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

        if self._has_noise_word(chunk, context_config):
            return True

        _token_count, _unique_ratio, digits, digits_ratio, cjk_ratio = (
            self._chunk_stats(chunk)
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
        return tpl >= float(self.config.select.template_hard_drop_threshold)

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
                if self._fetcher is not None:
                    raw = self._fetcher(url)
                    if not raw:
                        return (item, [], [], "empty response")
                    blocks = self.extract_blocks(raw, kind="html")
                else:
                    fetched = self.fetch(url)
                    if fetched.error:
                        return (item, [], [], fetched.error)
                    kind: ContentKind = "html"
                    ct = (fetched.content_type or "").lower()
                    if "text/plain" in ct and "text/html" not in ct:
                        kind = "text"
                    blocks = self.extract_blocks(fetched.text, kind=kind)

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

                scored.sort(key=lambda t: t[0], reverse=True)
                top: list[tuple[float, str]] = []
                dedupe_th = float(self.config.select.dedupe_threshold)
                for score, chunk in scored:
                    if is_duplicate_text(
                        chunk, [c for _, c in top], threshold=dedupe_th
                    ):
                        continue
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


__all__ = ["FetchResult", "WebEnricher"]
