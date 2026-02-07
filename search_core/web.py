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

from .config import ScoreNormalizationConfig
from .models import PageChunk, PageEnrichment
from .scoring import BM25_AVAILABLE, blend_scores, bm25_scores, normalize_scores
from .utils import TextUtils

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import (
        RankingConfig,
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
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
_DATE_JP_RE = re.compile(r"\d{4}\u5e74\d{1,2}\u6708")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SEP_RE = re.compile(r"[\|/>\u00bb]")
_TEMPLATE_KEYWORDS = {
    "archive",
    "archives",
    "sitemap",
    "privacy",
    "terms",
    "about",
    "contact",
    "rss",
    "login",
    "signup",
    "subscribe",
    "category",
    "categories",
    "tag",
    "tags",
    "next",
    "previous",
    "older",
    "newer",
    "page",
    "pages",
    "copyright",
    "all rights reserved",
    "cookie",
    "cookies",
    "policy",
    "footer",
    "header",
    "navigation",
    "\u30a2\u30fc\u30ab\u30a4\u30d6",
    "\u30ab\u30c6\u30b4\u30ea",
    "\u30ab\u30c6\u30b4\u30ea\u30fc",
    "\u30b5\u30a4\u30c8\u30de\u30c3\u30d7",
    "\u30d7\u30e9\u30a4\u30d0\u30b7\u30fc",
    "\u5229\u7528\u898f\u7d04",
    "\u304a\u554f\u3044\u5408\u308f\u305b",
    "\u6708\u3092\u9078\u629e",
    "\u691c\u7d22",
    "\u30db\u30fc\u30e0",
    "\u30c8\u30c3\u30d7",
    "\u4e00\u89a7",
    "\u6b21\u3078",
    "\u524d\u3078",
    "\u5f52\u6863",
    "\u5206\u7c7b",
    "\u76ee\u5f55",
    "\u7ad9\u70b9\u5730\u56fe",
    "\u9690\u79c1",
    "\u6761\u6b3e",
    "\u8054\u7cfb\u6211\u4eec",
    "\u5173\u4e8e",
    "\u9996\u9875",
    "\u641c\u7d22",
    "\u4e0a\u4e00\u9875",
    "\u4e0b\u4e00\u9875",
}


class WebEnricher:
    def __init__(
        self,
        config: WebEnrichmentConfig,
        *,
        user_agent: str,
        fetcher: Callable[[str], str] | None = None,
        score_normalization: ScoreNormalizationConfig | None = None,
    ) -> None:
        self.config = config
        self.user_agent = user_agent
        self._fetcher = fetcher
        self.score_normalization = score_normalization or ScoreNormalizationConfig()

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

        title_patterns = self._compile_title_patterns(
            context_config.title_tail_patterns
        )

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
            if t >= float(self.config.scoring.block_hard_drop_threshold):
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
        ranking_config: RankingConfig,
    ) -> list[tuple[float, str]]:
        if not chunks:
            return []

        strategy: Literal["heuristic", "bm25", "hybrid"] = ranking_config.strategy
        if strategy in {"bm25", "hybrid"} and not BM25_AVAILABLE:
            strategy = "heuristic"

        title_patterns = self._compile_title_patterns(
            context_config.title_tail_patterns
        )
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))
        min_query_hits = max(0, int(self.config.scoring.min_query_hits))

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

        domain_bonus = self._domain_bonus(domain, context_config)
        heuristic_scores = []
        for idx, chunk in filtered:
            heuristic_scores.append(
                self._heuristic_score(
                    chunk,
                    query=query,
                    query_tokens=query_tokens,
                    intent_tokens=intent_tokens,
                    domain_bonus=domain_bonus,
                    position=idx,
                )
            )

        bm25: list[float] | None = None
        if strategy in {"bm25", "hybrid"} and BM25_AVAILABLE:
            bm25 = bm25_scores([chunk for _, chunk in filtered], query=query)

        final_scores = blend_scores(
            strategy=strategy,
            heuristic=heuristic_scores,
            bm25=bm25,
            weights=ranking_config.normalized_weights(),
        )

        # NOTE: Chunk thresholds are independent from RankingConfig.min_* (result-level scale).
        min_score = max(
            float(self.config.scoring.min_chunk_score),
            float(self.config.scoring.min_final_score),
        )
        intent_missing_penalty = float(self.config.scoring.intent_missing_penalty)
        tpl_w = float(self.config.scoring.template_penalty_weight)
        tpl_b = float(self.config.scoring.template_penalty_bias)
        tpl_hard = float(self.config.scoring.template_hard_drop_threshold)

        scored: list[tuple[float, str]] = []
        for (_, chunk), score in zip(filtered, final_scores, strict=False):
            tpl = self.template_score(
                chunk,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="chunk",
            )
            if tpl >= tpl_hard:
                continue

            final = float(score) * (1.0 - tpl * tpl_w) - tpl * tpl_b

            if intent_tokens and not self._has_any_token(chunk, intent_tokens):
                final -= intent_missing_penalty

            if final < min_score:
                continue
            scored.append((final, chunk))

        return scored

    def _heuristic_score(
        self,
        chunk: str,
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        domain_bonus: int,
        position: int,
    ) -> float:
        if not chunk:
            return 0.0

        lowered = chunk.lower()
        hits = [t for t in query_tokens if t and t.lower() in lowered]
        if not hits:
            return 0.0

        unique_hits = set(hits)
        score = 0.0
        score += 6.0 * len(unique_hits)
        score += 1.5 * sum(min(lowered.count(t.lower()), 5) for t in unique_hits)
        score += 5.0 * sum(1 for t in intent_tokens if t and t.lower() in lowered)

        normalized_query = TextUtils.normalize_text(query)
        if normalized_query and normalized_query in TextUtils.normalize_text(chunk):
            score += float(self.config.scoring.phrase_bonus)

        score += float(domain_bonus)

        if len(chunk) < 80:
            score *= 0.85

        early_bonus = float(self.config.scoring.early_bonus)
        if early_bonus > 1.0:
            score *= early_bonus ** (-position)

        return float(score)

    @staticmethod
    def _has_any_token(text: str, tokens: list[str]) -> bool:
        lowered = (text or "").lower()
        return any(t and t.lower() in lowered for t in tokens)

    @staticmethod
    def _query_hit_count(chunk: str, *, query_tokens: list[str]) -> int:
        lowered = (chunk or "").lower()
        return sum(1 for t in set(query_tokens) if t and t.lower() in lowered)

    @staticmethod
    def _has_noise_word(text: str, context_config: SearchContextConfig) -> bool:
        lowered = TextUtils.normalize_text(text)
        for word in context_config.noise_words:
            if word and word.lower() in lowered:
                return True
        return False

    @staticmethod
    def _domain_bonus(domain: str | None, context_config: SearchContextConfig) -> int:
        normalized = (domain or "").lower()
        if not normalized:
            return 0
        if normalized in context_config.domain_bonus:
            return context_config.domain_bonus[normalized]
        for key, value in context_config.domain_bonus.items():
            if normalized.endswith(key):
                return value
        return 0

    @staticmethod
    def _is_duplicate_chunk(chunk: str, kept: list[str], threshold: float) -> bool:
        if not kept:
            return False
        if threshold <= 0:
            return False

        a = TextUtils.normalize_text(chunk)
        if not a:
            return True
        a_grams = TextUtils.char_ngrams(a, 2)

        for b in kept:
            b_norm = TextUtils.normalize_text(b)
            if not b_norm:
                continue
            jac = TextUtils.jaccard(a_grams, TextUtils.char_ngrams(b_norm, 2))
            if jac >= threshold:
                return True
        return False

    @staticmethod
    def _compile_title_patterns(
        patterns: tuple[str, ...],
    ) -> list[re.Pattern[str]]:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            if not pattern:
                continue
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                logger.warning("Invalid title_tail_patterns regex: %s", pattern)
        return compiled

    def _chunk_stats(
        self,
        chunk: str,
    ) -> tuple[int, float, int, float, float]:
        tokens = TextUtils.tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(_CJK_CHAR_RE.findall(chunk))
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

        # Noise words are always hard drop.
        if self._has_noise_word(chunk, context_config):
            return True

        token_count, unique_ratio, digits, digits_ratio, cjk_ratio = self._chunk_stats(
            chunk
        )

        if digits >= 18 and digits_ratio > 0.2:
            return True
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.06:
            # If the query is CJK-heavy but the chunk isn't, it is very likely nav/boilerplate.
            return True

        # Only hard drop extreme boilerplate.
        tpl = self.template_score(
            chunk,
            query_has_cjk=query_has_cjk,
            title_patterns=title_patterns,
            mode="chunk",
        )
        return tpl >= float(self.config.scoring.template_hard_drop_threshold)

    def _is_template_like(self, chunk: str, lowered: str) -> bool:
        for kw in _TEMPLATE_KEYWORDS:
            if kw in lowered:
                return True

        if len(_DATE_JP_RE.findall(chunk)) >= 2:
            return True

        return len(_YEAR_RE.findall(chunk)) >= 6

    def template_score(
        self,
        text: str,
        *,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
        mode: Literal["block", "chunk"],
    ) -> float:
        """Return template-likeness score in [0, 1] (higher => more boilerplate)."""

        if not text:
            return 1.0

        lowered = TextUtils.normalize_text(text)
        if not lowered:
            return 1.0

        # Strong signals first.
        kw_strong = {
            "archive",
            "archives",
            "sitemap",
            "privacy",
            "terms",
            "cookie",
            "cookies",
            "\u30a2\u30fc\u30ab\u30a4\u30d6",
            "\u30b5\u30a4\u30c8\u30de\u30c3\u30d7",
            "\u30d7\u30e9\u30a4\u30d0\u30b7\u30fc",
            "\u5229\u7528\u898f\u7d04",
            "\u6708\u3092\u9078\u629e",
            "\u7ad9\u70b9\u5730\u56fe",
            "\u9690\u79c1",
            "\u6761\u6b3e",
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
            "\u4e0a\u4e00\u9875",
            "\u4e0b\u4e00\u9875",
            "\u4e00\u89a7",
            "\u76ee\u5f55",
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
            0.25 * len(_DATE_JP_RE.findall(text)) + 0.15 * len(_YEAR_RE.findall(text)),
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
        sep_ratio = len(_SEP_RE.findall(text)) / max(1, len(text))
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

        # Combine as noisy-OR: 1 - Π(1 - component_i)
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

        # Extra bias for tiny blocks with strong keywords.
        if mode == "block" and len(lowered) <= 64 and kw_component >= 0.35:
            tpl = max(tpl, 0.9)

        return float(max(0.0, min(1.0, tpl)))

    def enrich_results(
        self,
        results: list[SearchResult],
        *,
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
        context_config: SearchContextConfig,
        ranking_config: RankingConfig,
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
                    ranking_config=ranking_config,
                )
                if not scored:
                    return (item, [], [], "no matching chunks")

                # Normalize per-page chunk scores to [0,1] for output (ordering preserved).
                raw_scores = [float(s) for s, _ in scored]
                norm_scores = normalize_scores(raw_scores, self.score_normalization)
                scored_norm = [
                    (float(norm_scores[i]), scored[i][1]) for i in range(len(scored))
                ]

                scored_norm.sort(key=lambda t: t[0], reverse=True)
                top: list[tuple[float, str]] = []
                dedupe_th = float(self.config.scoring.dedupe_threshold)
                for score, chunk in scored_norm:
                    if self._is_duplicate_chunk(chunk, [c for _, c in top], dedupe_th):
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
