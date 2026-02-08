"""Web page crawler (sync + async) for enrichment.

The crawler is responsible for:
- fetching bytes from a URL with size/time limits
- decoding with best-effort charset detection (header/meta/BOM/heuristics)
- extracting visible text blocks (HTML -> text) suitable for chunking/scoring
"""

from __future__ import annotations

import codecs
import html as html_mod
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Literal
from typing_extensions import override
from urllib.parse import urlparse

import httpx
import requests

from search_core.text import TextUtils

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from search_core.config import WebFetchConfig

logger = logging.getLogger(__name__)

ContentKind = Literal["html", "text"]


class _WebCrawlerBase:
    """Shared decoding/extraction logic for both sync and async crawlers."""

    fetch_cfg: WebFetchConfig
    user_agent: str

    def __init__(self, *, fetch_cfg: WebFetchConfig, user_agent: str) -> None:
        self.fetch_cfg = fetch_cfg
        self.user_agent = user_agent

    def _guess_apparent_encoding(self, data: bytes) -> str | None:
        """Best-effort charset guess from bytes without consuming streamed responses."""

        sample = data[:65536]

        # requests depends on charset_normalizer on py3; use it if available.
        with suppress(Exception):
            from charset_normalizer import from_bytes  # noqa: PLC0415

            best = from_bytes(sample).best()
            enc = getattr(best, "encoding", None) if best is not None else None
            if enc:
                return str(enc)

        # Fallback for environments that still have chardet.
        with suppress(Exception):
            import chardet  # noqa: PLC0415

            det = chardet.detect(sample)
            enc = det.get("encoding") if isinstance(det, dict) else None
            if enc:
                return str(enc)

        return None

    def _extract_blocks(self, raw: str, *, kind: ContentKind) -> list[str]:
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

        if self.fetch_cfg.max_extracted_chars and len(text) > int(
            self.fetch_cfg.max_extracted_chars
        ):
            text = text[: int(self.fetch_cfg.max_extracted_chars)]

        # Preserve newlines for block segmentation; only normalize spaces/tabs.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [TextUtils.clean_whitespace(line) for line in text.split("\n")]
        return [line for line in lines if line]

    @staticmethod
    def _looks_like_html(sample: bytes) -> bool:
        head = sample[:8192].lower()
        return any(
            tok in head
            for tok in (b"<!doctype", b"<html", b"<meta", b"<body", b"</p", b"</div")
        )

    def _decode_best_effort(
        self,
        data: bytes,
        *,
        content_type: str | None,
        resp_encoding: str | None = None,
        apparent_encoding: str | None = None,
    ) -> tuple[str, ContentKind]:
        if not data:
            return ("", "text")

        kind: ContentKind = "html" if self._looks_like_html(data) else "text"

        candidates: list[str] = []
        declared: list[str] = []

        # BOM hints.
        if data.startswith(codecs.BOM_UTF8):
            candidates.append("utf-8-sig")
            declared.append("utf-8-sig")
        if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            candidates.append("utf-16")
            declared.append("utf-16")

        # Charset from Content-Type header.
        if content_type:
            m = _CT_CHARSET_RE.search(content_type)
            if m:
                cs = m.group(1)
                candidates.append(cs)
                declared.append(cs)

        # Charset from HTML meta.
        meta = self._extract_meta_charset(data)
        if meta:
            candidates.append(meta)
            declared.append(meta)

        # Requests hints.
        if resp_encoding:
            candidates.append(resp_encoding)
        if apparent_encoding:
            candidates.append(apparent_encoding)

        # Common fallbacks.
        candidates.extend(
            [
                "utf-8",
                "utf-8-sig",
                "gb18030",
                "shift_jis",
                "euc_jp",
                "iso-2022-jp",
                "cp1252",
                "latin-1",
            ]
        )

        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            c = (c or "").strip()
            if not c:
                continue
            lc = c.lower()
            if lc in seen:
                continue
            seen.add(lc)
            ordered.append(c)

        # If the page declares a charset (header/meta/BOM), prefer it strongly.
        for enc in declared:
            enc = (enc or "").strip()
            if not enc:
                continue
            try:
                text = data.decode(enc, errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Declared charset decode %r failed: %s", enc, exc)
                continue
            text = text.replace("\x00", "")
            total = max(1, len(text))
            repl = text.count("\ufffd") / total
            ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t") / total
            if repl <= 0.001 and ctrl <= 0.001:
                best_text = text
                best_text = best_text.replace("\r\n", "\n").replace("\r", "\n")
                best_text = _WS_RE.sub(" ", best_text) if kind == "text" else best_text
                return best_text, kind

        best_text = ""
        best_key: tuple[float, float, float, float] | None = None

        for enc in ordered:
            try:
                text = data.decode(enc, errors="replace")
            except Exception as exc:  # noqa: BLE001
                logger.debug("Charset decode attempt %r failed: %s", enc, exc)
                continue

            text = text.replace("\x00", "")
            total = max(1, len(text))
            repl = text.count("\ufffd") / total
            ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t") / total
            cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", text)) / total
            short_penalty = 1.0 if len(text) < 200 else 0.0

            key = (repl, ctrl, -cjk, short_penalty)
            if best_key is None or key < best_key:
                best_key = key
                best_text = text

        if not best_text:
            best_text = data.decode("utf-8", errors="replace").replace("\x00", "")

        best_text = best_text.replace("\r\n", "\n").replace("\r", "\n")
        best_text = _WS_RE.sub(" ", best_text) if kind == "text" else best_text
        return best_text, kind

    @staticmethod
    def _extract_meta_charset(data: bytes) -> str | None:
        head = data[:16384].decode("ascii", errors="ignore")
        m = _META_CHARSET_RE.search(head)
        if not m:
            return None
        cs = (m.group(1) or "").strip()
        return cs or None


@dataclass(frozen=True)
class CrawlBlocksResult:
    """Result of fetching and extracting visible blocks from a page."""

    blocks: list[str]
    content_type: str | None = None
    error: str | None = None


class VisibleTextParser(HTMLParser):
    """Very small HTML parser that keeps only visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._buf: list[str] = []
        self._skip_stack: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        """Start tag handler (skips script/style/noscript)."""
        lowered = (tag or "").lower()
        if lowered in {"script", "style", "noscript"}:
            self._skip_stack.append(lowered)

    @override
    def handle_endtag(self, tag: str) -> None:
        """End tag handler (adds newlines for common block tags)."""
        lowered = (tag or "").lower()
        if self._skip_stack and self._skip_stack[-1] == lowered:
            self._skip_stack.pop()
        if lowered in {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self._buf.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        """Text node handler."""
        if self._skip_stack:
            return
        if data:
            self._buf.append(data)

    @override
    def handle_entityref(self, name: str) -> None:
        """Entity reference handler (keeps raw entity)."""
        if self._skip_stack:
            return
        self._buf.append(f"&{name};")

    @override
    def handle_charref(self, name: str) -> None:
        """Numeric character reference handler (keeps raw reference)."""
        if self._skip_stack:
            return
        self._buf.append(f"&#{name};")

    def get_text(self) -> str:
        """Return concatenated visible text."""
        return "".join(self._buf)


_WS_RE = re.compile(r"\s+")
_META_CHARSET_RE = re.compile(
    r"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)
_CT_CHARSET_RE = re.compile(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE)


class WebCrawler(_WebCrawlerBase):
    """Fetch + decode + extract blocks from a URL.

    This exists to keep WebEnricher focused on chunking/scoring logic.
    """

    def __init__(
        self,
        *,
        fetch_cfg: WebFetchConfig,
        user_agent: str,
        fetcher: Callable[[str], bytes | str] | None = None,
    ) -> None:
        super().__init__(fetch_cfg=fetch_cfg, user_agent=user_agent)
        self._fetcher = fetcher

    def fetch_blocks(self, url: str) -> CrawlBlocksResult:  # noqa: PLR0911
        """Fetch a URL and extract visible text blocks (sync).

        Args:
            url: Target URL (http/https).

        Returns:
            A :class:`CrawlBlocksResult` containing extracted blocks or an error message.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return CrawlBlocksResult(
                blocks=[],
                content_type=None,
                error=f"unsupported scheme: {parsed.scheme}",
            )

        if self._fetcher is not None:
            try:
                payload = self._fetcher(url)
            except Exception as exc:  # noqa: BLE001
                return CrawlBlocksResult(blocks=[], content_type=None, error=str(exc))

            if payload is None:
                return CrawlBlocksResult(
                    blocks=[], content_type=None, error="empty response"
                )

            if isinstance(payload, bytes):
                text, kind = self._decode_best_effort(payload, content_type=None)
                blocks = self._extract_blocks(text, kind=kind)
                return CrawlBlocksResult(
                    blocks=blocks,
                    content_type=None,
                    error=None if blocks else "no blocks extracted",
                )

            # Assume string is HTML.
            text = str(payload)
            blocks = self._extract_blocks(text, kind="html")
            return CrawlBlocksResult(
                blocks=blocks,
                content_type=None,
                error=None if blocks else "no blocks extracted",
            )

        headers = {"User-Agent": self.user_agent}
        try:
            resp = requests.get(
                url,
                headers=headers,
                timeout=self.fetch_cfg.timeout,
                stream=True,
            )  # noqa: S113
        except requests.RequestException as exc:
            return CrawlBlocksResult(blocks=[], content_type=None, error=str(exc))

        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            return CrawlBlocksResult(
                blocks=[],
                content_type=resp.headers.get("content-type"),
                error=str(exc),
            )

        content_type = resp.headers.get("content-type") or ""
        if content_type:
            ct_lower = content_type.lower()
            allowed = tuple(self.fetch_cfg.allow_content_types or ())
            if allowed and not any(a in ct_lower for a in allowed):
                # If header looks wrong but content looks like HTML, still try.
                pass

        chunks: list[bytes] = []
        total = 0
        try:
            for part in resp.iter_content(chunk_size=64 * 1024):
                if not part:
                    continue
                total += len(part)
                if total > int(self.fetch_cfg.max_bytes):
                    return CrawlBlocksResult(
                        blocks=[],
                        content_type=content_type,
                        error=f"exceeded max_bytes={self.fetch_cfg.max_bytes}",
                    )
                chunks.append(part)
        finally:
            resp.close()

        data = b"".join(chunks)
        apparent = self._guess_apparent_encoding(data)
        text, kind = self._decode_best_effort(
            data,
            content_type=content_type or None,
            resp_encoding=resp.encoding,
            apparent_encoding=apparent,
        )
        blocks = self._extract_blocks(text, kind=kind)
        return CrawlBlocksResult(
            blocks=blocks,
            content_type=content_type or None,
            error=None if blocks else "no blocks extracted",
        )


class AsyncWebCrawler(_WebCrawlerBase):
    """Async fetch + decode + extract blocks from a URL (httpx/anyio)."""

    def __init__(
        self,
        *,
        fetch_cfg: WebFetchConfig,
        user_agent: str,
        afetcher: Callable[[str], Awaitable[bytes | str]] | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(fetch_cfg=fetch_cfg, user_agent=user_agent)
        self._afetcher = afetcher
        self._async_client: httpx.AsyncClient | None = async_client
        self._owns_async_client = async_client is None

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
        return self._async_client

    async def afetch_blocks(self, url: str) -> CrawlBlocksResult:  # noqa: PLR0911
        """Fetch a URL and extract visible text blocks (async).

        Args:
            url: Target URL (http/https).

        Returns:
            A :class:`CrawlBlocksResult` containing extracted blocks or an error message.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return CrawlBlocksResult(
                blocks=[],
                content_type=None,
                error=f"unsupported scheme: {parsed.scheme}",
            )

        if self._afetcher is not None:
            try:
                payload = await self._afetcher(url)
            except Exception as exc:  # noqa: BLE001
                return CrawlBlocksResult(blocks=[], content_type=None, error=str(exc))

            if payload is None:
                return CrawlBlocksResult(
                    blocks=[], content_type=None, error="empty response"
                )

            if isinstance(payload, bytes):
                text, kind = self._decode_best_effort(payload, content_type=None)
                blocks = self._extract_blocks(text, kind=kind)
                return CrawlBlocksResult(
                    blocks=blocks,
                    content_type=None,
                    error=None if blocks else "no blocks extracted",
                )

            text = str(payload)
            blocks = self._extract_blocks(text, kind="html")
            return CrawlBlocksResult(
                blocks=blocks,
                content_type=None,
                error=None if blocks else "no blocks extracted",
            )

        headers = {"User-Agent": self.user_agent}
        content_type = None

        try:
            async with self._get_async_client().stream(
                "GET",
                url,
                headers=headers,
                timeout=self.fetch_cfg.timeout,
                follow_redirects=True,
            ) as resp:
                content_type = resp.headers.get("content-type")
                if content_type:
                    ct_lower = content_type.lower()
                    allowed = tuple(self.fetch_cfg.allow_content_types or ())
                    if allowed and not any(a in ct_lower for a in allowed):
                        pass

                try:
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    return CrawlBlocksResult(
                        blocks=[],
                        content_type=content_type,
                        error=str(exc),
                    )

                chunks: list[bytes] = []
                total = 0
                async for part in resp.aiter_bytes():
                    if not part:
                        continue
                    total += len(part)
                    if total > int(self.fetch_cfg.max_bytes):
                        return CrawlBlocksResult(
                            blocks=[],
                            content_type=content_type,
                            error=f"exceeded max_bytes={self.fetch_cfg.max_bytes}",
                        )
                    chunks.append(part)
        except httpx.HTTPError as exc:
            return CrawlBlocksResult(blocks=[], content_type=None, error=str(exc))

        data = b"".join(chunks)
        apparent = self._guess_apparent_encoding(data)
        text, kind = self._decode_best_effort(
            data,
            content_type=content_type or None,
            resp_encoding=None,
            apparent_encoding=apparent,
        )
        blocks = self._extract_blocks(text, kind=kind)
        return CrawlBlocksResult(
            blocks=blocks,
            content_type=content_type or None,
            error=None if blocks else "no blocks extracted",
        )

    async def aclose(self) -> None:
        """Close the internally-owned ``httpx.AsyncClient`` if present."""
        if self._async_client is None:
            return
        if not self._owns_async_client:
            return
        await self._async_client.aclose()
        self._async_client = None


__all__ = ["CrawlBlocksResult", "WebCrawler", "AsyncWebCrawler"]
