from __future__ import annotations

import html as html_mod
import re
from typing import TYPE_CHECKING
from typing_extensions import override

from bs4 import BeautifulSoup

from serpsage.components.extract.html_basic import BasicHtmlExtractor
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.contracts.services import ExtractorBase
from serpsage.models.extract import ExtractedText
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from bs4.element import Tag  # type: ignore[import-untyped]

    from serpsage.core.runtime import Runtime


_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
}

_DROP_ROLES = {"navigation", "banner", "contentinfo"}

_ARIA_NOISE_RE = re.compile(r"(nav|menu|toc|breadcrumb)", re.IGNORECASE)

_NOISE_ID_CLASS_RE = re.compile(
    r"(nav|footer|header|sidebar|breadcrumb|menu|toc|comment|ads|cookie|banner|"
    r"mw-navigation|mw-panel|mw-head|vector-|mw-page-base|mw-head-base|"
    r"catlinks|printfooter|mw-footer)",
    re.IGNORECASE,
)

_BLOCK_TAGS = ("p", "li", "h1", "h2", "h3", "h4", "blockquote", "pre")


def _text_len(tag: Tag) -> int:
    try:
        t = tag.get_text(" ", strip=True)
    except Exception:
        return 0
    return len(t or "")


def _link_density(tag: Tag) -> float:
    """Approximate link density by anchor text length / total text length."""
    total = _text_len(tag)
    if total <= 0:
        return 1.0
    try:
        link_text = " ".join(a.get_text(" ", strip=True) for a in tag.find_all("a"))
    except Exception:
        return 0.0
    link_len = len(link_text or "")
    return float(link_len) / float(total)


class MainContentHtmlExtractor(ExtractorBase):
    """HTML extractor that tries hard to focus on main/article content.

    It is still best-effort and intentionally lightweight:
    - decode using existing heuristics
    - drop common noise regions and MediaWiki navigation
    - pick a main container (main/article/role=main, MediaWiki ids, or best-scoring)
    - emit blocks from common content tags
    """

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._fallback = BasicHtmlExtractor(rt=rt)
        self.bind_deps(self._fallback)

    @override
    def extract(
        self, *, url: str, content: bytes, content_type: str | None
    ) -> ExtractedText:
        _ = url
        apparent = guess_apparent_encoding(content)
        text, kind = decode_best_effort(
            content,
            content_type=content_type,
            resp_encoding=None,
            apparent_encoding=apparent,
        )

        if kind == "text":
            # Plain text: reuse existing basic extractor (already handles size limits).
            return self._fallback.extract(
                url=url, content=content, content_type=content_type
            )

        try:
            soup = BeautifulSoup(text, "html.parser")
        except Exception:
            return self._fallback.extract(
                url=url, content=content, content_type=content_type
            )

        self._drop_noise(soup)

        container = self._pick_container(soup)
        blocks = self._extract_blocks(container or soup)

        # Fallback: if we got too little content, try whole page; then basic extractor.
        if len(blocks) < 3:
            blocks = self._extract_blocks(soup)
        if len(blocks) < 3:
            return self._fallback.extract(
                url=url, content=content, content_type=content_type
            )

        blocks = [b for b in (clean_whitespace(b) for b in blocks) if b]

        joined = "\n".join(blocks)
        joined = html_mod.unescape(joined)
        return ExtractedText(text=joined, blocks=list(blocks))

    def _drop_noise(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(_DROP_TAGS)):
            try:
                tag.decompose()
            except Exception:  # noqa: S112
                continue

        # Attribute-based noise.
        for t in list(soup.find_all(True)):
            try:
                role = (str(t.get("role")) or "").strip().lower()
                if role and role in _DROP_ROLES:
                    t.decompose()
                    continue
                aria = (str(t.get("aria-label")) or "").strip()
                if aria and _ARIA_NOISE_RE.search(aria):
                    t.decompose()
                    continue
                ident = " ".join(
                    [
                        str(t.get("id") or ""),
                        " ".join(t.get("class") or []),
                    ]
                ).strip()
                if ident and _NOISE_ID_CLASS_RE.search(ident):
                    t.decompose()
            except Exception:  # noqa: S112
                continue

    def _pick_container(self, soup: BeautifulSoup) -> Tag | None:
        # Strong picks first.
        for sel in (
            "main",
            "article",
            '[role="main"]',
            "#mw-content-text",
            "#bodyContent",
            "#content",
        ):
            try:
                t = soup.select_one(sel)
            except Exception:
                t = None
            if t is not None and _text_len(t) >= 200:
                return t

        # Otherwise pick best scoring candidate.
        best: Tag | None = None
        best_score = -1.0
        try:
            candidates: list[Tag] = list(
                soup.find_all(["div", "section", "article", "main"])
            )
        except Exception:
            candidates = []
        for c in candidates:
            tl = _text_len(c)
            if tl < 200:
                continue
            ld = _link_density(c)
            if ld > 0.45:
                continue
            # Prefer longer text with low link density.
            score = float(tl) * (1.0 - float(ld))
            if score > best_score:
                best_score = score
                best = c
        return best

    def _extract_blocks(self, root: Tag | BeautifulSoup) -> list[str]:
        blocks: list[str] = []
        try:
            items = root.find_all(list(_BLOCK_TAGS))
        except Exception:
            return blocks

        for t in items:
            try:
                txt = t.get_text(" ", strip=True)
            except Exception:  # noqa: S112
                continue
            txt = clean_whitespace(txt or "")
            if not txt:
                continue
            blocks.append(txt)

        return blocks


__all__ = ["MainContentHtmlExtractor"]
