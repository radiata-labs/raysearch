from __future__ import annotations

import contextlib
import html
import re
from typing import TYPE_CHECKING
from typing_extensions import override

from bs4 import BeautifulSoup

from serpsage.components.extract.pdf import extract_pdf_document
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.contracts.services import ExtractorBase
from serpsage.models.extract import ExtractedDocument
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from bs4.element import Tag

    from serpsage.core.runtime import Runtime

_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "canvas",
    "button",
    "input",
    "select",
    "textarea",
    "form",
}
_NOISE_TAGS = {"nav", "header", "footer", "aside"}
_NOISE_ROLE = {"navigation", "banner", "contentinfo", "complementary", "search"}
_NOISE_PATTERNS = re.compile(
    r"(nav|footer|header|sidebar|breadcrumb|menu|cookie|consent|share|ads?|"
    r"sponsored|comment|recommend|related|newsletter|popup|modal|pager)",
    re.IGNORECASE,
)
_BLOCK_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "pre",
    "blockquote",
    "table",
}
_WS_RE = re.compile(r"\s+")
_MD_PREFIX_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+[.)]\s+|#{1,6}\s+|>\s+)")


class MarkdownExtractor(ExtractorBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
    ) -> ExtractedDocument:
        kind = classify_content_kind(
            content_type=content_type, url=url, content=content
        )
        if kind == "pdf":
            return extract_pdf_document(content=content)

        apparent = guess_apparent_encoding(content)
        decoded, decoded_kind = decode_best_effort(
            content,
            content_type=content_type,
            apparent_encoding=apparent,
        )
        if kind == "unknown":
            kind = "html" if decoded_kind == "html" else "text"

        if kind == "text":
            return self._extract_text(decoded)
        if kind != "html":
            return ExtractedDocument(content_kind="binary")

        soup = BeautifulSoup(decoded, "html.parser")
        title = clean_whitespace(
            html.unescape(
                soup.title.get_text(" ", strip=True) if soup.title is not None else ""
            )
        )
        self._drop_noise(soup)
        root = self._pick_main_container(soup)
        markdown = self._to_markdown(root)
        if not markdown:
            markdown = self._to_markdown(soup)
        markdown = self._dedupe_markdown(markdown)
        plain = self._markdown_to_plain(markdown)
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title=title,
            content_kind="html",
            stats={
                "markdown_chars": len(markdown),
                "plain_chars": len(plain),
                "lines": len(markdown.splitlines()),
            },
        )

    def _extract_text(self, text: str) -> ExtractedDocument:
        lines = [clean_whitespace(x) for x in text.splitlines() if clean_whitespace(x)]
        markdown = "\n\n".join(lines)
        plain = "\n".join(lines)
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title="",
            content_kind="text",
            stats={"plain_chars": len(plain)},
        )

    def _drop_noise(self, soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(_DROP_TAGS)):
            with contextlib.suppress(Exception):
                tag.decompose()

        for t in list(soup.find_all(True)):
            try:
                name = (t.name or "").lower()
                if name in _NOISE_TAGS:
                    t.decompose()
                    continue
                role = str(t.get("role") or "").lower()
                if role in _NOISE_ROLE:
                    t.decompose()
                    continue
                ident = " ".join(
                    [
                        str(t.get("id") or ""),
                        " ".join(t.get("class") or []),
                        str(t.get("aria-label") or ""),
                    ]
                )
                if ident and _NOISE_PATTERNS.search(ident):
                    t.decompose()
            except Exception:
                pass

    def _pick_main_container(self, soup: BeautifulSoup) -> Tag | BeautifulSoup:
        for sel in ("main", "article", '[role="main"]', "#content", "#main"):
            found = soup.select_one(sel)
            if found is not None and len(found.get_text(" ", strip=True)) >= 200:
                return found

        best = soup
        best_score = -1.0
        candidates = soup.find_all(["article", "main", "section", "div"])
        for c in candidates:
            text = c.get_text(" ", strip=True)
            text_len = len(text)
            if text_len < 200:
                continue
            links = " ".join(a.get_text(" ", strip=True) for a in c.find_all("a"))
            link_density = len(links) / max(1, text_len)
            punct_density = len(re.findall(r"[,.!?;:，。！？；：]", text)) / max(
                1, text_len
            )
            heading_count = len(c.find_all(["h1", "h2", "h3"]))
            score = (
                text_len * (1.0 - min(0.9, link_density)) * (1.0 + punct_density)
                + heading_count * 40
            )
            if score > best_score:
                best_score = score
                best = c
        return best

    def _to_markdown(self, root: Tag | BeautifulSoup) -> str:
        lines: list[str] = []
        for el in root.find_all(list(_BLOCK_TAGS)):
            if self._has_block_ancestor(el, root):
                continue
            line = self._render_block(el)
            if not line:
                continue
            lines.append(line)
        lines = [x for x in lines if x]
        return "\n\n".join(lines).strip()

    def _has_block_ancestor(self, tag: Tag, root: Tag | BeautifulSoup) -> bool:
        parent = tag.parent
        while parent is not None and parent is not root:
            if getattr(parent, "name", None) in _BLOCK_TAGS:
                return True
            parent = parent.parent
        return False

    def _render_block(self, tag: Tag) -> str:
        name = (tag.name or "").lower()
        text = clean_whitespace(html.unescape(tag.get_text(" ", strip=True)))
        if not text:
            return ""
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            level = min(6, max(1, int(name[1])))
            return f"{'#' * level} {text}"
        if name == "li":
            return f"- {text}"
        if name == "blockquote":
            return "\n".join(
                f"> {clean_whitespace(part)}"
                for part in text.split("\n")
                if part.strip()
            )
        if name == "pre":
            code = tag.get_text("\n", strip=False).strip("\n")
            if not code:
                return ""
            return f"```\n{code}\n```"
        if name == "table":
            return self._render_table(tag)
        return text

    def _render_table(self, table: Tag) -> str:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                clean_whitespace(td.get_text(" ", strip=True))
                for td in tr.find_all(["th", "td"])
            ]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        head = rows[0]
        sep = ["---"] * width
        body = rows[1:]
        out = [
            "| " + " | ".join(head) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        out.extend("| " + " | ".join(r) + " |" for r in body)
        return "\n".join(out)

    def _dedupe_markdown(self, markdown: str) -> str:
        lines = [x.rstrip() for x in markdown.splitlines()]
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized = _WS_RE.sub(" ", line).strip()
            if not normalized:
                if out and out[-1] != "":
                    out.append("")
                continue
            key = normalized.lower()
            if len(key) > 8 and key in seen:
                continue
            seen.add(key)
            out.append(line)
        # collapse excessive blank lines
        compact: list[str] = []
        blank = 0
        for line in out:
            if line == "":
                blank += 1
                if blank <= 1:
                    compact.append(line)
                continue
            blank = 0
            compact.append(line)
        return "\n".join(compact).strip()

    def _markdown_to_plain(self, markdown: str) -> str:
        out: list[str] = []
        in_code = False
        for raw in markdown.splitlines():
            line = raw.strip()
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                out.append(clean_whitespace(line))
                continue
            line = _MD_PREFIX_RE.sub("", line)
            line = re.sub(r"`([^`]+)`", r"\1", line)
            line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", line)
            line = line.replace("|", " ")
            line = clean_whitespace(line)
            if line:
                out.append(line)
        return "\n".join(out).strip()


__all__ = ["MarkdownExtractor"]
