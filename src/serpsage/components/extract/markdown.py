from __future__ import annotations

import contextlib
import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from serpsage.core.tuning import extract_profile_for_depth
from serpsage.models.extract import ExtractedDocument
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from bs4.element import Tag

    from serpsage.core.runtime import Runtime

try:
    from readability import (
        Document as ReadabilityDocument,
    )

    READABILITY_AVAILABLE = True
except Exception:  # noqa: BLE001
    ReadabilityDocument = None
    READABILITY_AVAILABLE = False

try:
    import trafilatura  # type: ignore[import-not-found]

    TRAFILATURA_AVAILABLE = True
except Exception:  # noqa: BLE001
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

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
    "meta",
    "picture",
    "source",
}
_NOISE_TAGS = {"nav", "header", "footer", "aside"}
_NOISE_ROLE = {"navigation", "banner", "contentinfo", "complementary", "search"}
_NOISE_PATTERNS = re.compile(
    r"(nav|footer|header|sidebar|breadcrumb|menu|cookie|consent|share|ads?|"
    r"sponsored|comment|recommend|related|newsletter|popup|modal|pager|"
    r"subscribe|signin|login|toolbar|promo|banner|social)",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(
    r"(privacy policy|cookie policy|terms of service|all rights reserved|"
    r"sign up|subscribe|advertisement|sponsored content|related posts)",
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
_PUNCT_RE = re.compile(r"[,.!?;:\u3002\uff01\uff1f\uff1b]")


@dataclass(slots=True)
class _Candidate:
    markdown: str
    plain_text: str
    extractor_used: str
    quality_score: float
    warnings: list[str]
    stats: dict[str, int | float | str]


class MarkdownExtractor(ExtractorBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._profile = extract_profile_for_depth("medium")

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
            return ExtractedDocument(
                content_kind="binary",
                extractor_used="binary",
                quality_score=0.0,
                warnings=["unsupported binary content"],
            )

        html_doc = decoded[: int(self._profile.max_html_chars)]
        candidates = self._extract_html_candidates(html_doc)
        if not candidates:
            return ExtractedDocument(
                content_kind="html",
                extractor_used="none",
                quality_score=0.0,
                warnings=["html parsing produced no candidate"],
            )
        best = max(candidates, key=lambda c: float(c.quality_score))
        markdown = self._finalize_markdown(best.markdown)
        plain = self._markdown_to_plain(markdown)
        quality = self._score_markdown(markdown, plain)
        warnings = self._merge_warnings(candidates)
        if len(plain) < int(self._profile.min_plain_chars):
            warnings.append("extracted text is short")

        soup = BeautifulSoup(html_doc, "html.parser")
        title = clean_whitespace(
            html.unescape(
                soup.title.get_text(" ", strip=True) if soup.title is not None else ""
            )
        )
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title=title,
            content_kind="html",
            extractor_used=best.extractor_used,
            quality_score=float(quality),
            warnings=warnings,
            stats={
                **best.stats,
                "markdown_chars": len(markdown),
                "plain_chars": len(plain),
                "candidate_count": len(candidates),
            },
        )

    def _extract_text(self, text: str) -> ExtractedDocument:
        lines = [clean_whitespace(x) for x in text.splitlines() if clean_whitespace(x)]
        markdown = "\n\n".join(lines)
        plain = "\n".join(lines)
        quality = self._score_markdown(markdown, plain)
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title="",
            content_kind="text",
            extractor_used="plain_text",
            quality_score=float(quality),
            warnings=[],
            stats={"plain_chars": len(plain)},
        )

    def _extract_html_candidates(self, html_doc: str) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        warnings: list[str] = []

        fast = self._fastdom_candidate(html_doc)
        candidates.append(fast)

        need_fallback = fast.quality_score < float(
            self._profile.quality_threshold
        ) or len(fast.plain_text) < int(self._profile.min_plain_chars)
        if not need_fallback:
            return candidates

        fallback_jobs: list[tuple[str, Callable[[], _Candidate | None]]] = [
            ("density", lambda: self._density_candidate(html_doc))
        ]
        if bool(self._profile.readability_enabled) and READABILITY_AVAILABLE:
            fallback_jobs.append(
                ("readability", lambda: self._readability_candidate(html_doc))
            )
        else:
            warnings.append("readability unavailable")
        if bool(self._profile.trafilatura_enabled) and TRAFILATURA_AVAILABLE:
            fallback_jobs.append(
                ("trafilatura", lambda: self._trafilatura_candidate(html_doc))
            )
        else:
            warnings.append("trafilatura unavailable")

        if bool(self._profile.fallback_parallel) and len(fallback_jobs) > 1:

            def _run_job(
                item: tuple[str, Callable[[], _Candidate | None]],
            ) -> _Candidate | None:
                return item[1]()

            with ThreadPoolExecutor(max_workers=min(3, len(fallback_jobs))) as pool:
                for cand in pool.map(_run_job, fallback_jobs):
                    if cand is not None:
                        candidates.append(cand)  # noqa: PERF401
        else:
            for _, job in fallback_jobs:
                cand = job()
                if cand is not None:
                    candidates.append(cand)

        if warnings:
            candidates.append(
                _Candidate(
                    markdown=fast.markdown,
                    plain_text=fast.plain_text,
                    extractor_used=fast.extractor_used,
                    quality_score=fast.quality_score,
                    warnings=warnings,
                    stats=dict(fast.stats),
                )
            )
        return candidates

    def _fastdom_candidate(self, html_doc: str) -> _Candidate:
        soup = BeautifulSoup(html_doc, "html.parser")
        self._drop_noise(soup)
        root = self._pick_main_container(soup)
        markdown = self._to_markdown(root)
        if not markdown:
            markdown = self._to_markdown(soup)
        markdown = self._finalize_markdown(markdown)
        plain = self._markdown_to_plain(markdown)
        quality = self._score_markdown(markdown, plain)
        warnings: list[str] = []
        if len(plain) < int(self._profile.min_plain_chars):
            warnings.append("fastdom low text output")
        return _Candidate(
            markdown=markdown,
            plain_text=plain,
            extractor_used="fastdom",
            quality_score=float(quality),
            warnings=warnings,
            stats={"engine": "fastdom"},
        )

    def _density_candidate(self, html_doc: str) -> _Candidate:
        soup = BeautifulSoup(html_doc, "html.parser")
        self._drop_noise(soup)
        blocks = self._density_blocks(soup)
        markdown = "\n\n".join(blocks).strip()
        markdown = self._finalize_markdown(markdown)
        plain = self._markdown_to_plain(markdown)
        quality = self._score_markdown(markdown, plain)
        warnings: list[str] = []
        if len(plain) < int(self._profile.min_plain_chars):
            warnings.append("density fallback low text")
        return _Candidate(
            markdown=markdown,
            plain_text=plain,
            extractor_used="density",
            quality_score=float(quality),
            warnings=warnings,
            stats={"engine": "density"},
        )

    def _readability_candidate(self, html_doc: str) -> _Candidate | None:
        if not READABILITY_AVAILABLE or ReadabilityDocument is None:
            return None
        try:
            doc = ReadabilityDocument(html_doc)
            title = clean_whitespace(doc.short_title() or "")
            article_html = doc.summary() or ""
            soup = BeautifulSoup(article_html, "html.parser")
            self._drop_noise(soup)
            markdown = self._to_markdown(soup)
            if title:
                markdown = f"# {title}\n\n{markdown}".strip()
            markdown = self._finalize_markdown(markdown)
            plain = self._markdown_to_plain(markdown)
            quality = self._score_markdown(markdown, plain)
            return _Candidate(
                markdown=markdown,
                plain_text=plain,
                extractor_used="readability",
                quality_score=float(quality),
                warnings=[],
                stats={"engine": "readability"},
            )
        except Exception as exc:  # noqa: BLE001
            return _Candidate(
                markdown="",
                plain_text="",
                extractor_used="readability",
                quality_score=0.0,
                warnings=[f"readability_failed:{type(exc).__name__}"],
                stats={"engine": "readability"},
            )

    def _trafilatura_candidate(self, html_doc: str) -> _Candidate | None:
        if not TRAFILATURA_AVAILABLE or trafilatura is None:
            return None
        try:
            md = trafilatura.extract(
                html_doc,
                output_format="markdown",
                include_tables=True,
                include_links=True,
                favor_precision=True,
                favor_recall=True,
            )
            markdown = self._finalize_markdown(md or "")
            plain = self._markdown_to_plain(markdown)
            quality = self._score_markdown(markdown, plain)
            return _Candidate(
                markdown=markdown,
                plain_text=plain,
                extractor_used="trafilatura",
                quality_score=float(quality),
                warnings=[],
                stats={"engine": "trafilatura"},
            )
        except Exception as exc:  # noqa: BLE001
            return _Candidate(
                markdown="",
                plain_text="",
                extractor_used="trafilatura",
                quality_score=0.0,
                warnings=[f"trafilatura_failed:{type(exc).__name__}"],
                stats={"engine": "trafilatura"},
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
                    continue
                hidden = str(t.get("aria-hidden") or "").lower() == "true"
                style = str(t.get("style") or "").lower()
                if hidden or "display:none" in style or "visibility:hidden" in style:
                    t.decompose()
            except Exception:
                pass

    def _pick_main_container(self, soup: BeautifulSoup) -> Tag | BeautifulSoup:
        for sel in ("main", "article", '[role="main"]', "#content", "#main"):
            found = soup.select_one(sel)
            if found is not None and len(found.get_text(" ", strip=True)) >= 220:
                return found

        best: Tag | BeautifulSoup = soup
        best_score = -1.0
        candidates = soup.find_all(["article", "main", "section", "div"])
        for c in candidates:
            text = clean_whitespace(c.get_text(" ", strip=True))
            text_len = len(text)
            if text_len < 180:
                continue
            links = " ".join(a.get_text(" ", strip=True) for a in c.find_all("a"))
            link_density = len(links) / max(1, text_len)
            punct_density = len(_PUNCT_RE.findall(text)) / max(1, text_len)
            heading_count = len(c.find_all(["h1", "h2", "h3"]))
            p_count = len(c.find_all("p"))
            noise_hits = len(_NOISE_PATTERNS.findall(str(c.get("class") or "")))
            score = (
                text_len * (1.0 - min(0.92, link_density)) * (1.0 + punct_density)
                + heading_count * 65
                + p_count * 14
                - noise_hits * 140
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
        return "\n\n".join(lines).strip()

    def _density_blocks(self, root: Tag | BeautifulSoup) -> list[str]:
        scored: list[tuple[float, str]] = []
        for el in root.find_all(["p", "div", "section", "article", "li"]):
            text = clean_whitespace(el.get_text(" ", strip=True))
            if len(text) < 90:
                continue
            links = " ".join(a.get_text(" ", strip=True) for a in el.find_all("a"))
            link_density = len(links) / max(1, len(text))
            punct_density = len(_PUNCT_RE.findall(text)) / max(1, len(text))
            noise_penalty = (
                0.4 if _NOISE_PATTERNS.search(str(el.get("class") or "")) else 0.0
            )
            score = (
                (len(text) / 600.0)
                + punct_density * 2.4
                + (1.0 - min(1.0, link_density))
                - noise_penalty
            )
            if score > 0.65:
                scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        blocks: list[str] = []
        seen: set[str] = set()
        for _, text in scored[:120]:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            blocks.append(text)
        return blocks

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
        if _NOISE_LINE_RE.search(text):
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
        out = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
        out.extend("| " + " | ".join(r) + " |" for r in body)
        return "\n".join(out)

    def _finalize_markdown(self, markdown: str) -> str:
        if not markdown:
            return ""
        if len(markdown) > int(self._profile.max_markdown_chars):
            markdown = markdown[: int(self._profile.max_markdown_chars)]
        lines = [x.rstrip() for x in markdown.splitlines()]
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            normalized = _WS_RE.sub(" ", line).strip()
            if not normalized:
                if out and out[-1] != "":
                    out.append("")
                continue
            if _NOISE_LINE_RE.search(normalized):
                continue
            key = normalized.lower()
            if len(key) > 10 and key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        compact: list[str] = []
        blank = 0
        for line in out:
            if line == "":
                blank += 1
                if blank <= 1:
                    compact.append(line)
            else:
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

    def _score_markdown(self, markdown: str, plain: str) -> float:
        if not plain:
            return 0.0
        chars = len(plain)
        paragraphs = [ln for ln in markdown.split("\n\n") if ln.strip()]
        para_count = len(paragraphs)
        punct_density = len(_PUNCT_RE.findall(plain)) / max(1, chars)
        heading_count = sum(
            1 for ln in markdown.splitlines() if ln.startswith("#") and len(ln) > 2
        )
        link_mentions = len(re.findall(r"\[[^\]]+\]\([^)]+\)", markdown))
        link_density = float(link_mentions) / float(max(1, para_count))
        repeat_ratio = self._repeat_ratio(paragraphs)
        noise_hits = sum(1 for ln in paragraphs if _NOISE_LINE_RE.search(ln))
        score = (
            min(1.0, chars / 3200.0) * 0.60
            + min(1.0, para_count / 22.0) * 0.16
            + min(1.0, punct_density * 140.0) * 0.10
            + min(1.0, heading_count / 8.0) * 0.08
            + (1.0 - min(1.0, link_density / 4.0)) * 0.04
            + (1.0 - min(1.0, repeat_ratio)) * 0.02
            - min(0.25, noise_hits * 0.05)
        )
        return max(0.0, min(1.0, float(score)))

    def _repeat_ratio(self, paragraphs: list[str]) -> float:
        if not paragraphs:
            return 1.0
        normalized = [clean_whitespace(x).lower() for x in paragraphs if x.strip()]
        if not normalized:
            return 1.0
        uniq = len(set(normalized))
        return max(0.0, 1.0 - (uniq / len(normalized)))

    def _merge_warnings(self, candidates: list[_Candidate]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for cand in candidates:
            for item in cand.warnings:
                it = str(item).strip()
                if not it or it in seen:
                    continue
                seen.add(it)
                merged.append(it)
        return merged


__all__ = ["MarkdownExtractor"]
