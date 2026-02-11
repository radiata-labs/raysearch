from __future__ import annotations

import html as html_mod
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.contracts.services import ExtractorBase
from serpsage.models.extract import ExtractedText
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class BasicHtmlExtractor(ExtractorBase):
    """Lightweight visible-text extraction.

    - best-effort decoding (charset header/meta/BOM + heuristics)
    - HTMLParser-based visible text extraction
    - block segmentation by newlines
    """

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

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
            visible = text
        else:
            parser = VisibleTextParser()
            try:
                parser.feed(text)
                parser.close()
            except Exception:
                # best effort: treat as plain text
                visible = text
            else:
                visible = html_mod.unescape(parser.get_text())

        max_chars = int(self.settings.enrich.fetch.max_extracted_chars)
        if max_chars and len(visible) > max_chars:
            visible = visible[:max_chars]

        visible = visible.replace("\r\n", "\n").replace("\r", "\n")
        lines = [clean_whitespace(line) for line in visible.split("\n")]
        blocks = [line for line in lines if line]

        return ExtractedText(text="\n".join(blocks), blocks=blocks)


_SKIP_TAGS = {"script", "style", "noscript"}
_BLOCK_TAGS = {"p", "br", "div", "li", "section", "article", "h1", "h2", "h3", "h4"}
_NOISE_ID_CLASS_RE = re.compile(
    r"(nav|footer|header|sidebar|breadcrumb|menu|toc|comment|ads|cookie|banner)",
    re.IGNORECASE,
)


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._buf: list[str] = []
        self._skip_stack: list[str] = []
        # Track "noise" elements (nav/footer/etc, plus common id/class patterns).
        # Use a stack so we can reliably exit the noise region on the matching end tag.
        self._noise_stack: list[str] = []

    @override
    def handle_starttag(self, tag: str, attrs: Any) -> None:  # noqa: ANN401
        lowered = (tag or "").lower()
        if lowered in _SKIP_TAGS:
            self._skip_stack.append(lowered)
            return

        if lowered in {"nav", "footer", "header", "aside"}:
            self._noise_stack.append(lowered)
            return

        if attrs:
            for k, v in attrs:
                if (
                    k in {"id", "class"}
                    and isinstance(v, str)
                    and _NOISE_ID_CLASS_RE.search(v)
                ):
                    self._noise_stack.append(lowered)
                    break

    @override
    def handle_endtag(self, tag: str) -> None:
        lowered = (tag or "").lower()
        if self._skip_stack and self._skip_stack[-1] == lowered:
            self._skip_stack.pop()
            return

        if self._noise_stack and self._noise_stack[-1] == lowered:
            self._noise_stack.pop()
            return

        if lowered in _BLOCK_TAGS:
            self._buf.append("\n")

    @override
    def handle_data(self, data: str) -> None:
        if self._skip_stack or self._noise_stack:
            return
        if data:
            self._buf.append(data)

    @override
    def handle_entityref(self, name: str) -> None:
        if self._skip_stack or self._noise_stack:
            return
        self._buf.append(f"&{name};")

    @override
    def handle_charref(self, name: str) -> None:
        if self._skip_stack or self._noise_stack:
            return
        self._buf.append(f"&#{name};")

    def get_text(self) -> str:
        return "".join(self._buf)


__all__ = ["BasicHtmlExtractor"]
