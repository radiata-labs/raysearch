from __future__ import annotations

import contextlib
import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import fill
from typing import Literal, cast
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from bs4.element import Comment, Doctype, NavigableString, Tag

from serpsage.components.extract.markdown.dom import is_descendant_of
from serpsage.components.extract.markdown.postprocess import markdown_to_text
from serpsage.utils import clean_whitespace

RenderStatValue = int | float | bool | str
RenderStats = dict[str, RenderStatValue]
ConverterFn = Callable[[Tag | BeautifulSoup, str, set[str]], str]

_COUNT_KEYS = (
    "heading_count",
    "list_count",
    "ordered_list_count",
    "table_count",
    "table_row_count",
    "code_block_count",
    "inline_code_count",
    "link_count",
    "image_count",
    "block_count",
)

ATX = "atx"
ATX_CLOSED = "atx_closed"
UNDERLINED = "underlined"

SPACES = "spaces"
BACKSLASH = "backslash"

ASTERISK = "*"
UNDERSCORE = "_"

LSTRIP = "lstrip"
RSTRIP = "rstrip"
STRIP = "strip"
STRIP_ONE = "strip_one"

_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "caption",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}

_PSEUDO_BLOCK_NAMES = {
    "p",
    "pre",
    "table",
    "blockquote",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
}

_DATA_BLOCK_ATTR_KEYS = (
    "data-as",
    "data-tag",
    "data-type",
    "data-element",
    "data-block",
    "as",
)

_INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "code",
    "em",
    "i",
    "kbd",
    "label",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
}

_RE_HTML_HEADING = re.compile(r"h(\d+)")
_RE_MAKE_CONV_FN = re.compile(r"[\[\]:-]")
_RE_EXTRACT_NEWLINES = re.compile(r"^(\n*)((?:.*[^\n])?)(\n*)$", flags=re.DOTALL)
_RE_LINE_WITH_CONTENT = re.compile(r"^(.*)", flags=re.MULTILINE)
_RE_WHITESPACE = re.compile(r"[\t ]+")
_RE_ALL_WHITESPACE = re.compile(r"[\t \r\n]+")
_RE_NEWLINE_WHITESPACE = re.compile(r"[\t \r\n]*[\r\n][\t \r\n]*")
_RE_BACKTICK_RUNS = re.compile(r"`+")
_RE_PRE_LSTRIP1 = re.compile(r"^ *\n")
_RE_PRE_RSTRIP1 = re.compile(r"\n *$")
_RE_PRE_LSTRIP = re.compile(r"^[ \n]*\n")
_RE_PRE_RSTRIP = re.compile(r"[ \n]*$")
_RE_ESCAPE_MISC_CHARS = re.compile(r"([]\\&<`[>~=+|])")
_RE_ESCAPE_MISC_DASH_SEQS = re.compile(r"(\s|^)(-+(?:\s|$))")
_RE_ESCAPE_MISC_HASHES = re.compile(r"(\s|^)(#{1,6}(?:\s|$))")
_RE_ESCAPE_MISC_LIST_ITEMS = re.compile(r"((?:\s|^)[0-9]{1,9})([.)](?:\s|$))")

_RE_STYLE_DISPLAY = re.compile(r"display\s*:\s*([a-z-]+)", re.IGNORECASE)
_STYLE_BLOCK_DISPLAY_VALUES = {
    "block",
    "flex",
    "grid",
    "table",
    "list-item",
    "flow-root",
}
_RE_HEADING_LEVEL = re.compile(r"(?:^|[^a-z0-9])h([1-6])(?:$|[^a-z0-9])", re.IGNORECASE)
_RE_HEADING_WORD_LEVEL = re.compile(r"heading[-_\s]*([1-6])", re.IGNORECASE)
_RE_PARAGRAPH_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:p|para|paragraph)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_RE_CODE_BLOCK_TOKEN = re.compile(
    r"(?:code[-_]?block|codeblock|highlight|language-[a-z0-9_+#.-]+)",
    re.IGNORECASE,
)
_RE_BLOCKQUOTE_TOKEN = re.compile(r"(?:blockquote|quote[-_]?block)", re.IGNORECASE)
_RE_TABLE_TOKEN = re.compile(r"(?:^|[^a-z0-9])table(?:$|[^a-z0-9])", re.IGNORECASE)
_RE_HR_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:hr|divider|separator)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)

_RE_STYLE_STRONG = re.compile(r"font-weight\s*:\s*(?:bold|[6-9]00)", re.IGNORECASE)
_RE_STYLE_ITALIC = re.compile(r"font-style\s*:\s*italic", re.IGNORECASE)
_RE_STYLE_DEL = re.compile(r"text-decoration[^;]*line-through", re.IGNORECASE)
_RE_STYLE_CODE = re.compile(
    r"font-family\s*:\s*[^;]*(?:monospace|consolas|courier|menlo)",
    re.IGNORECASE,
)
_RE_CLASS_STRONG = re.compile(r"\b(?:bold|strong|fw[6-9]00|semibold)\b", re.IGNORECASE)
_RE_CLASS_ITALIC = re.compile(r"\b(?:italic|em)\b", re.IGNORECASE)
_RE_CLASS_DEL = re.compile(
    r"\b(?:line-through|strikethrough|strike|deleted)\b",
    re.IGNORECASE,
)
_RE_CLASS_CODE = re.compile(r"\b(?:code|mono|monospace)\b", re.IGNORECASE)
_RE_LANG_CLASS = re.compile(r"(?:^|\b)(?:language|lang)-([A-Za-z0-9_+#.-]+)(?:\b|$)")
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_RE_HINT_TOKEN = re.compile(r"[a-z0-9]+")
_RE_SENTENCE_PUNCT = re.compile(r"[.!?;:\u3002\uff01\uff1f\uff1b]")

_COMPACT_DIV_HINT_TOKENS = {
    "inline",
    "badge",
    "chip",
    "meta",
    "param",
    "field",
    "label",
    "value",
    "items",
    "center",
    "baseline",
    "pill",
}
_DIV_HEADING_HINT_TOKENS = {"heading", "title", "subtitle", "section", "chapter"}
_DIV_INLINE_BLOCKING_TAGS = {
    "p",
    "pre",
    "blockquote",
    "table",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
}

_RECOVER_BLOCK_RECALL_RATIO = 0.60
_RECOVER_MIN_SOURCE_CHARS = 80


@dataclass(slots=True)
class _ConverterOptions:
    autolinks: bool = True
    bullets: str = "*+-"
    code_language: str = ""
    code_language_callback: object | None = None
    convert: set[str] | None = None
    default_title: bool = False
    escape_asterisks: bool = True
    escape_underscores: bool = True
    escape_misc: bool = False
    heading_style: str = UNDERLINED
    keep_inline_images_in: set[str] | None = None
    newline_style: str = SPACES
    strip: set[str] | None = None
    strip_document: str | None = STRIP
    strip_pre: str | None = STRIP
    strong_em_symbol: str = ASTERISK
    sub_symbol: str = ""
    sup_symbol: str = ""
    table_infer_header: bool = False
    wrap: bool = False
    wrap_width: int | None = 80

    def __post_init__(self) -> None:
        if self.keep_inline_images_in is None:
            self.keep_inline_images_in = set()
        self.heading_style = str(self.heading_style or UNDERLINED).lower()
        self.newline_style = str(self.newline_style or SPACES).lower()


@dataclass(slots=True)
class _RenderMetrics:
    heading_count: int = 0
    list_count: int = 0
    ordered_list_count: int = 0
    table_count: int = 0
    table_row_count: int = 0
    code_block_count: int = 0
    inline_code_count: int = 0
    link_count: int = 0
    image_count: int = 0
    block_count: int = 0


@dataclass(slots=True)
class _RenderContext:
    base_url: str
    preserve_html_tags: bool
    options: _ConverterOptions
    metrics: _RenderMetrics


class _ManualMarkdownConverter:
    def __init__(
        self,
        *,
        base_url: str,
        preserve_html_tags: bool,
        skip_roots: list[Tag] | None,
    ) -> None:
        self.ctx = _RenderContext(
            base_url=base_url,
            preserve_html_tags=preserve_html_tags,
            options=_ConverterOptions(),
            metrics=_RenderMetrics(),
        )
        self._skip_roots = list(skip_roots or [])
        self._conv_fn_cache: dict[str, ConverterFn | None] = {}

    def convert_root(self, root: Tag | BeautifulSoup) -> str:
        text = self._process_element(root, parent_tags=set())
        if isinstance(root, BeautifulSoup):
            text = self._convert_document(text)
        return text

    def _process_element(self, node: object, *, parent_tags: set[str]) -> str:
        if isinstance(node, NavigableString):
            return self._process_text(node, parent_tags=parent_tags)
        if isinstance(node, Tag):
            return self._process_tag(node, parent_tags=parent_tags)
        if isinstance(node, BeautifulSoup):
            return self._process_tag(node, parent_tags=parent_tags)
        return ""

    def _process_tag(self, node: Tag | BeautifulSoup, *, parent_tags: set[str]) -> str:
        if isinstance(node, Tag) and self._is_skipped(node):
            return ""

        tag_name = (
            self._normalized_tag_name(node) if isinstance(node, Tag) else "[document]"
        )
        remove_inside = self._should_remove_whitespace_inside(tag_name)
        children: list[object] = [
            child
            for child in node.children
            if not self._can_ignore_child(child, remove_inside=remove_inside)
        ]

        child_tags = set(parent_tags)
        child_tags.add(tag_name)
        if _RE_HTML_HEADING.fullmatch(tag_name) is not None or tag_name in {"td", "th"}:
            child_tags.add("_inline")
        if tag_name in {"pre", "code", "kbd", "samp"}:
            child_tags.add("_noformat")

        parts = [
            self._process_element(child, parent_tags=child_tags) for child in children
        ]
        parts = [part for part in parts if part]
        if tag_name != "pre" and "pre" not in parent_tags:
            parts = self._collapse_boundary_newlines(parts)
        text = "".join(parts)

        conv_fn = self._get_conv_fn_cached(tag_name)
        if conv_fn is not None:
            text = conv_fn(node, text, parent_tags)

        if isinstance(node, Tag) and self._is_block_tag_name(tag_name) and text.strip():
            self.ctx.metrics.block_count += 1
        return text

    def _process_text(self, node: NavigableString, *, parent_tags: set[str]) -> str:
        text = _RE_ZERO_WIDTH.sub("", str(node or ""))
        if "pre" not in parent_tags:
            if self.ctx.options.wrap:
                text = _RE_ALL_WHITESPACE.sub(" ", text)
            else:
                text = _RE_NEWLINE_WHITESPACE.sub("\n", text)
                text = _RE_WHITESPACE.sub(" ", text)
        if "_noformat" not in parent_tags:
            text = self._escape(text)

        parent = node.parent if isinstance(node.parent, Tag) else None
        prev_sibling = node.previous_sibling
        next_sibling = node.next_sibling
        if self._should_remove_whitespace_outside(prev_sibling) or (
            parent is not None
            and self._should_remove_whitespace_inside(self._normalized_tag_name(parent))
            and prev_sibling is None
        ):
            text = text.lstrip(" \t\r\n")
        if self._should_remove_whitespace_outside(next_sibling) or (
            parent is not None
            and self._should_remove_whitespace_inside(self._normalized_tag_name(parent))
            and next_sibling is None
        ):
            text = text.rstrip()
        return text

    def _collapse_boundary_newlines(self, parts: list[str]) -> list[str]:
        if not parts:
            return parts
        out: list[str] = [""]
        for part in parts:
            match = _RE_EXTRACT_NEWLINES.match(part)
            if match is None:
                out.append(part)
                continue
            leading, content, trailing = match.groups()
            if out[-1] and leading:
                prev_trailing = out.pop()
                newlines = min(2, max(len(prev_trailing), len(leading)))
                leading = "\n" * newlines
            out.extend([leading, content, trailing])
        return out

    def _get_conv_fn_cached(self, tag_name: str) -> ConverterFn | None:
        if tag_name not in self._conv_fn_cache:
            self._conv_fn_cache[tag_name] = self._get_conv_fn(tag_name)
        return self._conv_fn_cache[tag_name]

    def _get_conv_fn(self, tag_name: str) -> ConverterFn | None:
        if not self._should_convert_tag(tag_name):
            return None
        conv_fn_name = f"convert_{_RE_MAKE_CONV_FN.sub('_', tag_name)}"
        maybe_conv_fn = getattr(self, conv_fn_name, None)
        if maybe_conv_fn is not None and callable(maybe_conv_fn):
            return cast("ConverterFn", maybe_conv_fn)
        match = _RE_HTML_HEADING.fullmatch(tag_name)
        if match is not None:
            n = int(match.group(1))

            def _convert_heading(
                el: Tag | BeautifulSoup, text: str, parent_tags: set[str]
            ) -> str:
                if not isinstance(el, Tag):
                    return text
                return self.convert_hN(n, el, text, parent_tags)

            return _convert_heading
        return None

    def _should_convert_tag(self, tag_name: str) -> bool:
        if self.ctx.options.strip is not None:
            return tag_name not in self.ctx.options.strip
        if self.ctx.options.convert is not None:
            return tag_name in self.ctx.options.convert
        return True

    def _escape(self, text: str) -> str:
        if not text:
            return ""
        if self.ctx.options.escape_misc:
            text = _RE_ESCAPE_MISC_CHARS.sub(r"\\\1", text)
            text = _RE_ESCAPE_MISC_DASH_SEQS.sub(r"\1\\\2", text)
            text = _RE_ESCAPE_MISC_HASHES.sub(r"\1\\\2", text)
            text = _RE_ESCAPE_MISC_LIST_ITEMS.sub(r"\1\\\2", text)
        if self.ctx.options.escape_asterisks:
            text = text.replace("*", r"\*")
        if self.ctx.options.escape_underscores:
            text = text.replace("_", r"\_")
        return text

    def _convert_document(self, text: str) -> str:
        style = self.ctx.options.strip_document
        if style == LSTRIP:
            return text.lstrip("\n")
        if style == RSTRIP:
            return text.rstrip("\n")
        if style == STRIP:
            return text.strip("\n")
        return text

    def _can_ignore_child(self, child: object, *, remove_inside: bool) -> bool:
        if isinstance(child, Tag):
            return False
        if isinstance(child, (Comment, Doctype)):
            return True
        if isinstance(child, NavigableString):
            if str(child).strip() != "":
                return False
            if remove_inside and (
                child.previous_sibling is None or child.next_sibling is None
            ):
                return True
            return bool(
                self._should_remove_whitespace_outside(child.previous_sibling)
                or self._should_remove_whitespace_outside(child.next_sibling)
            )
        return True

    def _should_remove_whitespace_inside(self, tag_name: str) -> bool:
        if tag_name == "pre":
            return False
        if _RE_HTML_HEADING.fullmatch(tag_name) is not None:
            return True
        return tag_name in _BLOCK_TAGS

    def _should_remove_whitespace_outside(self, node: object) -> bool:
        if not isinstance(node, Tag):
            return False
        tag_name = self._normalized_tag_name(node)
        return self._should_remove_whitespace_inside(tag_name) or tag_name == "pre"

    def _is_skipped(self, tag: Tag) -> bool:
        return any(
            tag is skip or is_descendant_of(tag, skip) for skip in self._skip_roots
        )

    def _normalized_tag_name(self, tag: Tag) -> str:
        raw = (tag.name or "").lower()
        if raw in _BLOCK_TAGS:
            return raw
        inferred = (
            self._infer_from_role(tag)
            or self._infer_from_data_attrs(tag)
            or self._infer_from_identifier(tag)
            or self._infer_from_style(tag)
        )
        if inferred and self._is_viable_pseudo_block(tag, inferred):
            return inferred
        return raw

    def _infer_from_role(self, tag: Tag) -> str | None:
        role = str(tag.get("role") or "").strip().lower()
        if role == "heading":
            return f"h{self._heading_level(tag)}"
        if role in {"paragraph", "doc-paragraph"}:
            return "p"
        return None

    def _infer_from_data_attrs(self, tag: Tag) -> str | None:
        for key in _DATA_BLOCK_ATTR_KEYS:
            value = str(tag.get(key) or "").strip().lower()
            inferred = self._infer_from_text(value, tag)
            if inferred:
                return inferred
        return None

    def _infer_from_identifier(self, tag: Tag) -> str | None:
        ident = (
            " ".join(
                [
                    str(tag.get("id") or ""),
                    " ".join(str(item) for item in (tag.get("class") or [])),
                    str(tag.get("data-testid") or ""),
                ]
            )
            .strip()
            .lower()
        )
        if not ident:
            return None
        inferred = self._infer_from_text(ident, tag)
        if inferred in {"pre", "table", "blockquote", "hr"}:
            return inferred
        raw = (tag.name or "").lower()
        if inferred in _PSEUDO_BLOCK_NAMES and raw in _INLINE_TAGS:
            return inferred
        return None

    def _infer_from_style(self, tag: Tag) -> str | None:
        style = str(tag.get("style") or "")
        match = _RE_STYLE_DISPLAY.search(style)
        if match is None:
            return None
        display = match.group(1).strip().lower()
        if display in {
            "inline",
            "inline-block",
            "inline-flex",
            "inline-grid",
            "contents",
        }:
            return None
        if (
            display in _STYLE_BLOCK_DISPLAY_VALUES
            and (tag.name or "").lower() in _INLINE_TAGS
        ):
            return "p"
        return None

    def _infer_from_text(self, value: str, tag: Tag) -> str | None:
        if not value:
            return None
        heading = self._heading_from_text(value, tag)
        if heading:
            return heading
        if _RE_PARAGRAPH_TOKEN.search(value):
            return "p"
        if _RE_CODE_BLOCK_TOKEN.search(value):
            return "pre"
        if _RE_BLOCKQUOTE_TOKEN.search(value):
            return "blockquote"
        if _RE_TABLE_TOKEN.search(value) and "table-of-contents" not in value:
            return "table"
        if _RE_HR_TOKEN.search(value):
            return "hr"
        return None

    def _heading_from_text(self, value: str, tag: Tag) -> str | None:
        direct = _RE_HEADING_LEVEL.search(value)
        if direct:
            return f"h{direct.group(1)}"
        word = _RE_HEADING_WORD_LEVEL.search(value)
        if word:
            return f"h{word.group(1)}"
        if "heading" in value:
            return f"h{self._heading_level(tag)}"
        return None

    def _heading_level(self, tag: Tag) -> int:
        for key in ("aria-level", "data-level"):
            raw = str(tag.get(key) or "").strip()
            if raw.isdigit():
                value = int(raw)
                if 1 <= value <= 6:
                    return value
        return 2

    def _is_viable_pseudo_block(self, tag: Tag, name: str) -> bool:
        if name not in _PSEUDO_BLOCK_NAMES:
            return False
        if not clean_whitespace(tag.get_text(" ", strip=True)):
            return False
        if self._has_explicit_inline_display(tag):
            return False
        return not self._contains_nested_block(tag)

    def _has_explicit_inline_display(self, tag: Tag) -> bool:
        style = str(tag.get("style") or "")
        match = _RE_STYLE_DISPLAY.search(style)
        if match is None:
            return False
        display = match.group(1).strip().lower()
        return display in {
            "inline",
            "inline-block",
            "inline-flex",
            "inline-grid",
            "contents",
        }

    def _contains_nested_block(self, tag: Tag) -> bool:
        for node in tag.descendants:
            if not isinstance(node, Tag) or node is tag:
                continue
            raw = (node.name or "").lower()
            if raw in _BLOCK_TAGS:
                return True
            inferred = self._infer_from_role(node) or self._infer_from_data_attrs(node)
            if inferred in _PSEUDO_BLOCK_NAMES:
                return True
        return False

    def _is_block_tag_name(self, tag_name: str) -> bool:
        return (
            tag_name in _BLOCK_TAGS or _RE_HTML_HEADING.fullmatch(tag_name) is not None
        )

    @staticmethod
    def _chomp(text: str) -> tuple[str, str, str]:
        prefix = " " if text[:1] == " " else ""
        suffix = " " if text[-1:] == " " else ""
        return prefix, suffix, text.strip()

    def _normalize_anchor_label(self, text: str) -> str:
        return clean_whitespace(_RE_ZERO_WIDTH.sub("", text or ""))

    @staticmethod
    def _md_escape_label(text: str) -> str:
        return text.replace("[", r"\[").replace("]", r"\]")

    def _safe_join(self, href: str) -> str | None:
        candidate = (href or "").strip()
        if not candidate:
            return None
        if candidate.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
            return None
        try:
            return str(urljoin(self.ctx.base_url, candidate))
        except Exception:
            return None

    def _is_same_page_fragment(self, href: str) -> bool:
        parsed = urlparse(href)
        if not parsed.fragment:
            return False
        base = urlparse(self.ctx.base_url)
        return (
            parsed.scheme == base.scheme
            and parsed.netloc == base.netloc
            and parsed.path == base.path
            and parsed.params == base.params
            and parsed.query == base.query
        )

    @staticmethod
    def _hint_tokens(value: str) -> set[str]:
        return {match.group(0) for match in _RE_HINT_TOKEN.finditer(str(value).lower())}

    def _has_compact_div_hint(self, el: Tag) -> bool:
        hints = " ".join(
            [
                str(el.get("id") or ""),
                " ".join(str(item) for item in (el.get("class") or [])),
                str(el.get("data-component-part") or ""),
                str(el.get("data-testid") or ""),
                str(el.get("data-element") or ""),
                str(el.get("data-tag") or ""),
            ]
        )
        tokens = self._hint_tokens(hints)
        if not tokens:
            return False
        if tokens & _DIV_HEADING_HINT_TOKENS:
            return False
        return bool(tokens & _COMPACT_DIV_HINT_TOKENS)

    @staticmethod
    def _has_inline_blocking_descendants(el: Tag) -> bool:
        return any(
            isinstance(node, Tag)
            and node is not el
            and (node.name or "").lower() in _DIV_INLINE_BLOCKING_TAGS
            for node in el.descendants
        )

    def _should_render_div_inline(self, el: Tag) -> bool:
        source_text = clean_whitespace(el.get_text(" ", strip=True))
        if not source_text:
            return False
        if len(source_text) > 72:
            return False
        if len(source_text.split()) > 12:
            return False
        if _RE_SENTENCE_PUNCT.search(source_text):
            return False
        if not self._has_compact_div_hint(el):
            return False
        return not self._has_inline_blocking_descendants(el)

    def _first_image_src(self, el: Tag) -> str:
        candidates: list[str] = []
        for key in ("src", "data-src"):
            value = str(el.get(key) or "").strip()
            if value:
                candidates.append(value)
        for key in ("srcset", "data-srcset"):
            value = str(el.get(key) or "").strip()
            if not value:
                continue
            for item in value.split(","):
                src = item.strip().split(" ", 1)[0].strip()
                if src:
                    candidates.append(src)
        if not candidates and (el.name or "").lower() == "video":
            for source in el.find_all("source"):
                src = str(source.get("src") or "").strip()
                if src:
                    candidates.append(src)
        return candidates[0] if candidates else ""

    def _detect_code_language(self, pre: Tag, code: Tag | None) -> str:
        for node in [code, pre]:
            if not isinstance(node, Tag):
                continue
            data_lang = str(node.get("data-lang") or node.get("lang") or "").strip()
            if data_lang:
                return self._sanitize_info(data_lang)
            for cls in node.get("class") or []:
                match = _RE_LANG_CLASS.search(str(cls))
                if match:
                    return self._sanitize_info(match.group(1))
        return ""

    @staticmethod
    def _sanitize_info(value: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_+#.-]", "", value.strip().lower())
        return clean[:32]

    @staticmethod
    def _html_attrs(tag: Tag, keys: tuple[str, ...]) -> str:
        attrs: list[str] = []
        for key in keys:
            value = tag.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                payload = " ".join(
                    str(item).strip() for item in value if str(item).strip()
                )
            else:
                payload = str(value).strip()
            if payload:
                attrs.append(f'{key}="{html.escape(payload, quote=True)}"')
        return f" {' '.join(attrs)}" if attrs else ""

    # converters
    def convert__document_(self, el: object, text: str, parent_tags: set[str]) -> str:
        return self._convert_document(text)

    def convert_script(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return ""

    def convert_style(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return ""

    def convert_a(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_noformat" in parent_tags:
            return text
        prefix, suffix, body = self._chomp(text)
        raw_href = str(el.get("href") or "").strip()
        if not body:
            body = self._normalize_anchor_label(el.get_text(" ", strip=True))
        href = self._safe_join(raw_href)
        if not body and raw_href.startswith("#"):
            return ""
        if not body and href and self._is_same_page_fragment(href):
            return ""
        title = str(el.get("title") or "").strip() or None
        if not body and href:
            body = href
        if not body:
            return ""
        if not href:
            return f"{prefix}{body}{suffix}"

        self.ctx.metrics.link_count += 1
        if self.ctx.preserve_html_tags:
            title_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return f'{prefix}<a href="{html.escape(href, quote=True)}"{title_attr}>{body}</a>{suffix}'

        normalized = body.replace(r"\_", "_")
        if (
            self.ctx.options.autolinks
            and normalized == href
            and not title
            and not self.ctx.options.default_title
        ):
            if self._is_same_page_fragment(href):
                return ""
            return f"{prefix}<{href}>{suffix}"
        if self.ctx.options.default_title and not title:
            title = href
        escaped_title = title.replace(chr(34), r"\\\"") if title else ""
        title_part = f' "{escaped_title}"' if title else ""
        return f"{prefix}[{self._md_escape_label(body)}]({href}{title_part}){suffix}"

    def convert_br(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_inline" in parent_tags:
            return " "
        if self.ctx.preserve_html_tags:
            return "<br />\n"
        return "\\\n" if self.ctx.options.newline_style == BACKSLASH else "  \n"

    def convert_code(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_noformat" in parent_tags:
            return text
        prefix, suffix, body = self._chomp(text)
        if not body:
            return ""
        self.ctx.metrics.inline_code_count += 1
        if self.ctx.preserve_html_tags:
            return f"{prefix}<code>{html.escape(body.replace(chr(10), ' '))}</code>{suffix}"
        max_ticks = max(
            (len(run) for run in re.findall(_RE_BACKTICK_RUNS, body)), default=0
        )
        delim = "`" * (max_ticks + 1)
        if max_ticks > 0:
            body = f" {body} "
        return f"{prefix}{delim}{body}{delim}{suffix}"

    def convert_kbd(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_code(el, text, parent_tags)

    def convert_samp(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_code(el, text, parent_tags)

    def _inline_wrap(self, text: str, markup: str, *, parent_tags: set[str]) -> str:
        if "_noformat" in parent_tags:
            return text
        prefix, suffix, body = self._chomp(text)
        if not body:
            return ""
        if markup.startswith("<") and markup.endswith(">"):
            end = "</" + markup[1:]
        else:
            end = markup
        return f"{prefix}{markup}{body}{end}{suffix}"

    def convert_b(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        markup = (
            "<strong>"
            if self.ctx.preserve_html_tags
            else self.ctx.options.strong_em_symbol * 2
        )
        return self._inline_wrap(text, markup, parent_tags=parent_tags)

    def convert_strong(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_b(el, text, parent_tags)

    def convert_em(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        markup = (
            "<em>" if self.ctx.preserve_html_tags else self.ctx.options.strong_em_symbol
        )
        return self._inline_wrap(text, markup, parent_tags=parent_tags)

    def convert_i(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_em(el, text, parent_tags)

    def convert_del(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        markup = "<del>" if self.ctx.preserve_html_tags else "~~"
        return self._inline_wrap(text, markup, parent_tags=parent_tags)

    def convert_s(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_del(el, text, parent_tags)

    def convert_strike(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_del(el, text, parent_tags)

    def convert_sub(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if self.ctx.preserve_html_tags:
            return self._inline_wrap(text, "<sub>", parent_tags=parent_tags)
        symbol = self.ctx.options.sub_symbol
        return (
            self._inline_wrap(text, symbol, parent_tags=parent_tags) if symbol else text
        )

    def convert_sup(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if self.ctx.preserve_html_tags:
            return self._inline_wrap(text, "<sup>", parent_tags=parent_tags)
        symbol = self.ctx.options.sup_symbol
        return (
            self._inline_wrap(text, symbol, parent_tags=parent_tags) if symbol else text
        )

    def convert_q(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return f'"{text}"'

    def convert_span(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if not text:
            return ""
        classes = " ".join(str(item) for item in (el.get("class") or []))
        style = str(el.get("style") or "")
        as_code = bool(_RE_CLASS_CODE.search(classes) or _RE_STYLE_CODE.search(style))
        as_strong = bool(
            _RE_CLASS_STRONG.search(classes) or _RE_STYLE_STRONG.search(style)
        )
        as_em = bool(_RE_CLASS_ITALIC.search(classes) or _RE_STYLE_ITALIC.search(style))
        as_del = bool(_RE_CLASS_DEL.search(classes) or _RE_STYLE_DEL.search(style))

        out = text
        if as_code:
            out = self.convert_code(el, out, parent_tags)
        if as_strong:
            out = self.convert_b(el, out, parent_tags)
        if as_em:
            out = self.convert_em(el, out, parent_tags)
        if as_del:
            out = self.convert_del(el, out, parent_tags)

        if self.ctx.preserve_html_tags and not (
            as_code or as_strong or as_em or as_del
        ):
            attrs = self._html_attrs(el, ("class", "style"))
            return f"<span{attrs}>{text}</span>"
        return out

    def convert_p(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_inline" in parent_tags:
            return " " + text.strip(" \t\r\n") + " "
        body = text.strip(" \t\r\n")
        if (
            self.ctx.options.wrap
            and self.ctx.options.wrap_width is not None
            and not self.ctx.preserve_html_tags
        ):
            lines = body.split("\n")
            wrapped: list[str] = []
            for line in lines:
                line = line.lstrip(" \t\r\n")
                stripped = line.rstrip()
                trailing = line[len(stripped) :]
                wrapped.append(
                    fill(
                        line,
                        width=self.ctx.options.wrap_width,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    + trailing
                )
            body = "\n".join(wrapped)
        if not body:
            return ""
        if self.ctx.preserve_html_tags:
            return f"<p>{body}</p>"
        return f"\n\n{body}\n\n"

    def convert_div(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_inline" in parent_tags:
            return " " + text.strip() + " "
        body = text.strip()
        if not body:
            return ""
        if self._should_render_div_inline(el):
            return " " + clean_whitespace(body) + " "
        if self.ctx.preserve_html_tags:
            return f"<div>{body}</div>"
        return f"\n\n{body}\n\n"

    def convert_article(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_section(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_blockquote(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip(" \t\r\n")
        if "_inline" in parent_tags:
            return f" {body} "
        if not body:
            return "\n"
        if self.ctx.preserve_html_tags:
            return f"\n<blockquote>{body}</blockquote>\n\n"

        def _indent(match: re.Match[str]) -> str:
            line = match.group(1)
            return "> " + line if line else ">"

        body = _RE_LINE_WITH_CONTENT.sub(_indent, body)
        return "\n" + body + "\n\n"

    def convert_hN(self, n: int, el: Tag, text: str, parent_tags: set[str]) -> str:
        if "_inline" in parent_tags:
            return text
        n = max(1, min(6, n))
        body = clean_whitespace(text)
        if not body:
            return ""
        self.ctx.metrics.heading_count += 1
        if self.ctx.preserve_html_tags:
            return f"\n\n<h{n}>{body}</h{n}>\n\n"
        style = self.ctx.options.heading_style
        if style == UNDERLINED and n <= 2:
            line = "=" if n == 1 else "-"
            return f"\n\n{body}\n{line * len(body)}\n\n"
        hashes = "#" * n
        if style == ATX_CLOSED:
            return f"\n\n{hashes} {body} {hashes}\n\n"
        return f"\n\n{hashes} {body}\n\n"

    def convert_hr(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if self.ctx.preserve_html_tags:
            return "\n\n<hr />\n\n"
        return "\n\n---\n\n"

    def convert_pre(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if not text:
            return ""
        style = self.ctx.options.strip_pre
        if style == STRIP:
            text = _RE_PRE_LSTRIP.sub("", text)
            text = _RE_PRE_RSTRIP.sub("", text)
        elif style == STRIP_ONE:
            text = _RE_PRE_LSTRIP1.sub("", text)
            text = _RE_PRE_RSTRIP1.sub("", text)
        elif style == LSTRIP:
            text = text.lstrip("\n")
        elif style == RSTRIP:
            text = text.rstrip("\n")
        elif style is not None:
            raise ValueError(f"Invalid strip_pre: {style}")

        code = el.find("code")
        lang = (
            self._detect_code_language(el, code if isinstance(code, Tag) else None)
            or self.ctx.options.code_language
        )
        callback = self.ctx.options.code_language_callback
        if callable(callback):
            with contextlib.suppress(Exception):
                lang = str(callback(el) or lang)

        self.ctx.metrics.code_block_count += 1
        if self.ctx.preserve_html_tags:
            info_attr = f' class="language-{html.escape(lang)}"' if lang else ""
            return f"\n\n<pre><code{info_attr}>{html.escape(text)}</code></pre>\n\n"
        return f"\n\n```{lang}\n{text}\n```\n\n"

    def convert_ul(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self._convert_list(el, text, parent_tags)

    def convert_ol(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self._convert_list(el, text, parent_tags)

    def _convert_list(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        next_sibling = self._next_block_content_sibling(el)
        before_paragraph = isinstance(next_sibling, Tag) and (
            next_sibling.name or ""
        ).lower() not in {"ul", "ol"}
        if "li" in parent_tags:
            return "\n" + text.rstrip()
        return "\n\n" + text + ("\n" if before_paragraph else "")

    def convert_li(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if not body:
            return "\n"
        parent = el.parent if isinstance(el.parent, Tag) else None
        ordered = bool(parent is not None and (parent.name or "").lower() == "ol")
        if ordered:
            start = 1
            raw_start = (
                str(parent.get("start") or "").strip() if parent is not None else ""
            )
            if raw_start.isdigit():
                start = int(raw_start)
            bullet = f"{start + len(el.find_previous_siblings('li'))}. "
            self.ctx.metrics.ordered_list_count += 1
        else:
            depth = -1
            node: object = el
            while isinstance(node, Tag):
                if (node.name or "").lower() == "ul":
                    depth += 1
                node = node.parent
            bullets = self.ctx.options.bullets or "*"
            bullet = bullets[depth % len(bullets)] + " "

        self.ctx.metrics.list_count += 1
        width = len(bullet)
        indent = " " * width

        def _indent(match: re.Match[str]) -> str:
            line = match.group(1)
            return indent + line if line else ""

        body = _RE_LINE_WITH_CONTENT.sub(_indent, body)
        body = bullet + body[width:]
        return body + "\n"

    def convert_dl(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_div(el, text, parent_tags)

    def convert_dt(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = _RE_ALL_WHITESPACE.sub(" ", text.strip())
        if "_inline" in parent_tags:
            return f" {body} "
        if not body:
            return "\n"
        return f"\n\n{body}\n"

    def convert_dd(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if "_inline" in parent_tags:
            return f" {body} "
        if not body:
            return "\n"

        def _indent(match: re.Match[str]) -> str:
            line = match.group(1)
            return "    " + line if line else ""

        body = _RE_LINE_WITH_CONTENT.sub(_indent, body)
        body = ":" + body[1:]
        return body + "\n"

    def convert_table(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if not body:
            return ""
        self.ctx.metrics.table_count += 1
        if self.ctx.preserve_html_tags:
            return f"\n\n<table>{body}</table>\n\n"
        return f"\n\n{body}\n\n"

    def convert_caption(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        return body + "\n\n" if body else ""

    def convert_figcaption(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if not body:
            return ""
        if self.ctx.preserve_html_tags:
            return f"\n\n<figcaption>{body}</figcaption>\n\n"
        return f"\n\n{body}\n\n"

    def convert_td(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip().replace("\n", " ")
        colspan = 1
        raw_colspan = str(el.get("colspan") or "").strip()
        if raw_colspan.isdigit():
            colspan = max(1, min(1000, int(raw_colspan)))
        return " " + body + " |" * colspan

    def convert_th(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        return self.convert_td(el, text, parent_tags)

    def convert_tr(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        cells = el.find_all(["td", "th"])
        is_first_row = el.find_previous_sibling() is None
        is_headrow = all((cell.name or "").lower() == "th" for cell in cells) or (
            isinstance(el.parent, Tag)
            and (el.parent.name or "").lower() == "thead"
            and len(el.parent.find_all("tr")) == 1
        )
        full_colspan = 0
        for cell in cells:
            raw_colspan = str(cell.get("colspan") or "").strip()
            full_colspan += (
                max(1, min(1000, int(raw_colspan))) if raw_colspan.isdigit() else 1
            )

        overline = ""
        underline = ""
        is_head_row_missing = is_first_row and not is_headrow
        if (
            is_headrow or (is_head_row_missing and self.ctx.options.table_infer_header)
        ) and is_first_row:
            underline = "| " + " | ".join(["---"] * full_colspan) + " |\n"
        elif is_head_row_missing and not self.ctx.options.table_infer_header:
            overline = "| " + " | ".join([""] * full_colspan) + " |\n"
            overline += "| " + " | ".join(["---"] * full_colspan) + " |\n"

        if any((cell.name or "").lower() == "td" for cell in cells):
            self.ctx.metrics.table_row_count += 1
        return overline + "|" + text + "\n" + underline

    def convert_img(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        alt = clean_whitespace(str(el.get("alt") or ""))
        src = self._safe_join(self._first_image_src(el))
        title = str(el.get("title") or "").strip()
        parent_name = (
            (el.parent.name or "").lower() if isinstance(el.parent, Tag) else ""
        )
        if "_inline" in parent_tags and parent_name not in (
            self.ctx.options.keep_inline_images_in or set()
        ):
            return alt
        if not src:
            return alt
        self.ctx.metrics.image_count += 1
        if self.ctx.preserve_html_tags:
            title_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return (
                f'<img src="{html.escape(src, quote=True)}" '
                f'alt="{html.escape(alt, quote=True)}"{title_attr} />'
            )
        escaped_title = title.replace(chr(34), r"\\\"") if title else ""
        title_part = f' "{escaped_title}"' if title else ""
        return f"![{self._md_escape_label(alt)}]({src}{title_part})"

    def convert_picture(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        if text.strip():
            return text
        img = el.find("img")
        if isinstance(img, Tag):
            return self.convert_img(img, "", parent_tags)
        return ""

    def convert_video(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        src = self._safe_join(self._first_image_src(el))
        poster = self._safe_join(str(el.get("poster") or ""))
        body = clean_whitespace(text)
        if "_inline" in parent_tags:
            return body
        if self.ctx.preserve_html_tags:
            attrs: list[str] = []
            if src:
                attrs.append(f'src="{html.escape(src, quote=True)}"')
            if poster:
                attrs.append(f'poster="{html.escape(poster, quote=True)}"')
            attrs_part = (" " + " ".join(attrs)) if attrs else ""
            return f"<video{attrs_part}>{body}</video>"
        if src and poster:
            return f"[![{self._md_escape_label(body)}]({poster})]({src})"
        if src:
            return f"[{self._md_escape_label(body)}]({src})"
        if poster:
            return f"![{self._md_escape_label(body)}]({poster})"
        return body

    def convert_details(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if not body:
            return ""
        if self.ctx.preserve_html_tags:
            return f"\n<details>{body}</details>\n"
        return f"\n\n{body}\n\n"

    def convert_summary(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        body = text.strip()
        if not body:
            return ""
        if self.ctx.preserve_html_tags:
            return f"<summary>{body}</summary>"
        return f"**{body}**"

    @staticmethod
    def _is_content_element(node: object) -> bool:
        if isinstance(node, Tag):
            return True
        if isinstance(node, (Comment, Doctype)):
            return False
        if isinstance(node, NavigableString):
            return str(node).strip() != ""
        return False

    def _next_block_content_sibling(self, el: Tag) -> Tag | NavigableString | None:
        node: object | None = el
        while isinstance(node, Tag):
            node = node.next_sibling
            if self._is_content_element(node):
                return node  # type: ignore[return-value]
        return None


def markdownify_available() -> bool:
    # compatibility API: internal renderer is always available
    return True


def render_markdown(
    *,
    root: Tag | BeautifulSoup,
    base_url: str,
    skip_roots: list[Tag] | None = None,
    preserve_html_tags: bool = False,
) -> tuple[str, RenderStats]:
    converter = _ManualMarkdownConverter(
        base_url=base_url,
        preserve_html_tags=preserve_html_tags,
        skip_roots=skip_roots,
    )
    fallback_used = False
    fallback_reasons: list[str] = []
    source_text = _source_text(root)

    try:
        markdown = _normalize_fragment(converter.convert_root(root))
    except Exception as exc:  # noqa: BLE001
        markdown = _recover_markdown_from_text(
            source_text, preserve_html_tags=preserve_html_tags
        )
        fallback_used = True
        fallback_reasons.append(f"document_exception:{type(exc).__name__}")

    markdown = _recover_empty_doc(
        markdown=markdown,
        source_text=source_text,
        preserve_html_tags=preserve_html_tags,
        fallback_reasons=fallback_reasons,
    )
    if markdown and _should_recover_doc(source_text=source_text, markdown=markdown):
        recovered = _recover_merge_markdown(
            markdown=markdown,
            source_text=source_text,
            preserve_html_tags=preserve_html_tags,
        )
        if recovered != markdown:
            markdown = recovered
            fallback_used = True
            fallback_reasons.append("document_low_recall")

    stats: RenderStats = {
        "renderer_backend": "markdownify",
        "renderer_fallback_used": bool(fallback_used),
        "renderer_text_recall_ratio": float(
            _text_recall_ratio(source_text=source_text, markdown=markdown)
        ),
    }
    if fallback_reasons:
        stats["renderer_fallback_reason"] = ",".join(sorted(set(fallback_reasons)))
    return markdown, stats


def render_secondary_markdown(
    *,
    secondary_roots: list[Tag],
    base_url: str,
    preserve_html_tags: bool = False,
) -> tuple[str, RenderStats]:
    blocks: list[str] = []
    merged = _empty_stats()
    ratio_sum = 0.0
    ratio_count = 0
    fallback_reasons: list[str] = []

    for root in secondary_roots:
        markdown, stats = render_markdown(
            root=root,
            base_url=base_url,
            preserve_html_tags=preserve_html_tags,
        )
        if markdown:
            blocks.append(markdown)
        for key in _COUNT_KEYS:
            merged[key] = int(merged.get(key, 0)) + int(stats.get(key, 0))
        merged["renderer_fallback_used"] = bool(
            bool(merged.get("renderer_fallback_used", False))
            or bool(stats.get("renderer_fallback_used", False))
        )
        ratio_sum += float(stats.get("renderer_text_recall_ratio", 0.0))
        ratio_count += 1
        reason = str(stats.get("renderer_fallback_reason", "")).strip()
        if reason:
            fallback_reasons.append(reason)

    merged["renderer_backend"] = "markdownify"
    merged["renderer_text_recall_ratio"] = (
        float(ratio_sum / float(ratio_count)) if ratio_count > 0 else 1.0
    )
    if fallback_reasons:
        merged["renderer_fallback_reason"] = ",".join(sorted(set(fallback_reasons)))

    return _normalize_fragment("\n\n".join(blocks).strip()), merged


def _empty_stats() -> RenderStats:
    return {
        "heading_count": 0,
        "list_count": 0,
        "ordered_list_count": 0,
        "table_count": 0,
        "table_row_count": 0,
        "code_block_count": 0,
        "inline_code_count": 0,
        "link_count": 0,
        "image_count": 0,
        "block_count": 0,
        "renderer_backend": "markdownify",
        "renderer_fallback_used": False,
        "renderer_text_recall_ratio": 1.0,
    }


def _normalize_fragment(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ""
    lines = [line.rstrip() for line in normalized.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(line)
    return "\n".join(out).strip()


def _source_text(root: Tag | BeautifulSoup) -> str:
    return clean_whitespace(root.get_text(" ", strip=True))


def _text_recall_ratio(*, source_text: str, markdown: str) -> float:
    source_chars = len(source_text)
    if source_chars <= 0:
        return 1.0
    plain = markdown_to_text(markdown)
    if not plain:
        return 0.0
    ratio = float(len(plain)) / float(source_chars)
    return max(0.0, min(1.0, ratio))


def _should_recover_doc(*, source_text: str, markdown: str) -> bool:
    if len(source_text) < _RECOVER_MIN_SOURCE_CHARS:
        return False
    recall = _text_recall_ratio(source_text=source_text, markdown=markdown)
    return bool(recall < _RECOVER_BLOCK_RECALL_RATIO)


def _recover_markdown_from_text(source_text: str, *, preserve_html_tags: bool) -> str:
    text = clean_whitespace(source_text)
    if not text:
        return ""
    if preserve_html_tags:
        return f"<p>{html.escape(text)}</p>"
    return text


def _recover_empty_doc(
    *,
    markdown: str,
    source_text: str,
    preserve_html_tags: bool,
    fallback_reasons: list[str],
) -> str:
    if markdown.strip() or len(source_text) < _RECOVER_MIN_SOURCE_CHARS:
        return markdown
    fallback_reasons.append("document_empty")
    return _recover_markdown_from_text(
        source_text, preserve_html_tags=preserve_html_tags
    )


def _recover_merge_markdown(
    *,
    markdown: str,
    source_text: str,
    preserve_html_tags: bool,
) -> str:
    source_clean = clean_whitespace(source_text)
    if not source_clean:
        return markdown
    plain = clean_whitespace(markdown_to_text(markdown))
    if plain and len(plain) >= int(len(source_clean) * 0.95):
        return markdown
    recovered = _recover_markdown_from_text(
        source_clean, preserve_html_tags=preserve_html_tags
    )
    if not recovered:
        return markdown
    if not markdown.strip():
        return recovered
    if plain and len(plain) >= 320:
        return markdown
    return f"{markdown.strip()}\n\n{recovered}"


BackendName = Literal["markdownify"]


__all__ = [
    "BackendName",
    "RenderStats",
    "RenderStatValue",
    "markdownify_available",
    "render_markdown",
    "render_secondary_markdown",
]
