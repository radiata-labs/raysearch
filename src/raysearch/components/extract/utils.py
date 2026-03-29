from __future__ import annotations

import codecs
import re
from contextlib import suppress
from typing import Literal

from raysearch.utils import clean_whitespace

_MD_PREFIX_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+[.)]\s+|#{1,6}\s+|>\s+)")
_NOISE_LINE_RE = re.compile(
    r"(privacy policy|cookie policy|terms of service|all rights reserved|"
    r"sign up|subscribe|advertisement|sponsored content|related posts)",
    re.IGNORECASE,
)
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,})(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_SPECIAL_BLOCK_RE = re.compile(r"^\s*(#{1,6}\s+|>|[-*+]\s+|\d+[.)]\s+|\|)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]+\)")
_MD_REF_LINK_RE = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")
_MD_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+]:\s+\S+.*$")
_URL_RE = re.compile(r"<?https?://[^>\s]+>?", re.IGNORECASE)
_HTML_LINK_RE = re.compile(
    r"<a\b[^>]*href\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)[^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_LINK_TAG_RE = re.compile(r"</?a\b[^>]*>", re.IGNORECASE)
_ABSTRACT_NOISE_RE = re.compile(
    r"^\s*(跳到主要内容|skip to main content)\s*$", re.IGNORECASE
)
_BLOCK_SEPARATOR_RE = re.compile(r"^\s*(?:[=-]{2,}|(?:[-*_]\s*){3,})\s*$")
ContentKind = Literal["html", "text"]
_WS_RE = re.compile(r"\s+")
_META_CHARSET_RE = re.compile(
    r"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)
_CT_CHARSET_RE = re.compile(r"charset\s*=\s*([a-zA-Z0-9_\-]+)", re.IGNORECASE)

_YAML_SPECIAL_CHARS = set(":\\\"'\n#[{}]")


def _build_frontmatter(metadata: dict[str, str]) -> str:
    if not metadata:
        return ""
    lines = ["---"]
    for key, value in metadata.items():
        if not value:
            continue
        item = str(value)
        if not item:
            item = '""'
        elif any(ch in _YAML_SPECIAL_CHARS for ch in item):
            item = '"' + item.replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines.append(f"{key}: {item}")
    lines.append("---")
    return "\n".join(lines)


def guess_apparent_encoding(data: bytes) -> str | None:
    sample = data[:65536]
    with suppress(Exception):
        from charset_normalizer import from_bytes  # noqa: I001, PLC0415

        best = from_bytes(sample).best()
        enc = getattr(best, "encoding", None) if best is not None else None
        if enc:
            return str(enc)
    with suppress(Exception):
        import chardet  # noqa: PLC0415

        det = chardet.detect(sample)
        enc = det.get("encoding") if isinstance(det, dict) else None
        if enc:
            return str(enc)
    return None


def decode_best_effort(
    data: bytes,
    *,
    content_type: str | None,
    resp_encoding: str | None = None,
    apparent_encoding: str | None = None,
) -> tuple[str, ContentKind]:
    if not data:
        return "", "text"
    head = data[:8192].lower()
    kind: ContentKind = (
        "html"
        if any(
            tok in head
            for tok in (b"<!doctype", b"<html", b"<meta", b"<body", b"</p", b"</div")
        )
        else "text"
    )
    candidates: list[str] = []
    declared: list[str] = []
    if data.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
        declared.append("utf-8-sig")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        candidates.append("utf-16")
        declared.append("utf-16")
    if content_type:
        m = _CT_CHARSET_RE.search(content_type)
        if m:
            cs = m.group(1)
            candidates.append(cs)
            declared.append(cs)
    meta_head = data[:16384].decode("ascii", errors="ignore")
    meta_match = _META_CHARSET_RE.search(meta_head)
    if meta_match:
        meta = (meta_match.group(1) or "").strip()
        if meta:
            candidates.append(meta)
            declared.append(meta)
    if resp_encoding:
        candidates.append(resp_encoding)
    if apparent_encoding:
        candidates.append(apparent_encoding)
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
    # Prefer declared charset when it produces "clean" output.
    for enc in declared:
        try:
            text = data.decode(enc, errors="replace")
        except Exception:  # noqa: S112
            continue
        text = text.replace("\x00", "")
        total = max(1, len(text))
        repl = text.count("\ufffd") / total
        ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t") / total
        if repl <= 0.001 and ctrl <= 0.001:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if kind == "text":
                text = _WS_RE.sub(" ", text)
            return text, kind
    best_text = ""
    best_key: tuple[float, float, float, float] | None = None
    for enc in ordered:
        try:
            text = data.decode(enc, errors="replace")
        except Exception:  # noqa: S112
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
    if kind == "text":
        best_text = _WS_RE.sub(" ", best_text)
    return best_text, kind


def finalize_markdown(
    *, markdown: str, max_chars: int, metadata: dict[str, str] | None = None
) -> str:
    if not markdown:
        return _build_frontmatter(metadata) if metadata else ""

    source_lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    paragraph_buf: list[str] = []
    seen_paragraphs: set[str] = set()
    table_buf: list[str] = []
    mode = "normal"
    active_fence = ""

    for raw_line in source_lines:
        fence = _fence_delimiter(raw_line)
        if mode == "fenced_code":
            out.append(raw_line)
            if fence and len(fence) >= len(active_fence):
                mode = "normal"
                active_fence = ""
            continue
        if mode == "table":
            if _is_table_line(raw_line):
                table_buf.append(raw_line.rstrip())
                continue
            table_block = "\n".join(table_buf).strip("\n")
            table_buf = []
            if table_block:
                out.append(table_block)
            mode = "normal"
        if fence:
            if paragraph_buf:
                block = "\n".join(paragraph_buf).strip()
                paragraph_buf = []
                if block:
                    key = _collapse_ws_outside_inline_code(block).lower().strip()
                    if len(key) <= 10:
                        key = ""
                    if not key or key not in seen_paragraphs:
                        if key:
                            seen_paragraphs.add(key)
                        out.append(block)
            if out and out[-1] != "":
                out.append("")
            out.append(raw_line.rstrip())
            mode = "fenced_code"
            active_fence = fence
            continue
        if _is_table_line(raw_line):
            if paragraph_buf:
                block = "\n".join(paragraph_buf).strip()
                paragraph_buf = []
                if block:
                    key = _collapse_ws_outside_inline_code(block).lower().strip()
                    if len(key) <= 10:
                        key = ""
                    if not key or key not in seen_paragraphs:
                        if key:
                            seen_paragraphs.add(key)
                        out.append(block)
            if out and out[-1] != "":
                out.append("")
            mode = "table"
            table_buf.append(raw_line.rstrip())
            continue
        if not raw_line.strip():
            if paragraph_buf:
                block = "\n".join(paragraph_buf).strip()
                paragraph_buf = []
                if block:
                    key = _collapse_ws_outside_inline_code(block).lower().strip()
                    if len(key) <= 10:
                        key = ""
                    if not key or key not in seen_paragraphs:
                        if key:
                            seen_paragraphs.add(key)
                        out.append(block)
            if out and out[-1] != "":
                out.append("")
            continue
        leading = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        leading = leading.replace("\t", "    ")
        body = _collapse_ws_outside_inline_code(raw_line[len(leading) :]).strip()
        normalized = (
            ""
            if not body
            else f"{leading}{body}".rstrip()
            if _SPECIAL_BLOCK_RE.match(body)
            else body
        )
        if not normalized:
            continue
        if _NOISE_LINE_RE.search(_collapse_ws_outside_inline_code(normalized).strip()):
            continue
        if _SPECIAL_BLOCK_RE.match(normalized):
            if paragraph_buf:
                block = "\n".join(paragraph_buf).strip()
                paragraph_buf = []
                if block:
                    key = _collapse_ws_outside_inline_code(block).lower().strip()
                    if len(key) <= 10:
                        key = ""
                    if not key or key not in seen_paragraphs:
                        if key:
                            seen_paragraphs.add(key)
                        out.append(block)
            out.append(normalized)
            continue
        paragraph_buf.append(normalized)
    if table_buf:
        table_block = "\n".join(table_buf).strip("\n")
        if table_block:
            out.append(table_block)
    if paragraph_buf:
        block = "\n".join(paragraph_buf).strip()
        if block:
            key = _collapse_ws_outside_inline_code(block).lower().strip()
            if len(key) <= 10:
                key = ""
            if not key or key not in seen_paragraphs:
                if key:
                    seen_paragraphs.add(key)
                out.append(block)
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
    result = "\n".join(compact).strip()

    if metadata:
        frontmatter = _build_frontmatter(metadata)
        if frontmatter:
            result = f"{frontmatter}\n\n{result}" if result else frontmatter

    if len(result) <= int(max_chars):
        return result
    return _clip_with_structure(result=result, max_chars=int(max_chars))


def markdown_to_text(markdown: str) -> str:
    out: list[str] = []
    in_code = False
    active_fence = ""
    for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence = _fence_delimiter(raw)
        if fence:
            if not in_code:
                in_code = True
                active_fence = fence
            elif len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
            continue
        if in_code:
            if raw:
                out.append(raw)
            continue
        line = _MD_PREFIX_RE.sub("", raw.strip())
        line = _strip_markdown_line(line, keep_image_text=False)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = line.replace("|", " ")
        line = clean_whitespace(line)
        if line:
            out.append(line)
    return "\n".join(out).strip()


def markdown_to_abstract_text(markdown: str) -> str:
    out: list[str] = []
    in_code = False
    active_fence = ""
    for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence = _fence_delimiter(raw)
        if fence:
            if not in_code:
                in_code = True
                active_fence = fence
            elif len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
            continue
        if in_code:
            continue
        line = raw.strip()
        if not line:
            continue
        if _MD_REF_DEF_RE.match(line):
            continue
        if _ABSTRACT_NOISE_RE.match(line):
            continue
        if _BLOCK_SEPARATOR_RE.match(line):
            continue
        if _TABLE_SEP_RE.match(line):
            continue
        line = _MD_PREFIX_RE.sub("", line)
        line = _strip_markdown_line(line, keep_image_text=False)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = line.replace("\\*", "*").replace("\\_", "_")
        line = line.replace("**", "").replace("__", "")
        line = line.replace("|", " ")
        line = clean_whitespace(line)
        if not line:
            continue
        if _ABSTRACT_NOISE_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


def strip_markdown_links(markdown: str) -> str:
    if not markdown:
        return ""
    out: list[str] = []
    in_code = False
    active_fence = ""
    for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence = _fence_delimiter(raw)
        if fence:
            if not in_code:
                in_code = True
                active_fence = fence
            elif len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
            out.append(raw.rstrip())
            continue
        if in_code:
            out.append(raw.rstrip())
            continue
        line = raw.rstrip()
        if _MD_REF_DEF_RE.match(line.strip()):
            continue
        line = _strip_markdown_line(line, keep_image_text=True)
        out.append(line)
    return "\n".join(out).strip()


def _strip_markdown_line(line: str, *, keep_image_text: bool) -> str:
    line = _HTML_LINK_RE.sub(r"\1", line)
    line = _HTML_LINK_TAG_RE.sub("", line)
    line = _IMAGE_RE.sub(r"\1" if keep_image_text else "", line)
    line = _MD_LINK_RE.sub(r"\1", line)
    line = _MD_REF_LINK_RE.sub(r"\1", line)
    return _URL_RE.sub("", line)


def _fence_delimiter(line: str) -> str | None:
    match = _FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return match.group(1)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" not in stripped:
        return False
    if _TABLE_SEP_RE.match(stripped):
        return True
    return stripped.count("|") >= 2


def _collapse_ws_outside_inline_code(text: str) -> str:
    out: list[str] = []
    in_code = False
    code_fence_len = 0
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "`":
            run = 1
            while idx + run < len(text) and text[idx + run] == "`":
                run += 1
            ticks = "`" * run
            out.append(ticks)
            if not in_code:
                in_code = True
                code_fence_len = run
            elif run >= code_fence_len:
                in_code = False
                code_fence_len = 0
            idx += run
            continue
        if in_code:
            out.append(ch)
            idx += 1
            continue
        if ch.isspace():
            if not out or out[-1] != " ":
                out.append(" ")
            idx += 1
            continue
        out.append(ch)
        idx += 1
    return "".join(out)


def _clip_with_structure(*, result: str, max_chars: int) -> str:
    if len(result) <= max_chars:
        return result
    lines = result.splitlines(keepends=True)
    total = 0
    last_safe = 0
    open_fence = ""
    for line in lines:
        next_total = total + len(line)
        stripped = line.strip()
        fence = _fence_delimiter(stripped)
        if fence:
            if not open_fence:
                open_fence = fence
            elif len(fence) >= len(open_fence):
                open_fence = ""
        if next_total <= max_chars and not open_fence:
            if (
                stripped == ""
                or stripped.startswith(("#", "|"))
                or bool(re.match(r"^[-*+]\s+", stripped))
                or bool(re.match(r"^\d+[.)]\s+", stripped))
            ):
                last_safe = next_total
        if next_total > max_chars:
            break
        total = next_total
    cut = max_chars
    if last_safe > int(max_chars * 0.55):
        cut = last_safe
    clipped = result[:cut].rstrip()
    offset = 0
    open_fence = ""
    unclosed_start = -1
    for line in clipped.splitlines(keepends=True):
        stripped = line.strip()
        fence = _fence_delimiter(stripped)
        if fence:
            if not open_fence:
                open_fence = fence
                unclosed_start = offset
            elif len(fence) >= len(open_fence):
                open_fence = ""
                unclosed_start = -1
        offset += len(line)
    if unclosed_start >= 0:
        clipped = clipped[:unclosed_start].rstrip()
    return clipped


__all__ = [
    "ContentKind",
    "decode_best_effort",
    "guess_apparent_encoding",
    "finalize_markdown",
    "markdown_to_abstract_text",
    "strip_markdown_links",
    "markdown_to_text",
]
