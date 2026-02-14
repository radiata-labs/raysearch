from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from serpsage.components.extract.markdown.dom import is_descendant_of

if TYPE_CHECKING:
    from bs4 import BeautifulSoup as SoupType

_BLOCK_TAGS = {
    "blockquote",
    "dl",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "ol",
    "p",
    "pre",
    "table",
    "ul",
}
_NOISE_LINE_RE = re.compile(
    r"(privacy policy|cookie policy|terms of service|all rights reserved|"
    r"sign up|subscribe|advertisement|sponsored content|related posts)",
    re.IGNORECASE,
)
_LANG_CLASS_RE = re.compile(r"(?:^|\b)(?:language|lang)-([A-Za-z0-9_+#.-]+)(?:\b|$)")
_WS_RE = re.compile(r"\s+")


@dataclass(slots=True)
class InlineToken:
    kind: Literal["text", "code", "link", "em", "strong", "linebreak"]
    text: str = ""
    href: str | None = None
    children: list[InlineToken] = field(default_factory=list)


def render_markdown(
    *,
    root: Tag | SoupType,
    base_url: str,
    skip_roots: list[Tag] | None = None,
) -> tuple[str, dict[str, int]]:
    lines: list[str] = []
    stats = {
        "heading_count": 0,
        "list_count": 0,
        "ordered_list_count": 0,
        "table_count": 0,
        "table_row_count": 0,
        "code_block_count": 0,
        "inline_code_count": 0,
        "link_count": 0,
        "block_count": 0,
    }
    skip_roots = list(skip_roots or [])
    for el in _iter_renderable_blocks(root=root):
        if _should_skip(el, root=root, skip_roots=skip_roots):
            continue
        rendered = _render_block(tag=el, base_url=base_url, stats=stats)
        if not rendered:
            continue
        lines.append(rendered)
        stats["block_count"] += 1
    return "\n\n".join(lines).strip(), stats


def render_secondary_markdown(
    *,
    secondary_roots: list[Tag],
    base_url: str,
) -> tuple[str, dict[str, int]]:
    blocks: list[str] = []
    merged = {
        "heading_count": 0,
        "list_count": 0,
        "ordered_list_count": 0,
        "table_count": 0,
        "table_row_count": 0,
        "code_block_count": 0,
        "inline_code_count": 0,
        "link_count": 0,
        "block_count": 0,
    }
    for root in secondary_roots:
        md, stats = render_markdown(root=root, base_url=base_url)
        if md:
            blocks.append(md)
        for key, value in stats.items():
            merged[key] = int(merged.get(key, 0)) + int(value)
    return "\n\n".join(blocks).strip(), merged


def _iter_renderable_blocks(*, root: Tag | SoupType) -> list[Tag]:
    out: list[Tag] = []
    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        name = (node.name or "").lower()
        if name not in _BLOCK_TAGS:
            continue
        out.append(node)
    return out


def _should_skip(
    tag: Tag,
    *,
    root: Tag | SoupType,
    skip_roots: list[Tag],
) -> bool:
    if _has_block_ancestor(tag, root):
        return True
    return any(tag is skip or is_descendant_of(tag, skip) for skip in skip_roots)


def _has_block_ancestor(tag: Tag, root: Tag | SoupType) -> bool:
    parent = tag.parent
    while parent is not None and parent is not root:
        if isinstance(parent, Tag) and (parent.name or "").lower() in _BLOCK_TAGS:
            return True
        parent = parent.parent
    return False


def _render_block(*, tag: Tag, base_url: str, stats: dict[str, int]) -> str:
    name = (tag.name or "").lower()
    if name == "pre":
        return _render_preformatted(tag=tag, stats=stats)

    if name in {"ul", "ol"}:
        return _render_list(list_tag=tag, base_url=base_url, stats=stats, depth=0)

    if name == "table":
        table_md = _render_table(table=tag, base_url=base_url, stats=stats)
        if table_md:
            stats["table_count"] += 1
        return table_md

    if name == "blockquote":
        text = _render_inline_text(tag=tag, base_url=base_url, stats=stats)
        if not text:
            return ""
        return "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))

    if name == "hr":
        return "---"

    if name == "dl":
        return _render_description_list(dl=tag, base_url=base_url, stats=stats)

    text = _render_inline_text(tag=tag, base_url=base_url, stats=stats)
    if not text:
        return ""
    if _NOISE_LINE_RE.search(_WS_RE.sub(" ", text).strip()):
        return ""

    if name.startswith("h") and len(name) == 2 and name[1].isdigit():
        level = min(6, max(1, int(name[1])))
        stats["heading_count"] += 1
        return f"{'#' * level} {text}"

    return text


def _render_preformatted(*, tag: Tag, stats: dict[str, int]) -> str:
    code_tag = tag.find("code")
    source = code_tag if isinstance(code_tag, Tag) else tag
    code = source.get_text("", strip=False)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    if code == "":
        return ""

    lang = _detect_code_language(pre=tag, code=code_tag if isinstance(code_tag, Tag) else None)
    fence = _pick_code_fence(code)
    info = lang or ""
    stats["code_block_count"] += 1

    if code.endswith("\n"):
        return f"{fence}{info}\n{code}{fence}"
    return f"{fence}{info}\n{code}\n{fence}"


def _render_description_list(*, dl: Tag, base_url: str, stats: dict[str, int]) -> str:
    out: list[str] = []
    terms = dl.find_all(["dt", "dd"], recursive=False)
    current_term = ""
    for item in terms:
        name = (item.name or "").lower()
        text = _render_inline_text(tag=item, base_url=base_url, stats=stats)
        if not text:
            continue
        if name == "dt":
            current_term = text
            out.append(f"- **{text}**")
            continue
        if current_term:
            out.append(f"  - {text}")
        else:
            out.append(f"- {text}")
    return "\n".join(out).strip()


def _render_list(
    *,
    list_tag: Tag,
    base_url: str,
    stats: dict[str, int],
    depth: int,
) -> str:
    ordered = (list_tag.name or "").lower() == "ol"
    items = list_tag.find_all("li", recursive=False)
    if not items:
        return ""

    lines: list[str] = []
    for idx, li in enumerate(items, start=1):
        marker = f"{idx}. " if ordered else "- "
        prefix = f"{'  ' * depth}{marker}"
        body = _render_list_item_text(li=li, base_url=base_url, stats=stats)
        if body:
            lines.append(f"{prefix}{body}")
        else:
            lines.append(prefix.rstrip())

        nested = li.find_all(["ul", "ol"], recursive=False)
        for child in nested:
            nested_md = _render_list(
                list_tag=child,
                base_url=base_url,
                stats=stats,
                depth=depth + 1,
            )
            if nested_md:
                lines.append(nested_md)

        stats["list_count"] += 1
        if ordered:
            stats["ordered_list_count"] += 1

    return "\n".join(lines).rstrip()


def _render_list_item_text(*, li: Tag, base_url: str, stats: dict[str, int]) -> str:
    clone_soup = BeautifulSoup(str(li), "html.parser")
    clone_li = clone_soup.find("li")
    if not isinstance(clone_li, Tag):
        return ""
    for nested in clone_li.find_all(["ul", "ol"]):
        nested.decompose()
    return _render_inline_text(tag=clone_li, base_url=base_url, stats=stats)


def _render_inline_text(*, tag: Tag, base_url: str, stats: dict[str, int]) -> str:
    tokens: list[InlineToken] = []
    for child in tag.children:
        tokens.extend(_collect_inline_tokens(node=child, base_url=base_url, stats=stats))
    if not tokens:
        tokens = _collect_inline_tokens(node=tag, base_url=base_url, stats=stats)
    return _serialize_inline_tokens(tokens=tokens).strip()


def _collect_inline_tokens(
    *,
    node: object,
    base_url: str,
    stats: dict[str, int],
) -> list[InlineToken]:
    if isinstance(node, NavigableString):
        return [InlineToken(kind="text", text=html.unescape(str(node)))]

    if not isinstance(node, Tag):
        return []

    name = (node.name or "").lower()
    if name in {"script", "style", "noscript"}:
        return []

    if name == "br":
        return [InlineToken(kind="linebreak")]

    if name in {"code", "kbd", "samp"}:
        text = node.get_text("", strip=False).replace("\r\n", "\n").replace("\r", "\n")
        stats["inline_code_count"] += 1
        return [InlineToken(kind="code", text=text)]

    if name == "a":
        href = _safe_join(base_url=base_url, href=str(node.get("href") or ""))
        children: list[InlineToken] = []
        for child in node.children:
            children.extend(_collect_inline_tokens(node=child, base_url=base_url, stats=stats))
        if not children:
            label = html.unescape(node.get_text(" ", strip=True))
            if label:
                children = [InlineToken(kind="text", text=label)]
        if href:
            stats["link_count"] += 1
            return [InlineToken(kind="link", href=href, children=children)]
        return children

    if name in {"em", "i"}:
        children = []
        for child in node.children:
            children.extend(_collect_inline_tokens(node=child, base_url=base_url, stats=stats))
        return [InlineToken(kind="em", children=children)] if children else []

    if name in {"strong", "b"}:
        children = []
        for child in node.children:
            children.extend(_collect_inline_tokens(node=child, base_url=base_url, stats=stats))
        return [InlineToken(kind="strong", children=children)] if children else []

    out: list[InlineToken] = []
    for child in node.children:
        out.extend(_collect_inline_tokens(node=child, base_url=base_url, stats=stats))
    return out


def _serialize_inline_tokens(*, tokens: list[InlineToken]) -> str:
    out_parts: list[str] = []
    for token in tokens:
        rendered = _render_inline_token(token=token)
        if not rendered:
            continue
        _append_chunk(out_parts, rendered)
    return "".join(out_parts).replace(" \n", "\n").replace("\n ", "\n")


def _render_inline_token(*, token: InlineToken) -> str:
    if token.kind == "text":
        return _normalize_text_fragment(token.text)

    if token.kind == "linebreak":
        return "\n"

    if token.kind == "code":
        return _render_inline_code(text=token.text)

    if token.kind == "link":
        label = _serialize_inline_tokens(tokens=token.children).strip()
        if not label:
            return ""
        if token.href:
            return f"[{label}]({token.href})"
        return label

    if token.kind == "em":
        inner = _serialize_inline_tokens(tokens=token.children).strip()
        return f"*{inner}*" if inner else ""

    if token.kind == "strong":
        inner = _serialize_inline_tokens(tokens=token.children).strip()
        return f"**{inner}**" if inner else ""

    return ""


def _normalize_text_fragment(text: str) -> str:
    normalized = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    normalized = normalized.replace("\t", " ")
    return _WS_RE.sub(" ", normalized)


def _append_chunk(parts: list[str], chunk: str) -> None:
    if not chunk:
        return
    if not parts:
        parts.append(chunk)
        return

    prev = parts[-1]
    next_chunk = chunk
    if prev.endswith("\n"):
        next_chunk = next_chunk.lstrip(" ")
    elif prev.endswith(" ") and next_chunk.startswith(" "):
        next_chunk = next_chunk[1:]

    if next_chunk:
        parts.append(next_chunk)


def _render_inline_code(*, text: str) -> str:
    payload = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    longest = max((len(x) for x in re.findall(r"`+", payload)), default=0)
    fence = "`" * max(1, longest + 1)
    if not payload:
        return f"{fence}{fence}"
    needs_padding = payload.startswith((" ", "`")) or payload.endswith((" ", "`"))
    content = f" {payload} " if needs_padding else payload
    return f"{fence}{content}{fence}"


def _render_table(*, table: Tag, base_url: str, stats: dict[str, int]) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        row: list[str] = []
        for cell in cells:
            text = _render_inline_text(tag=cell, base_url=base_url, stats=stats)
            row.append(_escape_table_cell(text))
        rows.append(row)

    if not rows:
        return ""

    width = max(len(row) for row in rows)
    if width <= 0:
        return ""

    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    sep = ["---"] * width
    body = padded[1:]
    out = [f"| {' | '.join(header)} |", f"| {' | '.join(sep)} |"]
    out.extend(f"| {' | '.join(row)} |" for row in body)

    stats["table_row_count"] += max(0, len(body))
    return "\n".join(out)


def _escape_table_cell(text: str) -> str:
    cleaned = text.replace("\n", "\\n")
    cleaned = cleaned.replace("|", "\\|")
    return cleaned.strip()


def _detect_code_language(*, pre: Tag, code: Tag | None) -> str:
    for node in [code, pre]:
        if not isinstance(node, Tag):
            continue

        data_lang = str(node.get("data-lang") or node.get("lang") or "").strip()
        if data_lang:
            return _sanitize_info_string(data_lang)

        for cls in node.get("class") or []:
            value = str(cls).strip()
            if not value:
                continue
            match = _LANG_CLASS_RE.search(value)
            if match:
                return _sanitize_info_string(match.group(1))

    return ""


def _sanitize_info_string(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_+#.-]", "", value.strip().lower())
    return out[:32]


def _pick_code_fence(code: str) -> str:
    max_ticks = max((len(run) for run in re.findall(r"`+", code)), default=0)
    return "`" * max(3, max_ticks + 1)


def _safe_join(*, base_url: str, href: str) -> str | None:
    candidate = (href or "").strip()
    if not candidate:
        return None
    if candidate.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    try:
        return str(urljoin(base_url, candidate))
    except Exception:
        return None


__all__ = ["render_markdown", "render_secondary_markdown"]
