from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, TypeAlias
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from selectolax.parser import HTMLParser, Node

from serpsage.models.extract import ExtractContentTag, ExtractedImageLink, ExtractedLink
from serpsage.utils import clean_whitespace

SectionName: TypeAlias = Literal["primary", "secondary"]

_TRACKING_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "msclkid",
}
_SECONDARY_PATTERNS = re.compile(
    r"(sidebar|related|recommend|comment|discussion|thread|reply|faq|"
    r"supplement|appendix|toc|table-of-contents|index|more-like-this)",
    re.IGNORECASE,
)
_BANNER_PATTERNS = re.compile(r"(banner|hero|masthead|topbar)", re.IGNORECASE)
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


@dataclass(slots=True)
class HtmlSnapshot:
    tree: HTMLParser
    primary_html: str
    primary_path: tuple[int, ...] | None
    secondary_html: list[str]
    secondary_paths: list[tuple[int, ...]]
    semantic_html: dict[ExtractContentTag, list[str]]
    favicon: str


def build_html_snapshot(*, raw_html: str, base_url: str) -> HtmlSnapshot:
    tree = HTMLParser(raw_html)
    primary_root = _pick_primary_root(tree)
    secondary_roots = _dedupe_nested_nodes(
        [
            node
            for node in _candidate_secondary_roots(tree)
            if primary_root is None or not _same_node(node, primary_root)
        ]
    )
    primary_path = _node_path(primary_root) if primary_root is not None else None
    secondary_paths = [
        path for node in secondary_roots if (path := _node_path(node)) is not None
    ]
    semantic_html: dict[ExtractContentTag, list[str]] = {
        "metadata": [],
        "header": _collect_unique_html(tree.css("header")),
        "navigation": _collect_unique_html(tree.css("nav, [role='navigation']")),
        "banner": _collect_unique_html(
            [
                node
                for selector in (
                    "[role='banner']",
                    "[class*='banner']",
                    "[id*='banner']",
                    "[class*='hero']",
                    "[id*='hero']",
                    "[class*='masthead']",
                    "[id*='masthead']",
                )
                for node in tree.css(selector)
                if _is_banner_node(node)
            ]
        ),
        "body": [str(primary_root.html or "")] if primary_root is not None else [],
        "sidebar": [str(node.html or "") for node in secondary_roots if node.html],
        "footer": _collect_unique_html(tree.css("footer, [role='contentinfo']")),
    }
    return HtmlSnapshot(
        tree=tree,
        primary_html=str(primary_root.html or "") if primary_root is not None else "",
        primary_path=primary_path,
        secondary_html=semantic_html["sidebar"],
        secondary_paths=secondary_paths,
        semantic_html=semantic_html,
        favicon=extract_favicon(tree=tree, base_url=base_url),
    )


def collect_links_inventory(
    *,
    snapshot: HtmlSnapshot,
    base_url: str,
    include_secondary_content: bool,
    max_links: int,
    keep_hash: bool,
) -> list[ExtractedLink]:
    out: list[ExtractedLink] = []
    seen: set[tuple[str, str, SectionName]] = set()
    base_norm = normalize_url(url=base_url, base_url=base_url, keep_hash=keep_hash)
    if base_norm is None:
        base_norm = base_url
    base_netloc = urlparse(base_norm).netloc.lower()
    position = 0
    for anchor in snapshot.tree.css("a"):
        position += 1
        href = str(anchor.attributes.get("href", "")).strip()
        text = _node_text(anchor)
        if not href or not text:
            continue
        normalized = normalize_url(url=href, base_url=base_url, keep_hash=keep_hash)
        if not normalized:
            continue
        section = classify_section(anchor=anchor, snapshot=snapshot)
        if section == "secondary" and not include_secondary_content:
            continue
        key = (normalized, text.casefold(), section)
        if key in seen:
            continue
        seen.add(key)
        parsed = urlparse(normalized)
        rel_value = str(anchor.attributes.get("rel", "")).strip().lower()
        out.append(
            ExtractedLink(
                url=normalized,
                anchor_text=text,
                section=section,
                is_internal=(parsed.netloc.lower() == base_netloc)
                if parsed.netloc
                else True,
                nofollow=("nofollow" in rel_value.split()),
                same_page=(strip_fragment(normalized) == strip_fragment(base_norm)),
                source_hint=source_hint(anchor),
                position=position,
            )
        )
        if len(out) >= max(1, int(max_links)):
            break
    return out


def collect_image_links_inventory(
    *,
    snapshot: HtmlSnapshot,
    base_url: str,
    include_secondary_content: bool,
    max_links: int,
    keep_hash: bool,
) -> list[ExtractedImageLink]:
    out: list[ExtractedImageLink] = []
    seen: set[tuple[str, SectionName]] = set()
    base_norm = normalize_url(url=base_url, base_url=base_url, keep_hash=keep_hash)
    if base_norm is None:
        base_norm = base_url
    base_netloc = urlparse(base_norm).netloc.lower()
    position = 0
    for image in snapshot.tree.css("img, source"):
        position += 1
        section = classify_section(anchor=image, snapshot=snapshot)
        if section == "secondary" and not include_secondary_content:
            continue
        alt_text = clean_whitespace(str(image.attributes.get("alt", "")).strip())
        for raw in image_url_candidates(image=image):
            normalized = normalize_url(
                url=raw,
                base_url=base_url,
                keep_hash=keep_hash,
            )
            if not normalized:
                continue
            key = (normalized, section)
            if key in seen:
                continue
            seen.add(key)
            parsed = urlparse(normalized)
            out.append(
                ExtractedImageLink(
                    url=normalized,
                    alt_text=alt_text,
                    section=section,
                    is_internal=(parsed.netloc.lower() == base_netloc)
                    if parsed.netloc
                    else True,
                    source_hint=source_hint(image),
                    position=position,
                )
            )
            if len(out) >= max(1, int(max_links)):
                return out
    return out


def classify_section(*, anchor: Node, snapshot: HtmlSnapshot) -> SectionName:
    path = _node_path(anchor)
    if path is None:
        return "secondary"
    if any(_is_path_within(path, secondary) for secondary in snapshot.secondary_paths):
        return "secondary"
    if snapshot.primary_path is not None and _is_path_within(
        path, snapshot.primary_path
    ):
        return "primary"
    return "secondary"


def extract_favicon(*, tree: HTMLParser, base_url: str) -> str:
    for node in tree.css("link[rel]"):
        rel_value = str(node.attributes.get("rel", "")).strip().lower()
        if "icon" not in rel_value:
            continue
        href = str(node.attributes.get("href", "")).strip()
        if not href:
            continue
        resolved = normalize_url(url=href, base_url=base_url, keep_hash=True)
        if resolved:
            return resolved
    return ""


def normalize_url(*, url: str, base_url: str, keep_hash: bool) -> str | None:
    raw = (url or "").strip()
    if not raw or raw.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    try:
        joined = str(urljoin(base_url, raw))
        parsed = urlparse(joined)
        if parsed.scheme not in {"http", "https"}:
            return None
        clean_pairs = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_KEYS and not key.lower().startswith("utm_")
        ]
        fragment = parsed.fragment if keep_hash else ""
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(clean_pairs, doseq=True),
                fragment,
            )
        )
    except Exception:
        return None


def strip_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def source_hint(node: Node) -> str:
    current: Node | None = node
    hops = 0
    while current is not None and hops < 4:
        ident = " ".join(
            [
                str(current.attributes.get("id", "")).strip(),
                str(current.attributes.get("class", "")).strip(),
            ]
        ).strip()
        if ident:
            return ident[:120]
        current = current.parent
        hops += 1
    return "unknown"


def image_url_candidates(*, image: Node) -> list[str]:
    out: list[str] = []
    for attr in ("src", "data-src"):
        value = str(image.attributes.get(attr, "")).strip()
        if value:
            out.append(value)
    for attr in ("srcset", "data-srcset"):
        value = str(image.attributes.get(attr, "")).strip()
        if not value:
            continue
        for item in value.split(","):
            url = item.strip().split(" ", 1)[0].strip()
            if url:
                out.append(url)
    return out


def text_len(node: Node | None) -> int:
    return len(_node_text(node))


def block_count(fragment_html: str) -> int:
    tree = HTMLParser(fragment_html)
    count = 0
    for node in tree.css(",".join(sorted(_BLOCK_TAGS))):
        if _node_text(node):
            count += 1
    return count


def _pick_primary_root(tree: HTMLParser) -> Node | None:
    for selector in ("main", "article", "[role='main']", "body"):
        candidates = [node for node in tree.css(selector) if _node_text(node)]
        if candidates:
            return max(candidates, key=text_len)
    body = tree.body
    return body if body is not None and _node_text(body) else None


def _candidate_secondary_roots(tree: HTMLParser) -> list[Node]:
    nodes = [
        *tree.css("aside, [role='complementary']"),
        *[
            node
            for selector in (
                "[class*='sidebar']",
                "[id*='sidebar']",
                "[class*='related']",
                "[id*='related']",
                "[class*='recommend']",
                "[id*='recommend']",
                "[class*='comment']",
                "[id*='comment']",
                "[class*='discussion']",
                "[id*='discussion']",
                "[class*='faq']",
                "[id*='faq']",
                "[class*='appendix']",
                "[id*='appendix']",
            )
            for node in tree.css(selector)
        ],
    ]
    return [node for node in nodes if _is_secondary_candidate(node)]


def _collect_unique_html(nodes: list[Node]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node in _dedupe_nested_nodes(nodes):
        html_value = str(node.html or "").strip()
        if not html_value or html_value in seen:
            continue
        seen.add(html_value)
        out.append(html_value)
    return out


def _dedupe_nested_nodes(nodes: list[Node]) -> list[Node]:
    unique_by_path: dict[tuple[int, ...], Node] = {}
    for node in nodes:
        path = _node_path(node)
        if path is None:
            continue
        unique_by_path.setdefault(path, node)
    deduped: list[tuple[tuple[int, ...], Node]] = []
    for path, node in sorted(unique_by_path.items(), key=lambda item: len(item[0])):
        if any(_is_path_within(path, kept_path) for kept_path, _ in deduped):
            continue
        deduped.append((path, node))
    return [node for _, node in deduped]


def _is_secondary_candidate(node: Node) -> bool:
    if text_len(node) < 20:
        return False
    if _is_hidden(node):
        return False
    ident = " ".join(
        [
            str(node.attributes.get("id", "")).strip(),
            str(node.attributes.get("class", "")).strip(),
        ]
    ).strip()
    if node.tag == "aside":
        return True
    if str(node.attributes.get("role", "")).strip().lower() == "complementary":
        return True
    return bool(ident and _SECONDARY_PATTERNS.search(ident))


def _is_banner_node(node: Node) -> bool:
    role_value = str(node.attributes.get("role", "")).strip().lower()
    if role_value == "banner":
        return True
    ident = " ".join(
        [
            str(node.attributes.get("id", "")).strip(),
            str(node.attributes.get("class", "")).strip(),
        ]
    ).strip()
    return bool(ident and _BANNER_PATTERNS.search(ident))


def _is_hidden(node: Node) -> bool:
    if str(node.attributes.get("aria-hidden", "")).strip().lower() == "true":
        return True
    style = str(node.attributes.get("style", "")).strip().lower().replace(" ", "")
    return "display:none" in style or "visibility:hidden" in style


def _node_path(node: Node | None) -> tuple[int, ...] | None:
    if node is None:
        return None
    parts: list[int] = []
    current: Node | None = node
    while current is not None and current.parent is not None:
        parent = current.parent
        if parent.child is None:
            break
        siblings: list[Node] = []
        cursor: Node | None = parent.child
        while cursor is not None:
            siblings.append(cursor)
            cursor = cursor.next
        try:
            index = next(
                idx
                for idx, sibling in enumerate(siblings)
                if _same_node(sibling, current)
            )
        except StopIteration:
            return None
        parts.append(index)
        current = parent
        if current.tag == "html":
            break
    parts.reverse()
    return tuple(parts)


def _is_path_within(path: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


def _same_node(left: Node, right: Node) -> bool:
    return _node_id(left) == _node_id(right)


def _node_id(node: Node) -> int:
    value = node.mem_id
    return int(value()) if callable(value) else int(value)


def _node_text(node: Node | None) -> str:
    if node is None:
        return ""
    return clean_whitespace(node.text(separator=" ", strip=True))


__all__ = [
    "HtmlSnapshot",
    "SectionName",
    "block_count",
    "build_html_snapshot",
    "classify_section",
    "collect_image_links_inventory",
    "collect_links_inventory",
    "extract_favicon",
    "image_url_candidates",
    "normalize_url",
    "source_hint",
    "strip_fragment",
    "text_len",
]
