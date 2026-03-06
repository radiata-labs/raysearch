from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, cast
from urllib.parse import urljoin

import trafilatura
from html_to_markdown import (
    ConversionOptions,
    MetadataConfig,
    PreprocessingOptions,
    TableExtractionResult,
    convert_with_tables,
)
from selectolax.parser import HTMLParser, Node
from trafilatura.metadata import extract_metadata

from serpsage.components.extract.html.dom import block_count
from serpsage.components.extract.html.postprocess import (
    finalize_markdown,
    markdown_to_text,
)
from serpsage.utils import clean_whitespace

RenderStatValue = int | float | bool | str
RenderStats = dict[str, RenderStatValue]
BackendName = Literal["html_to_markdown"]

_BACKEND_NAME: BackendName = "html_to_markdown"
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
_BASE_CONVERSION_OPTIONS = ConversionOptions(
    heading_style="atx",
    list_indent_type="spaces",
    list_indent_width=2,
    bullets="-*+",
    strong_em_symbol="*",
    escape_asterisks=False,
    escape_underscores=False,
    escape_misc=False,
    extract_metadata=False,
    whitespace_mode="normalized",
    wrap=False,
    newline_style="spaces",
    code_block_style="backticks",
)
_PREPROCESSING_OPTIONS = PreprocessingOptions(enabled=False)
_METADATA_CONFIG = MetadataConfig(
    extract_document=False,
    extract_headers=True,
    extract_links=True,
    extract_images=True,
    extract_structured_data=False,
)


@dataclass(slots=True)
class TrafilaturaMetadata:
    title: str = ""
    published_date: str = ""
    author: str = ""
    image: str = ""


def markdownify_available() -> bool:
    return True


def extract_trafilatura_metadata(*, raw_html: str, url: str) -> TrafilaturaMetadata:
    document = extract_metadata(raw_html, default_url=url)
    image = clean_whitespace(str(getattr(document, "image", "") or ""))
    return TrafilaturaMetadata(
        title=clean_whitespace(str(getattr(document, "title", "") or "")),
        published_date=clean_whitespace(str(getattr(document, "date", "") or "")),
        author=clean_whitespace(str(getattr(document, "author", "") or "")),
        image=(urljoin(url, image) if image else ""),
    )


def extract_trafilatura_markdown(
    *,
    raw_html: str,
    url: str,
    max_chars: int,
) -> str:
    markdown = trafilatura.extract(
        raw_html,
        url=url,
        include_comments=False,
        output_format="markdown",
        favor_precision=True,
        favor_recall=False,
        include_tables=True,
        include_images=False,
        include_formatting=True,
        include_links=True,
        deduplicate=False,
    )
    if not markdown:
        raise ValueError("trafilatura returned empty markdown")
    normalized = finalize_markdown(markdown=str(markdown), max_chars=max_chars)
    if not normalized:
        raise ValueError("trafilatura normalized markdown is empty")
    return normalized


def render_fragment_markdown(
    *,
    fragment_html: str,
    base_url: str,
    preserve_html_tags: bool,
) -> tuple[str, RenderStats]:
    prepared_html = absolutize_fragment_urls(
        fragment_html=fragment_html,
        base_url=base_url,
    )
    result = convert_with_tables(
        prepared_html,
        options=conversion_options(
            fragment_html=prepared_html,
            preserve_html_tags=preserve_html_tags,
        ),
        preprocessing=_PREPROCESSING_OPTIONS,
        metadata_config=_METADATA_CONFIG,
    )
    markdown = normalize_fragment(str(result["content"]))
    return markdown, build_render_stats(
        fragment_html=prepared_html,
        markdown=markdown,
        result=result,
    )


def merge_render_stats(stats_list: list[RenderStats]) -> RenderStats:
    if not stats_list:
        return empty_render_stats()
    merged = empty_render_stats()
    ratio_sum = 0.0
    for stats in stats_list:
        for key in _COUNT_KEYS:
            merged[key] = int(merged.get(key, 0)) + int(stats.get(key, 0))
        ratio_sum += float(stats.get("renderer_text_recall_ratio", 1.0))
    merged["renderer_text_recall_ratio"] = float(ratio_sum / float(len(stats_list)))
    return merged


def empty_render_stats() -> RenderStats:
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
        "renderer_backend": _BACKEND_NAME,
        "renderer_fallback_used": False,
        "renderer_text_recall_ratio": 1.0,
    }


def conversion_options(
    *,
    fragment_html: str,
    preserve_html_tags: bool,
) -> ConversionOptions:
    if not preserve_html_tags:
        return _BASE_CONVERSION_OPTIONS
    return replace(
        _BASE_CONVERSION_OPTIONS,
        preserve_tags=collect_tag_names(fragment_html),
    )


def absolutize_fragment_urls(*, fragment_html: str, base_url: str) -> str:
    tree = HTMLParser(fragment_html)
    for node in tree.css(
        "[href], [src], [poster], [data-src], [srcset], [data-srcset]"
    ):
        for attr in ("href", "src", "poster", "data-src"):
            raw = str(node.attributes.get(attr, "")).strip()
            if raw:
                node.attributes[attr] = urljoin(base_url, raw)
        for attr in ("srcset", "data-srcset"):
            raw = str(node.attributes.get(attr, "")).strip()
            if raw:
                node.attributes[attr] = join_srcset(raw=raw, base_url=base_url)
    body = tree.body
    return str(body.html or "") if body is not None else str(tree.html or "")


def normalize_fragment(value: str) -> str:
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


def collect_tag_names(fragment_html: str) -> set[str]:
    tree = HTMLParser(fragment_html)
    return {
        str(node.tag).lower() for node in tree.css("*") if str(node.tag or "").strip()
    }


def build_render_stats(
    *,
    fragment_html: str,
    markdown: str,
    result: TableExtractionResult,
) -> RenderStats:
    tree = HTMLParser(fragment_html)
    metadata = cast("dict[str, object]", result.get("metadata") or {})
    tables = cast("list[dict[str, object]]", result.get("tables", []))
    headers = cast("list[object]", metadata.get("headers", []))
    links = cast("list[object]", metadata.get("links", []))
    images = cast("list[object]", metadata.get("images", []))
    source_text = tree.text(separator=" ", strip=True)
    return {
        "heading_count": int(len(headers)),
        "list_count": int(len(tree.css("li"))),
        "ordered_list_count": int(
            sum(
                1
                for item in tree.css("li")
                if item.parent is not None
                and str(item.parent.tag or "").lower() == "ol"
            )
        ),
        "table_count": int(len(tables)),
        "table_row_count": int(
            sum(
                1
                for table in tables
                for is_header in cast("list[bool]", table.get("is_header_row", []))
                if not bool(is_header)
            )
        ),
        "code_block_count": int(len(tree.css("pre"))),
        "inline_code_count": int(
            sum(1 for node in tree.css("code, kbd, samp") if not has_pre_ancestor(node))
        ),
        "link_count": int(len(links)),
        "image_count": int(len(images)),
        "block_count": int(block_count(fragment_html)),
        "renderer_backend": _BACKEND_NAME,
        "renderer_fallback_used": False,
        "renderer_text_recall_ratio": float(
            text_recall_ratio(source_text=source_text, markdown=markdown)
        ),
    }


def join_srcset(*, raw: str, base_url: str) -> str:
    out: list[str] = []
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        if " " in candidate:
            url_part, descriptor = candidate.split(" ", 1)
            out.append(
                f"{urljoin(base_url, url_part.strip())} {descriptor.strip()}".strip()
            )
            continue
        out.append(urljoin(base_url, candidate))
    return ", ".join(out)


def has_pre_ancestor(node: Node) -> bool:
    current = node.parent
    while current is not None:
        if str(current.tag or "").lower() == "pre":
            return True
        current = current.parent
    return False


def text_recall_ratio(*, source_text: str, markdown: str) -> float:
    normalized_source = clean_whitespace(source_text)
    source_chars = len(normalized_source)
    if source_chars <= 0:
        return 1.0
    plain = markdown_to_text(markdown)
    if not plain:
        return 0.0
    ratio = float(len(plain)) / float(source_chars)
    return max(0.0, min(1.0, ratio))


__all__ = [
    "BackendName",
    "RenderStats",
    "RenderStatValue",
    "TrafilaturaMetadata",
    "absolutize_fragment_urls",
    "conversion_options",
    "empty_render_stats",
    "extract_trafilatura_markdown",
    "extract_trafilatura_metadata",
    "markdownify_available",
    "merge_render_stats",
    "normalize_fragment",
    "render_fragment_markdown",
]
