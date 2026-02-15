from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, cast
from typing_extensions import override

from serpsage.components.extract.markdown.config import build_extract_profile
from serpsage.components.extract.markdown.dom import cleanup_dom, parse_html_document
from serpsage.components.extract.markdown.engines import (
    boilerpy3_available,
    justext_available,
    readability_available,
    run_boilerpy3,
    run_fastdom,
    run_justext,
    run_readability,
    run_trafilatura,
    trafilatura_available,
)
from serpsage.components.extract.markdown.links import (
    collect_image_links as collect_image_links_inventory,
)
from serpsage.components.extract.markdown.links import (
    collect_links as collect_links_inventory,
)
from serpsage.components.extract.markdown.postprocess import (
    extract_feature_snippets,
    finalize_markdown,
    markdown_to_plain,
    merge_markdown,
)
from serpsage.components.extract.markdown.render import (
    render_markdown,
    render_secondary_markdown,
)
from serpsage.components.extract.markdown.scoring import (
    infer_markdown_stats,
    score_candidate,
)
from serpsage.components.extract.markdown.sectioning import split_sections
from serpsage.components.extract.markdown.types import (
    CandidateDoc,
    ExtractProfile,
    SectionBuckets,
    StatsMap,
)
from serpsage.components.extract.pdf import extract_pdf_document
from serpsage.components.extract.utils import (
    decode_best_effort,
    guess_apparent_encoding,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.contracts.services import ExtractorBase
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractContentTag,
    ExtractedDocument,
)
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Callable

    from bs4 import BeautifulSoup
    from bs4.element import Tag

    from serpsage.core.runtime import Runtime

_SEMANTIC_ORDER: tuple[ExtractContentTag, ...] = (
    "metadata",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
)
_BANNER_HINT_RE = re.compile(r"(banner|hero|masthead|topbar)", re.IGNORECASE)


class MarkdownExtractor(ExtractorBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        self._profile = build_extract_profile(settings=self.settings)

    @override
    def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        profile = self._profile
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

        options = content_options or ExtractContentOptions(
            depth="high" if include_secondary_content else "low"
        )
        if kind == "text":
            return self._extract_text(
                text=decoded,
                include_secondary_content=(options.depth == "high"),
            )

        if kind != "html":
            return ExtractedDocument(
                content_kind="binary",
                extractor_used="binary",
                quality_score=0.0,
                warnings=["unsupported binary content"],
                stats={"primary_chars": 0, "secondary_chars": 0},
            )

        html_doc = decoded[: int(profile.max_html_chars)]
        if content_options is None:
            return self._extract_html(
                html_doc=html_doc,
                url=url,
                profile=profile,
                include_secondary_content=(options.depth == "high"),
                collect_links=bool(collect_links),
                collect_images=bool(collect_images),
            )
        return self._extract_html_with_options(
            html_doc=html_doc,
            url=url,
            profile=profile,
            options=options,
            collect_links=bool(collect_links),
            collect_images=bool(collect_images),
        )

    def _extract_text(
        self,
        *,
        text: str,
        include_secondary_content: bool,
    ) -> ExtractedDocument:
        lines = [
            clean_whitespace(line)
            for line in text.splitlines()
            if clean_whitespace(line)
        ]
        markdown = finalize_markdown(
            markdown="\n\n".join(lines),
            max_chars=int(self._profile.max_markdown_chars),
        )
        plain = markdown_to_plain(markdown)
        inferred = infer_markdown_stats(markdown)
        stats: StatsMap = {
            "heading_count": int(inferred.get("heading_count", 0)),
            "table_count": int(inferred.get("table_count", 0)),
            "table_row_count": int(inferred.get("table_row_count", 0)),
            "code_block_count": int(inferred.get("code_block_count", 0)),
            "inline_code_count": int(inferred.get("inline_code_count", 0)),
            "ordered_list_count": int(inferred.get("ordered_list_count", 0)),
            "link_count": int(inferred.get("link_count", 0)),
            "fence_pairs_ok": bool(inferred.get("fence_pairs_ok", True)),
        }
        quality = score_candidate(
            markdown=markdown,
            plain_text=plain,
            stats=stats,
            warnings=[],
        )

        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title="",
            content_kind="text",
            extractor_used="plain_text",
            quality_score=float(quality),
            warnings=[],
            stats={
                "primary_chars": len(plain),
                "secondary_chars": 0,
                "engine_chain": "plain_text",
                "include_secondary_content": bool(include_secondary_content),
                **stats,
            },
        )

    def _extract_html_with_options(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        options: ExtractContentOptions,
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        low_doc = self._extract_html(
            html_doc=html_doc,
            url=url,
            profile=profile,
            include_secondary_content=False,
            collect_links=collect_links,
            collect_images=collect_images,
        )

        include_secondary = False
        base_doc = low_doc
        if options.depth == "high":
            include_secondary = True
            base_doc = self._extract_html(
                html_doc=html_doc,
                url=url,
                profile=profile,
                include_secondary_content=True,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        elif options.depth == "medium":
            primary_chars = int(low_doc.stats.get("primary_chars", 0))
            has_non_body_intent = any(
                tag not in {"body", "metadata"} for tag in options.include_tags
            )
            if primary_chars < int(profile.min_primary_chars) or has_non_body_intent:
                include_secondary = True
                base_doc = self._extract_html(
                    html_doc=html_doc,
                    url=url,
                    profile=profile,
                    include_secondary_content=True,
                    collect_links=collect_links,
                    collect_images=collect_images,
                )

        needs_semantic_render = bool(
            options.include_html_tags or options.include_tags or options.exclude_tags
        )
        if not needs_semantic_render:
            return base_doc
        return self._extract_html_semantic(
            html_doc=html_doc,
            url=url,
            profile=profile,
            options=options,
            include_secondary=include_secondary,
            collect_links=collect_links,
            collect_images=collect_images,
            base_doc=base_doc,
        )

    def _extract_html_semantic(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        options: ExtractContentOptions,
        include_secondary: bool,
        collect_links: bool,
        collect_images: bool,
        base_doc: ExtractedDocument,
    ) -> ExtractedDocument:
        selected_tags = self._resolve_selected_tags(
            options=options,
            include_secondary=include_secondary,
        )
        keep_tags = (
            {str(tag) for tag in selected_tags}
            | {str(tag) for tag in options.include_tags}
            | {str(tag) for tag in options.exclude_tags}
        )
        soup = parse_html_document(html_doc)
        cleanup_dom(soup, keep_semantic_tags=keep_tags)
        buckets = split_sections(soup=soup, min_primary_chars=profile.min_primary_chars)

        markdown = self._render_selected_tags_markdown(
            soup=soup,
            buckets=buckets,
            base_url=url,
            selected_tags=selected_tags,
            preserve_html_tags=bool(options.include_html_tags),
        )
        markdown = finalize_markdown(
            markdown=markdown, max_chars=profile.max_markdown_chars
        )
        plain = markdown_to_plain(markdown)

        merged_stats: StatsMap = dict(base_doc.stats)
        inferred = infer_markdown_stats(markdown)
        merged_stats.update(
            {
                "markdown_chars": len(markdown),
                "plain_chars": len(plain),
                "primary_chars": int(len(plain)),
                "secondary_chars": int(
                    base_doc.stats.get("secondary_chars", 0)
                    if "sidebar" in selected_tags
                    else 0
                ),
                "include_secondary_content": bool(include_secondary),
                "content_depth": str(options.depth),
                "include_html_tags": bool(options.include_html_tags),
                "selected_tags": ",".join(sorted(selected_tags)),
            }
        )
        merged_stats.update(
            {
                "heading_count": int(inferred.get("heading_count", 0)),
                "table_count": int(inferred.get("table_count", 0)),
                "table_row_count": int(inferred.get("table_row_count", 0)),
                "code_block_count": int(inferred.get("code_block_count", 0)),
                "inline_code_count": int(inferred.get("inline_code_count", 0)),
                "list_count": int(inferred.get("list_count", 0)),
                "ordered_list_count": int(inferred.get("ordered_list_count", 0)),
                "link_count": int(inferred.get("link_count", 0)),
                "fence_pairs_ok": bool(inferred.get("fence_pairs_ok", True)),
            }
        )

        merged_warnings = list(base_doc.warnings or [])
        if not plain.strip() and selected_tags:
            merged_warnings.append("selected tags produced empty content")

        quality = score_candidate(
            markdown=markdown,
            plain_text=plain,
            stats=merged_stats,
            warnings=merged_warnings,
        )
        links = (
            collect_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=("sidebar" in selected_tags),
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_links
            else []
        )
        image_links = (
            collect_image_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=("sidebar" in selected_tags),
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_images
            else []
        )
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title=title,
            content_kind="html",
            extractor_used=str(base_doc.extractor_used or "fastdom"),
            quality_score=float(quality),
            warnings=merged_warnings,
            stats=merged_stats,
            links=links,
            image_links=image_links,
        )

    def _resolve_selected_tags(
        self,
        *,
        options: ExtractContentOptions,
        include_secondary: bool,
    ) -> set[ExtractContentTag]:
        if options.include_tags:
            selected = set(options.include_tags)
        else:
            selected = {"body"}
            if include_secondary:
                selected.add("sidebar")
        selected -= set(options.exclude_tags)
        return cast("set[ExtractContentTag]", selected)

    def _render_selected_tags_markdown(
        self,
        *,
        soup: BeautifulSoup,
        buckets: SectionBuckets,
        base_url: str,
        selected_tags: set[ExtractContentTag],
        preserve_html_tags: bool,
    ) -> str:
        out: list[str] = []
        for tag in _SEMANTIC_ORDER:
            if tag not in selected_tags:
                continue
            block = self._render_semantic_block(
                soup=soup,
                buckets=buckets,
                base_url=base_url,
                semantic_tag=tag,
                preserve_html_tags=preserve_html_tags,
            )
            if not block:
                continue
            if preserve_html_tags:
                out.append(f'<section data-serpsage-tag="{tag}">\n{block}\n</section>')
                continue
            if tag == "sidebar":
                out.append(f"## Secondary Content\n\n{block}")
                continue
            if tag == "body":
                out.append(block)
                continue
            out.append(f"## {tag.capitalize()}\n\n{block}")
        return "\n\n".join(out).strip()

    def _render_semantic_block(
        self,
        *,
        soup: BeautifulSoup,
        buckets: SectionBuckets,
        base_url: str,
        semantic_tag: ExtractContentTag,
        preserve_html_tags: bool,
    ) -> str:
        if semantic_tag == "metadata":
            return self._render_metadata_block(
                soup=soup,
                preserve_html_tags=preserve_html_tags,
            )
        if semantic_tag == "body":
            body_md, _ = render_markdown(
                root=buckets.primary_root,
                base_url=base_url,
                skip_roots=buckets.secondary_roots,
                preserve_html_tags=preserve_html_tags,
            )
            return body_md
        if semantic_tag == "sidebar":
            md, _ = render_secondary_markdown(
                secondary_roots=buckets.secondary_roots,
                base_url=base_url,
                preserve_html_tags=preserve_html_tags,
            )
            return md
        roots = self._find_semantic_roots(soup=soup, semantic_tag=semantic_tag)
        if not roots:
            return ""
        parts: list[str] = []
        for root in roots:
            md, _ = render_markdown(
                root=root,
                base_url=base_url,
                preserve_html_tags=preserve_html_tags,
            )
            if md:
                parts.append(md)
        return "\n\n".join(part for part in parts if part).strip()

    def _render_metadata_block(
        self,
        *,
        soup: BeautifulSoup,
        preserve_html_tags: bool,
    ) -> str:
        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        meta_pairs: list[tuple[str, str]] = []
        for meta in soup.find_all("meta"):
            key = clean_whitespace(
                str(meta.get("name") or meta.get("property") or "").strip()
            )
            value = clean_whitespace(str(meta.get("content") or "").strip())
            if not key or not value:
                continue
            meta_pairs.append((key, value))
            if len(meta_pairs) >= 20:
                break

        if preserve_html_tags:
            lines: list[str] = []
            if title:
                lines.append(f"<title>{html.escape(title)}</title>")
            for key, value in meta_pairs:
                lines.append(
                    f'<meta name="{html.escape(key, quote=True)}" '
                    f'content="{html.escape(value, quote=True)}" />'
                )
            return "\n".join(lines).strip()

        lines = []
        if title:
            lines.append(f"- title: {title}")
        for key, value in meta_pairs:
            lines.append(f"- {key}: {value}")
        return "\n".join(lines).strip()

    def _find_semantic_roots(
        self,
        *,
        soup: BeautifulSoup,
        semantic_tag: ExtractContentTag,
    ) -> list[Tag]:
        roots: list[Tag] = []
        if semantic_tag == "header":
            roots.extend(soup.find_all("header"))
        elif semantic_tag == "navigation":
            roots.extend(soup.find_all("nav"))
            roots.extend(soup.select('[role="navigation"]'))
        elif semantic_tag == "banner":
            roots.extend(soup.select('[role="banner"]'))
            for node in soup.find_all(True):
                ident = " ".join(
                    [
                        str(node.get("id") or ""),
                        " ".join(node.get("class") or []),
                    ]
                ).strip()
                if ident and _BANNER_HINT_RE.search(ident):
                    roots.append(node)
        elif semantic_tag == "footer":
            roots.extend(soup.find_all("footer"))
            roots.extend(soup.select('[role="contentinfo"]'))

        deduped: list[Tag] = []
        for root in roots:
            if any(root is keep for keep in deduped):
                continue
            if any(root in keep.descendants for keep in deduped):
                continue
            deduped.append(root)
        return deduped

    def _extract_html(
        self,
        *,
        html_doc: str,
        url: str,
        profile: ExtractProfile,
        include_secondary_content: bool,
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        soup = parse_html_document(html_doc)
        cleanup_dom(soup)
        buckets = split_sections(soup=soup, min_primary_chars=profile.min_primary_chars)

        canonical_secondary_markdown = ""
        if include_secondary_content:
            canonical_secondary_markdown, _ = render_secondary_markdown(
                secondary_roots=buckets.secondary_roots,
                base_url=url,
            )

        candidates: list[CandidateDoc] = []
        warnings: list[str] = []
        engine_chain: list[str] = []

        fast = run_fastdom(
            buckets=buckets,
            profile=profile,
            base_url=url,
            include_secondary_content=include_secondary_content,
        )
        candidates.append(fast)
        engine_chain.append(fast.extractor_used)

        need_fallback = fast.quality_score < float(profile.quality_threshold) or int(
            fast.primary_chars
        ) < int(profile.min_primary_chars)
        if need_fallback:
            for name in profile.engine_order:
                if name == "fastdom":
                    continue

                cand, warn = self._run_engine(
                    name=name,
                    html_doc=html_doc,
                    profile=profile,
                    base_url=url,
                    include_secondary_content=include_secondary_content,
                    canonical_secondary_markdown=canonical_secondary_markdown,
                )
                if warn:
                    warnings.append(warn)
                if cand is None:
                    continue
                candidates.append(cand)
                engine_chain.append(cand.extractor_used)

        best = max(
            candidates,
            key=lambda item: (float(item.quality_score), len(item.plain_text)),
        )
        best = self._enhance_for_missing_features(
            best=best,
            candidates=candidates,
            profile=profile,
        )

        markdown = finalize_markdown(
            markdown=best.markdown,
            max_chars=profile.max_markdown_chars,
        )
        plain = markdown_to_plain(markdown)

        merged_stats: StatsMap = dict(best.stats)
        inferred = infer_markdown_stats(markdown)
        merged_stats.update(
            {
                "markdown_chars": len(markdown),
                "plain_chars": len(plain),
                "candidate_count": len(candidates),
                "engine_chain": "->".join(engine_chain),
                "include_secondary_content": bool(include_secondary_content),
                "primary_chars": int(best.primary_chars),
                "secondary_chars": int(
                    best.secondary_chars if include_secondary_content else 0
                ),
            }
        )
        merged_stats.update(
            {
                "heading_count": int(inferred.get("heading_count", 0)),
                "table_count": int(inferred.get("table_count", 0)),
                "table_row_count": int(inferred.get("table_row_count", 0)),
                "code_block_count": int(inferred.get("code_block_count", 0)),
                "inline_code_count": int(inferred.get("inline_code_count", 0)),
                "list_count": int(inferred.get("list_count", 0)),
                "ordered_list_count": int(inferred.get("ordered_list_count", 0)),
                "link_count": int(inferred.get("link_count", 0)),
                "fence_pairs_ok": bool(inferred.get("fence_pairs_ok", True)),
            }
        )

        merged_warnings = self._merge_warnings(candidates, extra=warnings)
        if include_secondary_content:
            if len(plain) < int(profile.min_total_chars_with_secondary):
                merged_warnings.append("extracted text is short with secondary content")
        elif int(best.primary_chars) < int(profile.min_primary_chars):
            merged_warnings.append("primary content is short")

        quality = score_candidate(
            markdown=markdown,
            plain_text=plain,
            stats=merged_stats,
            warnings=merged_warnings,
        )

        links = (
            collect_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=include_secondary_content,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_links
            else []
        )
        image_links = (
            collect_image_links_inventory(
                soup=soup,
                base_url=url,
                buckets=buckets,
                include_secondary_content=include_secondary_content,
                max_links=profile.link_max_count,
                keep_hash=profile.link_keep_hash,
            )
            if collect_images
            else []
        )

        title = clean_whitespace(
            html.unescape(soup.title.get_text(" ", strip=True) if soup.title else "")
        )
        return ExtractedDocument(
            markdown=markdown,
            plain_text=plain,
            title=title,
            content_kind="html",
            extractor_used=best.extractor_used,
            quality_score=float(quality),
            warnings=merged_warnings,
            stats=merged_stats,
            links=links,
            image_links=image_links,
        )

    def _run_engine(
        self,
        *,
        name: str,
        html_doc: str,
        profile: ExtractProfile,
        base_url: str,
        include_secondary_content: bool,
        canonical_secondary_markdown: str,
    ) -> tuple[CandidateDoc | None, str | None]:
        if name == "readability":
            if not readability_available():
                return None, "readability unavailable"

            def run() -> CandidateDoc | None:
                return run_readability(
                    html_doc=html_doc,
                    base_url=base_url,
                    profile=profile,
                    include_secondary_content=include_secondary_content,
                    canonical_secondary_markdown=canonical_secondary_markdown,
                )

            return self._run_with_timeout(
                name=name,
                runner=run,
                timeout_ms=profile.engine_timeout_ms,
            )

        if name == "trafilatura":
            if not trafilatura_available():
                return None, "trafilatura unavailable"

            def run() -> CandidateDoc | None:
                return run_trafilatura(
                    html_doc=html_doc,
                    profile=profile,
                    include_secondary_content=include_secondary_content,
                    canonical_secondary_markdown=canonical_secondary_markdown,
                )

            return self._run_with_timeout(
                name=name,
                runner=run,
                timeout_ms=profile.engine_timeout_ms,
            )

        if name == "justext":
            if not justext_available():
                return None, "justext unavailable"

            def run() -> CandidateDoc | None:
                return run_justext(
                    html_doc=html_doc,
                    profile=profile,
                    include_secondary_content=include_secondary_content,
                    canonical_secondary_markdown=canonical_secondary_markdown,
                )

            return self._run_with_timeout(
                name=name,
                runner=run,
                timeout_ms=profile.engine_timeout_ms,
            )

        if name == "boilerpy3":
            if not boilerpy3_available():
                return None, "boilerpy3 unavailable"

            def run() -> CandidateDoc | None:
                return run_boilerpy3(
                    html_doc=html_doc,
                    profile=profile,
                    include_secondary_content=include_secondary_content,
                    canonical_secondary_markdown=canonical_secondary_markdown,
                )

            return self._run_with_timeout(
                name=name,
                runner=run,
                timeout_ms=profile.engine_timeout_ms,
            )

        return None, f"engine_unknown:{name}"

    def _run_with_timeout(
        self,
        *,
        name: str,
        runner: Callable[[], CandidateDoc | None],
        timeout_ms: int,
    ) -> tuple[CandidateDoc | None, str | None]:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner)
            try:
                cand = future.result(timeout=max(0.1, float(timeout_ms) / 1000.0))
                return cand, None
            except FuturesTimeoutError:
                return None, f"{name}_timeout"
            except Exception as exc:  # noqa: BLE001
                return None, f"{name}_failed:{type(exc).__name__}"

    def _enhance_for_missing_features(
        self,
        *,
        best: CandidateDoc,
        candidates: list[CandidateDoc],
        profile: ExtractProfile,
    ) -> CandidateDoc:
        markdown = best.markdown
        for feature in (
            "code_block_count",
            "table_count",
            "heading_count",
            "ordered_list_count",
        ):
            if int(best.stats.get(feature, 0)) > 0:
                continue

            donor = next(
                (
                    candidate
                    for candidate in sorted(
                        candidates,
                        key=lambda item: item.quality_score,
                        reverse=True,
                    )
                    if candidate is not best
                    and int(candidate.stats.get(feature, 0)) > 0
                ),
                None,
            )
            if donor is None:
                continue

            snippet = extract_feature_snippets(markdown=donor.markdown, feature=feature)
            if not snippet:
                continue

            markdown = merge_markdown(
                base=markdown,
                extra=snippet,
                max_chars=profile.max_markdown_chars,
            )

        plain = markdown_to_plain(markdown)
        stats: StatsMap = dict(best.stats)
        inferred = infer_markdown_stats(markdown)
        stats.update(
            {
                "heading_count": int(inferred.get("heading_count", 0)),
                "table_count": int(inferred.get("table_count", 0)),
                "table_row_count": int(inferred.get("table_row_count", 0)),
                "code_block_count": int(inferred.get("code_block_count", 0)),
                "inline_code_count": int(inferred.get("inline_code_count", 0)),
                "list_count": int(inferred.get("list_count", 0)),
                "ordered_list_count": int(inferred.get("ordered_list_count", 0)),
                "link_count": int(inferred.get("link_count", 0)),
                "fence_pairs_ok": bool(inferred.get("fence_pairs_ok", True)),
            }
        )
        quality = score_candidate(
            markdown=markdown,
            plain_text=plain,
            stats=stats,
            warnings=list(best.warnings),
        )
        return CandidateDoc(
            markdown=markdown,
            plain_text=plain,
            extractor_used=best.extractor_used,
            quality_score=float(quality),
            warnings=list(best.warnings),
            stats=stats,
            primary_chars=int(best.primary_chars),
            secondary_chars=int(best.secondary_chars),
            links=list(best.links),
        )

    def _merge_warnings(
        self,
        candidates: list[CandidateDoc],
        *,
        extra: list[str] | None = None,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for cand in candidates:
            for item in cand.warnings:
                normalized = str(item).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(normalized)

        for item in extra or []:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)

        return merged
