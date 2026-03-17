"""Reddit specialized extractor for parsing Reddit JSON data.

Extracts structured content from Reddit's .json endpoint including:
- Post metadata (title, author, subreddit, score, comment count)
- Post body content
- Comments in threaded format

This extractor only handles JSON content from Reddit's .json API endpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, ClassVar
from typing_extensions import override
from urllib.parse import urlparse

from serpsage.components.extract.base import (
    ExtractConfigBase,
    SpecializedExtractorBase,
)
from serpsage.models.components.extract import (
    ExtractContent,
    ExtractContentTag,
    ExtractedDocument,
    ExtractMeta,
    ExtractRef,
    ExtractRefs,
    ExtractSpec,
    ExtractTrace,
)

# Reddit domains
_REDDIT_DOMAINS: frozenset[str] = frozenset(
    {
        "reddit.com",
        "www.reddit.com",
        "old.reddit.com",
        "new.reddit.com",
    }
)


class RedditExtractorConfig(ExtractConfigBase):
    __setting_family__ = "extract"
    __setting_name__ = "reddit"


@dataclass(slots=True)
class RedditMeta:
    """Extracted Reddit metadata."""

    title: str = ""
    author: str = ""
    subreddit: str = ""
    score: int = 0
    upvote_ratio: float = 0.0
    comment_count: int = 0
    flair: str = ""
    published_date: str = ""


class RedditExtractor(SpecializedExtractorBase[RedditExtractorConfig]):
    """Specialized extractor for Reddit JSON data.

    Extracts content from Reddit's .json endpoint:
    - Post content (title + body)
    - Comments with threaded structure
    """

    _DETAIL_DEFAULT_TAGS: ClassVar[dict[str, tuple[ExtractContentTag, ...]]] = {
        "concise": ("metadata", "body"),
        "standard": ("metadata", "body"),
        "full": ("metadata", "body"),
    }

    @classmethod
    @override
    def can_handle(
        cls,
        *,
        url: str,
        content_type: str | None,
        content: bytes | None = None,
    ) -> bool:
        # Only handle JSON content from Reddit
        if not content_type or "json" not in content_type.lower():
            return False

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if ":" in domain:
                domain = domain.split(":")[0]
            return domain in _REDDIT_DOMAINS
        except Exception:  # noqa: BLE001
            return False

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None = None,
        content_options: ExtractSpec | None = None,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        """Extract content from Reddit JSON endpoint data."""
        options = content_options or ExtractSpec()
        selected_tags = self._resolve_tags(options)

        return await self._extract_from_json(
            url=url,
            content=content,
            options=options,
            selected_tags=selected_tags,
            collect_links=collect_links,
            collect_images=collect_images,
        )

    async def _extract_from_json(
        self,
        *,
        url: str,
        content: bytes,
        options: ExtractSpec,
        selected_tags: set[ExtractContentTag],
        collect_links: bool,
        collect_images: bool,
    ) -> ExtractedDocument:
        """Extract content from Reddit JSON endpoint data."""
        data = json.loads(content.decode("utf-8", errors="replace"))

        # Parse Reddit JSON: [post_data, comments_data]
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError("Invalid Reddit JSON format")

        post_data = data[0].get("data", {}).get("children", [])
        comments_data = data[1].get("data", {}).get("children", [])

        if not post_data:
            raise ValueError("No post data in Reddit JSON")

        # Extract post info
        post = post_data[0].get("data", {})
        reddit_meta = self._extract_meta_from_post_json(post)

        # Build markdown
        markdown = self._build_markdown_from_json(
            post=post,
            comments=comments_data,
            options=options,
        )

        stats = self._build_stats(
            reddit_meta=reddit_meta,
            markdown=markdown,
            detail=options.detail,
            selected_tags=selected_tags,
        )

        links: list[ExtractRef] = []
        images: list[ExtractRef] = []
        if collect_links:
            links = self._extract_links_from_json(data)
        if collect_images:
            images = self._extract_images_from_json(post)

        return self._finalize_content(
            doc=ExtractedDocument(
                content=ExtractContent(markdown=markdown),
                meta=ExtractMeta(
                    title=reddit_meta.title or url,
                    author=reddit_meta.author,
                    published_date=reddit_meta.published_date,
                    favicon="https://www.reddit.com/favicon.ico",
                ),
                refs=ExtractRefs(links=links, images=images),
                trace=ExtractTrace(
                    kind="json",
                    engine="reddit:json_endpoint",
                    stats=stats,
                ),
            ),
            content_options=options,
        )

    def _extract_meta_from_post_json(self, post: dict[str, Any]) -> RedditMeta:
        """Extract metadata from Reddit post JSON data."""
        meta = RedditMeta()
        meta.title = str(post.get("title", "") or "")
        meta.author = f"u/{post.get('author', '')}" if post.get("author") else ""
        meta.subreddit = (
            f"r/{post.get('subreddit', '')}" if post.get("subreddit") else ""
        )
        meta.score = int(post.get("score", 0) or 0)
        meta.upvote_ratio = float(post.get("upvote_ratio", 0) or 0)
        meta.comment_count = int(post.get("num_comments", 0) or 0)
        meta.flair = str(post.get("link_flair_text", "") or "")
        meta.published_date = self._format_timestamp(post.get("created_utc"))
        return meta

    def _build_markdown_from_json(
        self,
        *,
        post: dict[str, Any],
        comments: list[dict[str, Any]],
        options: ExtractSpec,
    ) -> str:
        """Build markdown from Reddit JSON data."""
        parts: list[str] = []

        # Metadata section
        meta = self._extract_meta_from_post_json(post)
        meta_lines = [
            f"- **subreddit**: {meta.subreddit}",
            f"- **author**: {meta.author}",
            f"- **score**: {meta.score} points",
        ]
        if meta.upvote_ratio:
            meta_lines.append(f"- **upvote_ratio**: {meta.upvote_ratio:.0%}")
        if meta.comment_count:
            meta_lines.append(f"- **comments**: {meta.comment_count}")
        if meta.published_date:
            meta_lines.append(f"- **posted**: {meta.published_date}")
        if meta.flair:
            meta_lines.append(f"- **flair**: {meta.flair}")

        parts.append("## Metadata")
        parts.append("")
        parts.extend(meta_lines)
        parts.append("")

        # Post content
        parts.append(f"# {meta.title}")
        parts.append("")

        selftext = str(post.get("selftext", "") or "").strip()
        if selftext:
            parts.append(selftext)
            parts.append("")

        # Comments
        if options.detail in ("standard", "full"):
            parts.append("---")
            parts.append("")
            parts.append("## Comments")
            parts.append("")

            comment_md = self._format_comments_json(comments, depth=0, max_depth=3)
            if comment_md:
                parts.append(comment_md)

        return "\n".join(parts)

    def _format_comments_json(
        self,
        comments: list[dict[str, Any]],
        *,
        depth: int,
        max_depth: int,
    ) -> str:
        """Format Reddit comments from JSON data."""
        if depth >= max_depth:
            return ""

        parts: list[str] = []
        for item in comments:
            kind = item.get("kind", "")
            if kind != "t1":  # t1 is comment type
                continue

            data = item.get("data", {})
            if not isinstance(data, dict):
                continue

            author = str(data.get("author", "[deleted]") or "[deleted]")
            body = str(data.get("body", "") or "").strip()
            score = int(data.get("score", 0) or 0)

            # Skip empty or deleted comments
            if not body or body == "[deleted]":
                continue

            # Format comment
            indent = "> " * depth
            header = f"{indent}### **{author}** ({score} pts)"
            parts.append(header)
            parts.append("")

            # Format body with proper indentation
            for line in body.split("\n"):
                if line.strip():
                    parts.append(f"{indent}{line}")
                else:
                    parts.append("")

            parts.append("")

            # Process replies
            replies = data.get("replies", {})
            if isinstance(replies, dict):
                children = replies.get("data", {}).get("children", [])
                if children and depth < max_depth - 1:
                    reply_md = self._format_comments_json(
                        children, depth=depth + 1, max_depth=max_depth
                    )
                    if reply_md:
                        parts.append(reply_md)

        return "\n".join(parts)

    def _extract_links_from_json(
        self, data: list[dict[str, Any]]
    ) -> list[ExtractRef]:
        """Extract links from Reddit JSON."""
        links: list[ExtractRef] = []
        seen: set[str] = set()

        # Extract from post URL if it's a link post
        if len(data) >= 1:
            post_children = data[0].get("data", {}).get("children", [])
            if post_children:
                post_data = post_children[0].get("data", {})
                post_url = str(post_data.get("url", "") or "")
                if (
                    post_url
                    and post_url.startswith("http")
                    and "reddit.com" not in post_url.lower()
                ):
                    if post_url not in seen:
                        seen.add(post_url)
                        links.append(ExtractRef(url=post_url, text="original link"))

        # Extract from comments
        if len(data) >= 2:
            comments = data[1].get("data", {}).get("children", [])
            self._extract_links_from_comments(comments, links, seen)

        return links[: self.config.link_max_count]

    def _extract_links_from_comments(
        self,
        comments: list[dict[str, Any]],
        links: list[ExtractRef],
        seen: set[str],
    ) -> None:
        """Recursively extract links from comments."""
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

        for item in comments:
            if item.get("kind") != "t1":
                continue
            data = item.get("data", {})
            if not isinstance(data, dict):
                continue

            body = str(data.get("body", "") or "")
            for match in url_pattern.finditer(body):
                url = match.group()
                if url not in seen:
                    seen.add(url)
                    links.append(ExtractRef(url=url, text=""))

            # Process replies
            replies = data.get("replies", {})
            if isinstance(replies, dict):
                children = replies.get("data", {}).get("children", [])
                self._extract_links_from_comments(children, links, seen)

    def _extract_images_from_json(
        self, post: dict[str, Any]
    ) -> list[ExtractRef]:
        """Extract images from Reddit post JSON."""
        images: list[ExtractRef] = []
        seen: set[str] = set()

        # Check for preview images
        preview = post.get("preview", {})
        if isinstance(preview, dict):
            images_list = preview.get("images", [])
            if isinstance(images_list, list):
                for img in images_list:
                    source = img.get("source", {})
                    if isinstance(source, dict):
                        url = str(source.get("url", "") or "")
                        if url and url not in seen:
                            seen.add(url)
                            images.append(ExtractRef(url=url, text=""))

        # Check for thumbnail
        thumbnail = str(post.get("thumbnail", "") or "")
        if thumbnail.startswith("http") and thumbnail not in seen:
            seen.add(thumbnail)
            images.append(ExtractRef(url=thumbnail, text="thumbnail"))

        return images[: self.config.link_max_count]

    def _resolve_tags(self, options: ExtractSpec) -> set[ExtractContentTag]:
        if options.sections:
            selected = set(options.sections)
        else:
            selected = set(
                self._DETAIL_DEFAULT_TAGS.get(options.detail, ("metadata", "body"))
            )
        return selected

    def _build_stats(
        self,
        *,
        reddit_meta: RedditMeta,
        markdown: str,
        detail: str,
        selected_tags: set[ExtractContentTag],
    ) -> dict[str, int | float | str | bool]:
        """Build stats dictionary."""
        stats: dict[str, int | float | str | bool] = {
            "primary_chars": len(markdown),
            "text_chars": len(markdown.replace("```", "").replace("`", "")),
            "detail": detail,
            "selected_sections": ",".join(sorted(selected_tags)),
            "engine": "reddit_json",
        }
        if reddit_meta.subreddit:
            stats["subreddit"] = reddit_meta.subreddit
        if reddit_meta.score:
            stats["score"] = reddit_meta.score
        if reddit_meta.comment_count:
            stats["comment_count"] = reddit_meta.comment_count
        return stats

    def _format_timestamp(self, ts: Any) -> str:
        """Convert timestamp to ISO string."""
        if ts is None:
            return ""
        try:
            import datetime

            if isinstance(ts, str):
                ts = float(ts)
            dt = datetime.datetime.fromtimestamp(float(ts), tz=datetime.UTC)
            return dt.isoformat()
        except (ValueError, TypeError):
            return str(ts) if ts else ""


__all__ = ["RedditExtractor", "RedditExtractorConfig", "RedditMeta"]
