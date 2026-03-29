"""Demo: SearchRequest - comprehensive field showcase."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import anyio
from dotenv import load_dotenv

from raysearch import (
    Engine,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOthersRequest,
    FetchSubpagesRequest,
    SearchFetchRequest,
    SearchRequest,
)

load_dotenv()


async def main(
    query: str,
    user_location: str = "US",
    mode: Literal["fast", "auto", "deep"] = "auto",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Demonstrate SearchRequest.

    Args:
        query: Search query (required).
        user_location: ISO 3166-1 alpha-2 country code for localized results.
        mode: Search depth - fast/auto/deep.
        max_results: Maximum number of results.

    Returns:
        Dict containing search result as JSON.
    """
    # SearchRequest fields: query (required), user_location, mode, max_results,
    # additional_queries, start/end_published_date, include/exclude_domains,
    # include/exclude_text, moderation, fetchs
    req = SearchRequest(
        query=query,
        user_location=user_location,
        mode=mode,
        max_results=max_results,
        additional_queries=None,  # Extra queries (only for deep mode)
        start_published_date=None,  # ISO 8601 date for earliest publication
        end_published_date=None,  # ISO 8601 date for latest publication
        include_domains=None,  # Only include results from these domains
        exclude_domains=None,  # Exclude results from these domains
        include_text=None,  # Results must contain these text phrases
        exclude_text=None,  # Results must not contain these text phrases
        moderation=True,  # Enable content moderation filtering
        fetchs=SearchFetchRequest(
            crawl_mode="fallback",  # Options: never, fallback, preferred, always
            crawl_timeout=30.0,  # Timeout in seconds for crawling
            content=FetchContentRequest(
                detail="concise",  # Options: concise, standard, full
                max_chars=3000,  # Maximum characters for content extraction
            ),
            abstracts=FetchAbstractsRequest(
                query=query,  # Query for relevance-based abstract extraction
                max_chars=500,  # Maximum characters for abstracts
            ),
            overview=False,  # Enable overview extraction
            subpages=FetchSubpagesRequest(
                max_subpages=None,  # Optional: maximum subpages to fetch
                subpage_keywords=query,  # Keywords for subpage filtering
            ),
            others=FetchOthersRequest(
                max_links=5,  # Maximum related links to extract
                max_image_links=3,  # Maximum image links to extract
            ),
        ),
    )
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.search(req)
        t2 = time.time()
        print(f"Search took {t2 - t1:.4f} seconds")
    return {
        "search_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2),
    }


if __name__ == "__main__":
    # Classic search query: Latest developments in quantum computing
    out = anyio.run(
        main,
        "Latest developments in quantum computing 2024",
        "US",
        "fast",
        5,
    )
    print(out["search_result"])
