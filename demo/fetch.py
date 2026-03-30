"""Demo: FetchRequest - comprehensive field showcase."""

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
    FetchOverviewRequest,
    FetchRequest,
    FetchSubpagesRequest,
)

load_dotenv()


async def main(
    urls: list[str],
    crawl_mode: Literal["never", "fallback", "preferred", "always"] = "fallback",
    crawl_timeout: float | None = None,
) -> dict[str, Any]:
    """
    Demonstrate FetchRequest.

    Args:
        urls: List of URLs to fetch (required).
        crawl_mode: Crawl strategy - never/fallback/preferred/always.
        crawl_timeout: Timeout in seconds for crawling.

    Returns:
        Dict containing fetch result as JSON.
    """
    # FetchRequest fields: urls (required), crawl_mode, crawl_timeout,
    # content, abstracts, overview, subpages, others
    req = FetchRequest(
        urls=urls,
        crawl_mode=crawl_mode,
        crawl_timeout=crawl_timeout,
        content=FetchContentRequest(
            detail="standard",  # Options: concise, standard, full
            max_chars=5000,  # Maximum characters for content extraction
            include_markdown_links=True,
            include_html_tags=False,
            include_tags=["body"],  # Tags to include in extraction
            exclude_tags=["header", "footer", "navigation"],  # Tags to exclude
        ),
        abstracts=FetchAbstractsRequest(
            query=None,  # Optional: query for relevance-based abstract extraction
            max_chars=2000,  # Maximum characters for abstracts
        ),
        overview=FetchOverviewRequest(
            query=None,  # Optional: query for overview extraction
        ),
        subpages=FetchSubpagesRequest(
            max_subpages=None,  # Optional: maximum number of subpages to fetch
            subpage_keywords=None,  # Optional: keywords for subpage filtering
        ),
        others=FetchOthersRequest(
            max_links=5,  # Optional: maximum related links to extract
            max_image_links=3,  # Optional: maximum image links to extract
        ),
    )
    async with Engine("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.fetch(req)
        t2 = time.time()
        print(f"Fetch took {t2 - t1:.4f} seconds")
    return {"fetch_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)}


if __name__ == "__main__":
    # Classic example: Wikipedia article about Python programming language
    out = anyio.run(
        main,
        ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        "fallback",
        60.0,
    )
    print(out["fetch_result"])
