from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import (
    Engine,
    FetchChunksRequest,
    FetchOverviewRequest,
    FetchRequest,
    FetchRuntimeRequest,
    load_settings,
)

load_dotenv()


async def main(
    url: str,
    query: str | None = None,
    overview: bool = False,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = FetchRequest(
        urls=[url],
        crawl_mode="fallback",
        content=True,
        chunks=FetchChunksRequest(query=query) if query else None,
        overview=FetchOverviewRequest(query=(query or url)) if overview else None,
        runtime=FetchRuntimeRequest(max_links=100),
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.fetch(req)
    joined_content = "\n\n".join(item.content for item in resp.results if item.content)
    return {
        "fetch_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
        + "\n\n"
        + joined_content,
    }


if __name__ == "__main__":
    out = anyio.run(
        main,
        "https://exa.ai/docs/reference/search-best-practices",
        "Category Filters",
        False,
    )
    print(out["fetch_result"])
