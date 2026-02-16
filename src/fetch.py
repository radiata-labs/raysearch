from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import (
    Engine,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
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
        content=FetchContentRequest(depth="high"),
        abstracts=FetchAbstractsRequest(query=query) if query else None,
        overview=FetchOverviewRequest(query=query) if overview and query else None,
        others=FetchOthersRequest(max_links=100, max_image_links=50),
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
        "https://arxiv.org/abs/2307.06435",
        None,
        False,
    )
    print(out["fetch_result"])
