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
    FetchSubpagesRequest,
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
        content=FetchContentRequest(detail="full"),
        abstracts=FetchAbstractsRequest(query=query, max_chars=300) if query else False,
        overview=FetchOverviewRequest(query=query) if overview else False,
        subpages=FetchSubpagesRequest(max_subpages=2, subpage_keywords="Speciale"),
        others=FetchOthersRequest(max_links=5, max_image_links=5),
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.fetch(req)
    return {
        "fetch_result": json.dumps(
            resp.model_dump(exclude={"telemetry"}), ensure_ascii=False, indent=2
        )
    }


if __name__ == "__main__":
    out = anyio.run(
        main,
        "https://www.zenrows.com/blog/curl-cffi",
        "What is curl_cffi?",
        False,
    )
    print(out["fetch_result"])
