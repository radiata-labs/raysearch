from __future__ import annotations

import json
import time
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
)

load_dotenv()


async def main(
    urls: list[str],
    query: str | None = None,
    overview: bool = False,
) -> dict[str, Any]:
    req = FetchRequest(
        urls=urls,
        crawl_timeout=30,
        crawl_mode="fallback",
        content=FetchContentRequest(detail="full"),
        abstracts=FetchAbstractsRequest(query=query, max_chars=2000),
        overview=FetchOverviewRequest(query=query) if overview else False,
        subpages=FetchSubpagesRequest(max_subpages=None, subpage_keywords=None),
        others=FetchOthersRequest(max_links=5, max_image_links=5),
    )
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.fetch(req)
        t2 = time.time()
        print(f"Fetch took {t2 - t1:.4f} seconds")
    return {"fetch_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)}


if __name__ == "__main__":
    out = anyio.run(
        main,
        [
            "https://doi.org/10.3390/fi15060192",
        ],
        None,
        False,
    )
    print(out["fetch_result"])
