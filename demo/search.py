from __future__ import annotations

import json
from typing import Any, Literal

import anyio
from dotenv import load_dotenv

from serpsage import (
    Engine,
    FetchAbstractsRequest,
    FetchOthersRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
    load_settings,
)

load_dotenv()


async def main(
    query: str,
    mode: Literal["fast", "auto", "deep"] = "deep",
    max_results: int = 5,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = SearchRequest(
        query=query,
        mode=mode,
        max_results=max_results,
        fetchs=FetchRequestBase(
            abstracts=FetchAbstractsRequest(max_chars=400),
            subpages=FetchSubpagesRequest(max_subpages=2, subpage_keywords=query),
            others=FetchOthersRequest(max_links=5, max_image_links=5),
        ),
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.search(req)
    return {
        "search_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2),
    }


if __name__ == "__main__":
    import time

    t1 = time.time()
    out = anyio.run(main, "latest ai papers", "auto", 5)
    t2 = time.time()
    print(out["search_result"])
    print(f"Search took {t2 - t1:.2f} seconds")
