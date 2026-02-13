from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import Engine, FetchRequest, load_settings

load_dotenv()


async def main(
    url: str,
    query: str | None = None,
    overview: bool = False,
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = FetchRequest(
        url=url,
        query=query,
        overview=overview,
    )
    async with Engine.from_settings(settings) as engine:
        resp = await engine.fetch(req)
    return {
        "fetch_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2),
    }


if __name__ == "__main__":
    out = anyio.run(main, "https://en.wikipedia.org/wiki/Large_language_model", None, False)
    print(out["fetch_result"])
