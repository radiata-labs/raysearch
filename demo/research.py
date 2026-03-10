from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import Engine, ResearchRequest

load_dotenv()


async def main(
    themes: str,
    search_mode: str = "research",
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    req = ResearchRequest(
        search_mode=search_mode,  # type: ignore[arg-type]
        themes=themes,
        json_schema=json_schema,
    )
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        resp = await engine.research(req)
    return {
        "research_result": json.dumps(
            resp.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    }


if __name__ == "__main__":
    out = anyio.run(
        main,
        "qwen3.5和kimi-k2.5哪家强？给我综合比较报告。",
        "research-pro",
        None,
    )
    print(out["research_result"])
