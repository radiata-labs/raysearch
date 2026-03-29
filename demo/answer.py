from __future__ import annotations

import json
import time
from typing import Any

import anyio
from dotenv import load_dotenv

from raysearch import AnswerRequest, Engine

load_dotenv()


async def main(
    query: str,
    content: bool = True,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    req = AnswerRequest(
        query=query,
        content=content,
        json_schema=json_schema,
    )
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        await anyio.sleep(1)
        t1 = time.time()
        resp = await engine.answer(req)
        t2 = time.time()
        print(f"Answer took {t2 - t1:.4f} seconds")
    return {
        "answer": resp.answer,
        "answer_result": json.dumps(
            resp.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
    }


if __name__ == "__main__":
    out = anyio.run(main, "r906有哪些歌", False, None)
    print(out["answer_result"])
    print(f"Answer: {out['answer']}")
