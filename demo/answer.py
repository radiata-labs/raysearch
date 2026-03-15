from __future__ import annotations

import json
from typing import Any

import anyio
from dotenv import load_dotenv

from serpsage import AnswerRequest, Engine

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
        resp = await engine.answer(req)
    return {
        "answer": resp.answer,
        "answer_result": json.dumps(
            resp.model_dump(),
            ensure_ascii=False,
            indent=2,
        ),
    }


if __name__ == "__main__":
    import time

    t1 = time.time()
    out = anyio.run(main, "What is keepalive in http?", False, None)
    t2 = time.time()
    print(out["answer_result"])
    print(f"Answer: {out['answer']}")
