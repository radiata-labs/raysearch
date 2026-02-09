from __future__ import annotations

import json
from typing import Any, Literal

import anyio

from serpsage import Engine, SearchRequest, load_settings


async def main(
    query: str,
    depth: Literal["simple", "low", "medium", "high"] = "low",
    max_results: int = 5,
    type: Literal["json", "markdown"] = "json",
) -> dict[str, Any]:
    settings = load_settings("src/search_config_example.yaml")
    req = SearchRequest(query=query, depth=depth, max_results=max_results)

    async with Engine.from_settings(settings) as engine:
        resp = await engine.run(req)

    if type == "json":
        return {"search_result": json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)}

    # Minimal markdown rendering (intentionally thin; renderers can be modularized later).
    lines: list[str] = ["# 网络搜索结果", "", f"## 用户问题\n{query}", "", "## 搜索结果"]
    for r in resp.results:
        sid = r.source_id or "S?"
        lines.append(f"### [{sid}] {r.title or '(no-title)'}")
        if r.domain:
            lines.append(f"- 来源: {r.domain}")
        if r.url:
            lines.append(f"- 链接: {r.url}")
        if r.published_date:
            lines.append(f"- 时间: {r.published_date}")
        if r.snippet:
            lines.append(f"- 内容: {r.snippet}")
        if r.page and r.page.chunks:
            lines.append("- 页面片段:")
            for ch in r.page.chunks:
                lines.append(f"  - {ch.text}")  # noqa: PERF401
        lines.append("")

    if resp.overview:
        lines.append("## AI Overview")
        if resp.overview.summary:
            lines.append(resp.overview.summary)
        if resp.overview.key_points:
            lines.append("")
            lines.append("### 要点")
            for p in resp.overview.key_points:
                lines.append(f"- {p}")  # noqa: PERF401
        lines.append("")

    return {"search_result": "\n".join(lines)}


if __name__ == "__main__":
    out = anyio.run(main, "2026 llm 最新 研究", "high", 5, "json")
    print(out["search_result"])
