from __future__ import annotations

import re

from serpsage.app.response import OverviewResult, ResultItem
from serpsage.contracts.base import WorkUnit
from serpsage.overview.schema import overview_json_schema

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


class OverviewBuilder(WorkUnit):
    def schema(self) -> dict:
        return overview_json_schema()

    def build_messages(
        self, *, query: str, results: list[ResultItem]
    ) -> list[dict[str, str]]:
        max_sources = int(self.settings.overview.max_sources)
        max_chunks = int(self.settings.overview.max_chunks_per_source)
        max_chunk_chars = int(self.settings.overview.max_chunk_chars)
        max_prompt_chars = int(self.settings.overview.max_prompt_chars)

        lang = self.settings.overview.force_language
        if lang == "auto":
            lang = "zh" if _CJK_RE.search(query or "") else "en"

        sources: list[str] = []
        for r in results[:max_sources]:
            sid = r.source_id or "S?"
            parts = []
            if r.title:
                parts.append(f"TITLE: {r.title}")
            if r.url:
                parts.append(f"URL: {r.url}")
            if r.snippet:
                parts.append(f"SNIPPET: {r.snippet}")
            if r.page and r.page.chunks:
                for i, ch in enumerate(r.page.chunks[:max_chunks], 1):
                    t = ch.text
                    if max_chunk_chars and len(t) > max_chunk_chars:
                        t = t[:max_chunk_chars].rstrip() + "..."
                    cid = ch.chunk_id or f"{sid}:C{i}"
                    parts.append(f"CHUNK {cid}: {t}")
            sources.append(f"[{sid}]\n" + "\n".join(parts))

        if lang == "zh":
            task = (
                "TASK:\n"
                "用简洁中文输出 overview(JSON)。summary 一段话；key_points 为短要点列表；"
                "为关键结论添加 citations。"
            )
            cite_fmt = (
                "CITATION_RULES:\n"
                "- citations[].cite_id 必须唯一，例如 C1/C2...\n"
                "- citations[].source_id 必须引用提供的 SOURCES（例如 S1）\n"
                "- citations[].chunk_id 可选，但若提供必须存在（例如 S1:C1）\n"
                "- citations[].url 必须与 source_id 对应来源一致"
            )
        else:
            task = (
                "TASK:\n"
                "Write a concise overview in English (JSON). summary is a short paragraph; "
                "key_points is a list of short bullets; add citations for key claims."
            )
            cite_fmt = (
                "CITATION_RULES:\n"
                "- citations[].cite_id must be unique, e.g. C1/C2...\n"
                "- citations[].source_id must reference provided SOURCES (e.g. S1)\n"
                "- citations[].chunk_id is optional, but if present must exist (e.g. S1:C1)\n"
                "- citations[].url must match the referenced source_id"
            )

        # Budget: cap total prompt characters to avoid runaway token usage.
        # With small max_sources (default 8), a simple O(n^2) check is fine.
        kept: list[str] = []
        for s in sources:
            cand = kept + [s]
            user_cand = "\n\n".join(
                [
                    f"USER_QUERY:\n{query}",
                    "SOURCES:\n" + "\n\n".join(cand),
                    task,
                    cite_fmt,
                ]
            )
            if max_prompt_chars and len(user_cand) > max_prompt_chars:
                break
            kept = cand

        user = "\n\n".join(
            [
                f"USER_QUERY:\n{query}",
                "SOURCES:\n" + "\n\n".join(kept),
                task,
                cite_fmt,
            ]
        )
        return [
            {
                "role": "system",
                "content": "You are a research assistant. Output JSON only.",
            },
            {"role": "user", "content": user},
        ]

    def parse(self, data: dict) -> OverviewResult:
        return OverviewResult.model_validate(data)


__all__ = ["OverviewBuilder"]
