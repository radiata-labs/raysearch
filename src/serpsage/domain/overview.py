from __future__ import annotations

from serpsage.app.response import OverviewResult, ResultItem
from serpsage.contracts.base import WorkUnit
from serpsage.overview.schema import overview_json_schema


class OverviewBuilder(WorkUnit):
    def schema(self) -> dict:
        return overview_json_schema()

    def build_messages(
        self, *, query: str, results: list[ResultItem]
    ) -> list[dict[str, str]]:
        max_sources = int(self.settings.overview.max_sources)
        max_chunks = int(self.settings.overview.max_chunks_per_source)
        max_chunk_chars = int(self.settings.overview.max_chunk_chars)

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

        user = "\n\n".join(
            [
                f"USER_QUERY:\n{query}",
                "SOURCES:\n" + "\n\n".join(sources),
                "TASK:\nWrite a concise overview. Provide key_points as short bullets. Add citations for key claims.",
                "CITATION_FORMAT:\nCitations must reference source_id (e.g. S1) and optionally chunk_id (e.g. S1:C1).",
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
