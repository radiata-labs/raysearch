from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from serpsage.app.response import (
    OverviewLLMOutput,
    OverviewResult,
    ResultItem,
)
from serpsage.components.overview.schema import overview_json_schema
from serpsage.core.workunit import WorkUnit
from serpsage.models.llm import LLMUsage

if TYPE_CHECKING:
    from serpsage.contracts.services import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.llm import ChatJSONResult

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


class OverviewBuilder(WorkUnit):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

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
            parts: list[str] = []
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
                "Write a concise overview in Simplified Chinese (JSON). "
                "summary is a short paragraph; key_points is a list of short bullets; "
                "add citations for key claims."
            )
        else:
            task = (
                "TASK:\n"
                "Write a concise overview in English (JSON). "
                "summary is a short paragraph; key_points is a list of short bullets; "
                "add citations for key claims."
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

    async def build_overview(
        self, *, query: str, results: list[ResultItem]
    ) -> OverviewResult:
        llm_cfg = self.settings.overview.llm
        model = llm_cfg.model

        messages = self.build_messages(query=query, results=results)
        schema = self.schema()

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        retries = max(0, int(self.settings.overview.self_heal_retries))
        cur_messages = list(messages)

        with self.span("overview.build", model=model) as sp:
            sp.set_attr("schema_strict", bool(self.settings.overview.schema_strict))
            sp.set_attr("prompt_chars", int(prompt_chars))
            sp.set_attr("self_heal_retries", int(retries))

            last_exc: Exception | None = None
            last_res: ChatJSONResult | None = None

            for attempt in range(retries + 1):
                sp.set_attr("attempt", int(attempt))
                try:
                    last_res = await self._llm.chat_json(
                        model=model,
                        messages=cur_messages,
                        schema=schema,
                        timeout_s=float(llm_cfg.timeout_s),
                    )
                    out = OverviewLLMOutput.model_validate(last_res.data)
                    out = self._sanitize_overview(out, results)
                    summary = self._truncate_summary(out.summary)

                    return OverviewResult(
                        summary=summary,
                        key_points=list(out.key_points or []),
                        citations=list(out.citations or []),
                        usage=LLMUsage(
                            prompt_tokens=int(last_res.usage.prompt_tokens),
                            completion_tokens=int(last_res.usage.completion_tokens),
                            total_tokens=int(last_res.usage.total_tokens),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = (
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
                    sp.add_event(
                        "overview.attempt_failed",
                        attempt=int(attempt),
                        error_type=type(exc).__name__,
                    )
                    if attempt < retries and _healable(exc):
                        cur_messages = cur_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "Your previous output did not validate. "
                                    "Return JSON only that strictly matches the given schema. "
                                    "Do not include markdown or extra keys."
                                ),
                            }
                        ]
                        continue
                    raise

            assert last_exc is not None
            raise last_exc

    def _truncate_summary(self, summary: str) -> str:
        max_tokens = int(self.settings.overview.max_output_tokens or 0)
        if max_tokens <= 0:
            return summary or ""
        return _truncate_to_token_budget(summary or "", max_tokens=max_tokens)

    def _sanitize_overview(
        self, overview: OverviewLLMOutput, results: list[ResultItem]
    ) -> OverviewLLMOutput:
        sid_to_url: dict[str, str] = {}
        sid_to_chunks: dict[str, set[str]] = {}
        for r in results:
            sid = (r.source_id or "").strip()
            if not sid:
                continue
            if r.url:
                sid_to_url[sid] = r.url
            chunks = set()
            if r.page and r.page.chunks:
                for ch in r.page.chunks:
                    if ch.chunk_id:
                        chunks.add(ch.chunk_id)
            sid_to_chunks[sid] = chunks

        kept = []
        seen: set[tuple[str, str, str]] = set()
        for c in overview.citations or []:
            sid = (c.source_id or "").strip()
            if not sid or sid not in sid_to_chunks:
                continue

            url = sid_to_url.get(sid) or (c.url or "")
            chunk_id = c.chunk_id
            if chunk_id and chunk_id not in sid_to_chunks.get(sid, set()):
                chunk_id = None

            key = (sid, chunk_id or "", (c.quote or "").strip())
            if key in seen:
                continue
            seen.add(key)
            kept.append(
                c.model_copy(
                    update={
                        "url": url,
                        "chunk_id": chunk_id,
                    }
                )
            )

        renum = [
            c.model_copy(update={"cite_id": f"C{i}"}) for i, c in enumerate(kept, 1)
        ]
        return overview.model_copy(update={"citations": renum})


def _healable(exc: Exception) -> bool:
    return isinstance(exc, (ValidationError, TypeError, ValueError))


def _truncate_to_token_budget(text: str, *, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    if not text:
        return ""

    out_chars: list[str] = []
    tokens = 0
    i = 0
    n = len(text)

    def add_token(cnt: int) -> bool:
        nonlocal tokens
        if tokens + cnt > max_tokens:
            return False
        tokens += cnt
        return True

    while i < n:
        ch = text[i]
        if ch.isspace():
            out_chars.append(ch)
            i += 1
            continue

        if ch.isascii() and ch.isalnum():
            j = i + 1
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            run = text[i:j]
            need = (len(run) + 3) // 4
            if not add_token(need):
                break
            out_chars.append(run)
            i = j
            continue

        if _CJK_RE.fullmatch(ch):
            if not add_token(1):
                break
            out_chars.append(ch)
            i += 1
            continue

        if not add_token(1):
            break
        out_chars.append(ch)
        i += 1

    out = "".join(out_chars)
    if out == text:
        return out

    if out and tokens + 1 <= max_tokens:
        out = out + "..."
    return out


__all__ = ["OverviewBuilder"]
