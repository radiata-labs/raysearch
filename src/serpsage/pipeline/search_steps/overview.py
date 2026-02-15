from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.normalize import clean_whitespace
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.app.request import SearchOverviewRequest
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, LLMClientBase
    from serpsage.core.runtime import Runtime


class SearchOverviewStep(PipelineStep[SearchStepContext]):
    span_name = "step.search_overview"

    def __init__(
        self,
        *,
        rt: Runtime,
        llm: LLMClientBase,
        cache: CacheBase,
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self._cache = cache
        self.bind_deps(llm, cache)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if not ctx.results:
            return ctx

        enabled, req = self._resolve_overview_request(ctx)
        if not enabled:
            return ctx

        profile = self.settings.search.overview
        model_cfg = self.settings.llm.resolve_model(profile.use_model)
        schema = (
            dict(req.json_schema)
            if req is not None and isinstance(req.json_schema, dict)
            else None
        )
        mode = "json" if schema is not None else "text"
        messages = self._build_messages(
            query=ctx.request.query,
            results=ctx.results,
            max_sources=int(profile.max_sources),
            max_abstracts_per_source=int(profile.max_abstracts_per_source),
            max_abstract_chars=int(profile.max_abstract_chars),
            max_prompt_chars=int(profile.max_prompt_chars),
            mode=mode,
            language_hint=str(profile.force_language or "auto"),
        )
        cache_ttl_s = int(profile.cache_ttl_s)
        cache_key: str | None = None
        if cache_ttl_s > 0:
            cache_key = hashlib.sha256(
                stable_json(
                    {
                        "mode": mode,
                        "use_model": str(profile.use_model),
                        "messages": messages,
                        "json_schema": schema,
                    }
                ).encode("utf-8")
            ).hexdigest()
            cached = await self._cache.aget(
                namespace="overview:search:v4", key=cache_key
            )
            if cached:
                try:
                    ctx.overview = json.loads(cached.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    span.add_event("overview.cache_corrupt")
                else:
                    span.set_attr("cache_hit", True)
                    return ctx
        span.set_attr("cache_hit", False)
        span.set_attr("overview_mode", mode)

        retries = max(0, int(profile.self_heal_retries))
        cur_messages = list(messages)
        for attempt in range(retries + 1):
            try:
                res = await self._llm.chat(
                    model=str(model_cfg.name),
                    messages=cur_messages,
                    schema=schema,
                    timeout_s=float(model_cfg.timeout_s),
                )
                if schema is None:
                    output_text = clean_whitespace(res.text or "")
                    if not output_text:
                        raise ValueError("overview output is empty")
                    ctx.overview = output_text
                else:
                    output_obj = _coerce_json_output(
                        result_data=res.data,
                        raw_text=res.text,
                    )
                    _validate_json_output(schema=schema, value=output_obj)
                    ctx.overview = output_obj
                if cache_ttl_s > 0 and cache_key and ctx.overview is not None:
                    await self._cache.aset(
                        namespace="overview:search:v4",
                        key=cache_key,
                        value=json.dumps(
                            ctx.overview,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8"),
                        ttl_s=cache_ttl_s,
                    )
                return ctx
            except _SchemaMismatchError as exc:
                if attempt < retries:
                    cur_messages = cur_messages + [_self_heal_message()]
                    continue
                ctx.errors.append(
                    AppError(
                        code="overview_schema_mismatch",
                        message=str(exc),
                        details={"stage": "search_overview", "fatal": False},
                    )
                )
                return ctx
            except Exception as exc:  # noqa: BLE001
                if attempt < retries and schema is not None:
                    cur_messages = cur_messages + [_self_heal_message()]
                    continue
                ctx.errors.append(
                    AppError(
                        code="overview_failed",
                        message=str(exc),
                        details={
                            "stage": "search_overview",
                            "fatal": False,
                            "type": type(exc).__name__,
                        },
                    )
                )
                return ctx
        return ctx

    def _resolve_overview_request(
        self, ctx: SearchStepContext
    ) -> tuple[bool, SearchOverviewRequest | None]:
        raw = ctx.request.overview
        if raw is None:
            enabled = bool(self.settings.search.overview.enabled_default)
            return enabled, None
        if isinstance(raw, bool):
            return bool(raw), None
        return True, raw

    def _build_messages(
        self,
        *,
        query: str,
        results: list[ResultItem],
        max_sources: int,
        max_abstracts_per_source: int,
        max_abstract_chars: int,
        max_prompt_chars: int,
        mode: str,
        language_hint: str,
    ) -> list[dict[str, str]]:
        blocks: list[str] = []
        for result in results[: max(1, int(max_sources))]:
            sid = result.source_id or "S?"
            lines: list[str] = []
            if result.title:
                lines.append(f"TITLE: {result.title}")
            if result.url:
                lines.append(f"URL: {result.url}")
            if result.snippet:
                lines.append(f"SNIPPET: {result.snippet}")
            if result.page and result.page.abstracts:
                for idx, item in enumerate(
                    result.page.abstracts[: max(1, int(max_abstracts_per_source))], 1
                ):
                    text = item.text
                    if max_abstract_chars > 0 and len(text) > max_abstract_chars:
                        text = text[:max_abstract_chars].rstrip()
                    aid = item.abstract_id or f"{sid}:A{idx}"
                    lines.append(f"ABSTRACT {aid}: {text}")
            blocks.append(f"[{sid}]\n" + "\n".join(lines))

        prompt = "\n\n".join(
            [
                f"QUERY:\n{query}",
                "SOURCES:\n" + "\n\n".join(blocks),
            ]
        )
        if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]

        lang_line = (
            "Respond in Simplified Chinese."
            if language_hint == "zh"
            else (
                "Respond in English."
                if language_hint == "en"
                else "Follow user query language."
            )
        )
        task = (
            "Return JSON only. Do not output markdown fences or additional text."
            if mode == "json"
            else "Return plain text only. Do not wrap output in JSON."
        )
        return [
            {
                "role": "system",
                "content": f"You are a research assistant. {lang_line} {task}",
            },
            {"role": "user", "content": prompt},
        ]


def _coerce_json_output(*, result_data: object | None, raw_text: str) -> object:
    if result_data is not None:
        return result_data
    text = clean_whitespace(raw_text or "")
    if not text:
        raise ValueError("json output is empty")
    return json.loads(text)


def _validate_json_output(*, schema: dict[str, Any], value: object) -> None:
    try:
        from jsonschema import Draft202012Validator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("jsonschema dependency is required") from exc
    try:
        Draft202012Validator(schema).validate(value)
    except Exception as exc:  # noqa: BLE001
        raise _SchemaMismatchError(str(exc)) from exc


class _SchemaMismatchError(Exception):
    pass


def _self_heal_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Output did not validate. Return JSON only that strictly matches "
            "the provided schema."
        ),
    }


__all__ = ["SearchOverviewStep"]
