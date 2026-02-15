from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.utils import clean_whitespace, stable_json

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, LLMClientBase
    from serpsage.core.runtime import Runtime


class FetchOverviewStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_overview"

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
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        req = ctx.overview_request
        if req is None:
            return ctx

        source_items = self._build_source_items(ctx)
        if not source_items:
            ctx.errors.append(
                AppError(
                    code="fetch_overview_failed",
                    message="no overview source content",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "overview",
                        "fatal": False,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        profile = self.settings.fetch.overview
        model_cfg = self.settings.llm.resolve_model(profile.use_model)
        schema = dict(req.json_schema) if isinstance(req.json_schema, dict) else None
        mode = "json" if schema is not None else "text"
        messages = self._build_messages(
            query=req.query,
            url=ctx.url,
            title=(ctx.extracted.title if ctx.extracted else "") or "",
            source_items=source_items,
            max_prompt_chars=int(profile.max_prompt_chars),
            language_hint=str(profile.force_language or "auto"),
            mode=mode,
        )
        cache_key: str | None = None
        cache_ttl_s = int(profile.cache_ttl_s)
        if cache_ttl_s > 0:
            payload = {
                "mode": mode,
                "use_model": str(profile.use_model),
                "messages": messages,
                "json_schema": schema,
            }
            cache_key = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
            cached = await self._cache.aget(
                namespace="overview:fetch:v4", key=cache_key
            )
            if cached:
                try:
                    decoded = json.loads(cached.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    span.add_event("overview.cache_corrupt")
                else:
                    ctx.overview_output = decoded
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
                    ctx.overview_output = output_text
                else:
                    output_obj = _coerce_json_output(
                        result_data=res.data, raw_text=res.text
                    )
                    _validate_json_output(schema=schema, value=output_obj)
                    ctx.overview_output = output_obj
                if cache_ttl_s > 0 and cache_key and ctx.overview_output is not None:
                    await self._cache.aset(
                        namespace="overview:fetch:v4",
                        key=cache_key,
                        value=json.dumps(
                            ctx.overview_output,
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
                        details={
                            "url": ctx.url,
                            "url_index": ctx.url_index,
                            "stage": "overview",
                            "fatal": False,
                            "crawl_mode": ctx.others_runtime.crawl_mode,
                        },
                    )
                )
                return ctx
            except Exception as exc:  # noqa: BLE001
                if attempt < retries and schema is not None:
                    cur_messages = cur_messages + [_self_heal_message()]
                    continue
                ctx.errors.append(
                    AppError(
                        code="fetch_overview_failed",
                        message=str(exc),
                        details={
                            "type": type(exc).__name__,
                            "url": ctx.url,
                            "url_index": ctx.url_index,
                            "stage": "overview",
                            "fatal": False,
                            "crawl_mode": ctx.others_runtime.crawl_mode,
                        },
                    )
                )
                return ctx
        return ctx

    def _build_source_items(self, ctx: FetchStepContext) -> list[str]:
        abstracts = [
            item.text for item in list(ctx.scored_abstracts or []) if item.text
        ]
        if abstracts:
            return abstracts
        plain = clean_whitespace(
            (ctx.extracted.plain_text if ctx.extracted else "") or ""
        )
        if plain:
            return [plain]
        markdown = clean_whitespace(
            ((ctx.extracted.markdown if ctx.extracted else "") or "").replace("\n", " ")
        )
        if markdown:
            return [markdown]
        return []

    def _build_messages(
        self,
        *,
        query: str,
        url: str,
        title: str,
        source_items: list[str],
        max_prompt_chars: int,
        language_hint: str,
        mode: str,
    ) -> list[dict[str, str]]:
        lang_line = (
            "Respond in Simplified Chinese."
            if language_hint == "zh"
            else (
                "Respond in English."
                if language_hint == "en"
                else "Follow user query language."
            )
        )
        if mode == "json":
            task = (
                "Return JSON only. Do not output markdown fences or additional text. "
                "Your JSON must validate against the provided schema."
            )
        else:
            task = "Return plain text only. Do not wrap output in JSON."

        blocks = []
        for idx, item in enumerate(source_items, 1):
            blocks.append(f"[A{idx}] {item}")
        body = "\n".join(blocks)
        user = "\n\n".join(
            [
                f"QUERY:\n{query}",
                f"URL:\n{url}",
                f"TITLE:\n{title}",
                f"SOURCE_ABSTRACTS:\n{body}",
            ]
        )
        if max_prompt_chars > 0 and len(user) > max_prompt_chars:
            user = user[:max_prompt_chars]
        return [
            {
                "role": "system",
                "content": f"You are a research assistant. {lang_line} {task}",
            },
            {"role": "user", "content": user},
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


__all__ = ["FetchOverviewStep"]
