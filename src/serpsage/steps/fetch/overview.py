from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.app.request import FetchOverviewRequest
from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace, stable_json

if TYPE_CHECKING:
    from serpsage.components.cache import CacheBase
    from serpsage.components.llm import LLMClientBase
    from serpsage.core.runtime import Runtime

class FetchOverviewStep(StepBase[FetchStepContext]):

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
        self, ctx: FetchStepContext
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        enabled, req = self._resolve_overview_request(ctx)
        if not enabled or req is None:
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
                        "crawl_mode": ctx.runtime.crawl_mode,
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
            title=(ctx.artifacts.extracted.title if ctx.artifacts.extracted else "")
            or "",
            source_items=source_items,
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
                    pass
                else:
                    ctx.artifacts.overview_output = decoded
                    return ctx

        retries = max(0, int(profile.self_heal_retries))
        for attempt in range(retries + 1):
            try:
                res = await self._llm.chat(
                    model=str(model_cfg.name),
                    messages=messages,
                    response_format=schema,
                    timeout_s=float(model_cfg.timeout_s),
                )
                if schema is None:
                    output_text = str(res.text or "")
                    if not output_text.strip():
                        raise ValueError("overview output is empty")
                    ctx.artifacts.overview_output = output_text
                else:
                    output_obj = _coerce_json_output(
                        result_data=res.data, raw_text=res.text
                    )
                    _validate_json_output(schema=schema, value=output_obj)
                    ctx.artifacts.overview_output = output_obj
                if (
                    cache_ttl_s > 0
                    and cache_key
                    and ctx.artifacts.overview_output is not None
                ):
                    await self._cache.aset(
                        namespace="overview:fetch:v4",
                        key=cache_key,
                        value=json.dumps(
                            ctx.artifacts.overview_output,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8"),
                        ttl_s=cache_ttl_s,
                    )
                return ctx
            except _SchemaMismatchError as exc:
                if attempt < retries:
                    messages = messages + [_self_heal_message()]
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
                            "crawl_mode": ctx.runtime.crawl_mode,
                        },
                    )
                )
                return ctx
            except Exception as exc:  # noqa: BLE001
                if attempt < retries and schema is not None:
                    messages = messages + [_self_heal_message()]
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
                            "crawl_mode": ctx.runtime.crawl_mode,
                        },
                    )
                )
                return ctx
        return ctx

    def _resolve_overview_request(
        self, ctx: FetchStepContext
    ) -> tuple[bool, FetchOverviewRequest | None]:
        raw = ctx.request.overview
        if isinstance(raw, bool):
            return bool(raw), (FetchOverviewRequest() if raw else None)
        return True, raw

    def _build_source_items(self, ctx: FetchStepContext) -> list[str]:
        abstracts = [
            item.text
            for item in list(ctx.artifacts.overview_scored_abstracts or [])
            if item.text
        ]
        if abstracts:
            return abstracts
        md_for_abstract = clean_whitespace(
            (
                (
                    ctx.artifacts.extracted.md_for_abstract
                    if ctx.artifacts.extracted
                    else ""
                )
                or ""
            ).replace("\n", " ")
        )
        if md_for_abstract:
            return [md_for_abstract]
        markdown = clean_whitespace(
            (
                (ctx.artifacts.extracted.markdown if ctx.artifacts.extracted else "")
                or ""
            ).replace("\n", " ")
        )
        if markdown:
            return [markdown]
        return []

    def _build_messages(
        self,
        *,
        query: str | None,
        url: str,
        title: str,
        source_items: list[str],
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
        normalized_query = clean_whitespace(query or "")
        query_block = normalized_query or (
            "No explicit user query was provided. Generate a neutral page overview "
            "grounded in TITLE and SOURCE_ABSTRACTS, focusing on key facts and context."
        )

        blocks = []
        for idx, item in enumerate(source_items, 1):
            blocks.append(f"[A{idx}] {item}")
        body = "\n".join(blocks)
        user = "\n\n".join(
            [
                f"QUERY:\n{query_block}",
                f"URL:\n{url}",
                f"TITLE:\n{title}",
                f"SOURCE_ABSTRACTS:\n{body}",
            ]
        )
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
    text = str(raw_text or "").strip()
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
