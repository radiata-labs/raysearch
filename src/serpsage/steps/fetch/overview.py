from __future__ import annotations

import hashlib
import json
from typing import Any
from typing_extensions import override

from serpsage.components.cache import CacheBase
from serpsage.components.llm import LLMClientBase
from serpsage.dependencies import Depends
from serpsage.models.app.request import FetchOverviewRequest
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace, stable_json


class FetchOverviewStep(StepBase[FetchStepContext]):
    llm: LLMClientBase = Depends()
    cache: CacheBase = Depends()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        enabled, req = self._resolve_overview_request(ctx)
        if not enabled or req is None:
            return ctx
        source_items = self._build_source_items(ctx)
        if not source_items:
            await self.emit_tracking_event(
                event_name="fetch.overview.error",
                request_id=ctx.request_id,
                stage="overview",
                status="error",
                error_code="fetch_overview_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "no overview source content",
                },
            )
            return ctx
        profile = self.settings.fetch.overview
        schema = dict(req.json_schema) if isinstance(req.json_schema, dict) else None
        mode = "json" if schema is not None else "text"
        messages = self._build_messages(
            query=req.query,
            url=ctx.url,
            title=(ctx.page.doc.meta.title if ctx.page.doc else "") or "",
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
            cached = await self.cache.aget(namespace="overview:fetch:v4", key=cache_key)
            if cached:
                try:
                    decoded = json.loads(cached.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    pass
                else:
                    ctx.analysis.overview.output = decoded
                    return ctx
        retries = max(0, int(profile.self_heal_retries)) if schema is not None else 0
        retry_prompt = _self_heal_message() if schema is not None else ""
        retry_on = (ValueError, TypeError, RuntimeError)
        try:
            if schema is None:
                text_res = await self.llm.create(
                    model=self.settings.fetch.overview.use_model,
                    messages=messages,
                    response_format=schema,
                    retries=retries,
                    retry_on=retry_on,
                )
                output_text = str(text_res.text or "")
                if not output_text.strip():
                    raise ValueError("overview output is empty")
                ctx.analysis.overview.output = output_text
            else:
                attempt_messages = list(messages)
                output_obj: object | None = None
                attempts = max(1, retries + 1)
                for attempt_index in range(attempts):
                    try:
                        json_res = await self.llm.create(
                            model=self.settings.fetch.overview.use_model,
                            messages=attempt_messages,
                            response_format=schema,
                            retries=0,
                        )
                        output_obj = _coerce_json_output(
                            result_data=json_res.data, raw_text=json_res.text
                        )
                        _validate_json_output(schema=schema, value=output_obj)
                        break
                    except Exception as exc:  # noqa: BLE001
                        if attempt_index >= attempts - 1:
                            raise
                        if not isinstance(exc, retry_on + (SchemaMismatchError,)):
                            raise
                        attempt_messages = attempt_messages + [
                            {
                                "role": "user",
                                "content": retry_prompt,
                            }
                        ]
                if output_obj is None:
                    raise RuntimeError("json output retry loop exhausted")
                ctx.analysis.overview.output = output_obj
            if (
                cache_ttl_s > 0
                and cache_key
                and ctx.analysis.overview.output is not None
            ):
                await self.cache.aset(
                    namespace="overview:fetch:v4",
                    key=cache_key,
                    value=json.dumps(
                        ctx.analysis.overview.output,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8"),
                    ttl_s=cache_ttl_s,
                )
            return ctx
        except SchemaMismatchError as exc:
            await self.emit_tracking_event(
                event_name="fetch.overview.error",
                request_id=ctx.request_id,
                stage="overview",
                status="error",
                error_code="overview_schema_mismatch",
                error_type=type(exc).__name__,
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": str(exc),
                },
            )
            return ctx
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="fetch.overview.error",
                request_id=ctx.request_id,
                stage="overview",
                status="error",
                error_code="fetch_overview_failed",
                error_type=type(exc).__name__,
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": str(exc),
                },
            )
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
            item.text for item in list(ctx.analysis.overview.ranked or []) if item.text
        ]
        if abstracts:
            return abstracts
        abstract_text = clean_whitespace(
            (
                (ctx.page.doc.content.abstract_text if ctx.page.doc else "") or ""
            ).replace("\n", " ")
        )
        if abstract_text:
            return [abstract_text]
        markdown = clean_whitespace(
            ((ctx.page.doc.content.markdown if ctx.page.doc else "") or "").replace(
                "\n", " "
            )
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
        from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("jsonschema dependency is required") from exc
    try:
        Draft202012Validator(schema).validate(value)
    except Exception as exc:  # noqa: BLE001
        raise SchemaMismatchError(str(exc)) from exc


class SchemaMismatchError(Exception):
    pass


def _self_heal_message() -> str:
    return (
        "Output did not validate. Return JSON only that strictly matches "
        "the provided schema."
    )


__all__ = ["FetchOverviewStep"]
