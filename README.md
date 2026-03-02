# SerpSage

SerpSage is an async-only SERP + page intelligence engine with four first-class paths:

- `Engine.search(req)`: query search pipeline (`prepare -> optimize(optional) -> expand(deep) -> search(prefetch) -> fetch -> rank -> finalize`)
- `Engine.fetch(req)`: multi-URL pipeline (`prepare -> load(cache/crawl) -> extract -> optional abstract rank -> optional overview -> optional subpages -> finalize`)
- `Engine.answer(req)`: answer pipeline (`plan search params -> search -> generate answer`)
- `Engine.research(req)`: research pipeline (`multi-round search -> fetch -> abstract/content analysis -> markdown + structured`)

The `fetch` path is Markdown-first: output is centered on clean main-content markdown (`response.results[].content`).

## Public API

- `Engine`
- `SearchRequest` / `SearchResponse`
- `FetchRequest` / `FetchResponse`
- `AnswerRequest` / `AnswerResponse`
- `ResearchRequest` / `ResearchResponse`
- `load_settings`

## Configuration

Top-level settings:

- `provider`
- `http`
- `search`
- `answer`
- `research`
- `fetch`
- `rank`
- `llm`
- `cache`

Reference file: `demo/search_config_example.yaml`.

Env overrides:

- `SERPSAGE_CONFIG_PATH`
- `SEARXNG_BASE_URL`
- `SEARCH_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`

Runtime prerequisites:

- JS rendering uses Playwright. Install browsers with `playwright install`.
- Overview uses optional dependencies (`serpsage[overview]`) plus API keys.
- HTML->Markdown conversion uses an internal renderer (compat backend name remains `markdownify`).

Telemetry and metering (core component mode):

- This repo exposes pluggable telemetry/metering foundations for host-system integration.
- V2 is a breaking cut: legacy `response.errors` diagnostics are removed.
- Telemetry emits request/step/LLM/fetch events; metering emits usage events (`meter.*`).
- Metering ledger is optional and defaults to disabled (`null` backend).

Settings:

- `telemetry.enabled`: enable async event emission.
- `telemetry.queue_size`: bounded in-memory queue size.
- `telemetry.drop_noncritical_when_full`: drop low-priority events when queue is full.
- `telemetry.obs.backend`: `null|jsonl`.
- `telemetry.obs.jsonl_path`: JSONL output path when obs backend is `jsonl`.
- `telemetry.metering.backend`: `null|sqlite`.
- `telemetry.metering.sqlite_db_path`: SQLite ledger path for metering sink.

When `telemetry.metering.backend=sqlite`, an append-only table `metering_ledger` is created
with dedupe constraints on `event_id` and `idempotency_key`.

## Usage

```python
from serpsage import (
    AnswerRequest,
    Engine,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
    FetchSubpagesRequest,
    ResearchRequest,
    SearchRequest,
    load_settings,
)

settings = load_settings()

async with Engine.from_settings(settings) as engine:
    search_resp = await engine.search(
        SearchRequest(
            query="latest ai papers",
            mode="deep",
            max_results=8,
            fetchs={"content": True},
        )
    )
    fetch_resp = await engine.fetch(
        FetchRequest(
            urls=["https://example.com/article"],
            crawl_mode="fallback",
            crawl_timeout=2.5,
            content=FetchContentRequest(detail="standard"),
            abstracts=FetchAbstractsRequest(query="benchmark results", max_chars=2400),
            subpages=FetchSubpagesRequest(
                max_subpages=2,
                subpage_keywords="benchmark, evaluation",
            ),
            overview=FetchOverviewRequest(query="benchmark results"),
            others=FetchOthersRequest(max_links=20, max_image_links=10),
        )
    )
    answer_resp = await engine.answer(
        AnswerRequest(
            query="What are the latest benchmark results for model X?",
            content=True,
        )
    )
    research_resp = await engine.research(
        ResearchRequest(
            search_mode="research-pro",
            themes="Latest open-source coding LLM benchmark landscape",
            json_schema=None,
        )
    )
```

## Behavior notes

- `search.mode`: `fast|auto|deep`
- `search.mode=fast` keeps old `auto` behavior (no LLM query optimization, single-query retrieval path)
- `search.mode in {auto, deep}` enables a dedicated LLM query optimization step with explicit freshness handling
- `search.mode=deep` additionally enables mixed query expansion (manual/rule/LLM) and context-aware composite reranking
- deep search tuning is configured by `search.deep.*` (expansion limits, weights, prefetch budget, LLM model/timeout)
- `answer` planner is LLM-driven and receives current UTC time for recency-sensitive questions
- `answer` generation consumes budgeted abstracts (`answer.generate.max_abstract_chars`, default `3000`)
- `answer` uses `[citation:x]` markers in `answers`; only referenced pages appear in `citations`
- `research` request is simplified to `search_mode/themes/json_schema`
- `research` supports three modes: `research-fast|research|research-pro`
- `research` always returns standardized markdown (`content`) and optional structured output (`structured`)
- `research` model output uses `[citation:x]`, then post-processing rewrites to `[citation:url]`
- `answer.citations[]` is page-level unique (`url`, `title`, optional `content`), not abstract-level
- `AnswerRequest.content=true` enables page markdown output in search/fetch and citation `content` payload
- `fetch`: no depth tiers, single strategy
- `fetch.backend`: `auto|curl_cffi|playwright` (`httpx` backend removed)
- migration: old search profile/overview settings are removed
- `FetchRequest` V4 fields:
  - `urls`: list of URLs, processed concurrently with stable output order
  - `crawl_mode`: `never|fallback|preferred|always`
  - `crawl_timeout`: per-URL crawler timeout in seconds
  - `content`: `bool | FetchContentRequest`, default `false` (`false` hides output markdown only; internal extraction still runs)
  - `FetchContentRequest.detail`: `concise|standard|full` (mapped internally to original extraction depth behavior)
  - `abstracts`: `bool | FetchAbstractsRequest`, default `false` (`true` means enabled with default config)
  - `subpages`: `FetchSubpagesRequest | None`
  - `subpages.max_subpages`: required to enable subpages; `None` means disabled
  - `subpages.subpage_keywords`: one string or comma-separated keywords for ranking links
  - `overview`: `bool | FetchOverviewRequest`, default `false` (`true` means enabled with default config)
  - `FetchAbstractsRequest.query` / `FetchOverviewRequest.query` may be `None`; ranking query falls back to `title`, then `url`
  - when `overview.query` is `None`, the LLM query is fixed as `总结`
  - `others.max_links` / `others.max_image_links`: optional link collection caps; omitted means no links output
  - if `subpages` is enabled but `others.max_links` is omitted, fetch internally collects links for subpage ranking and may hide those links in final `others.links`
  - old fetch fields (`url/params/query/include_chunks/top_k_chunks/include_secondary_content/runtime`) are removed
- `search` does not include overview generation; overview remains fetch-only
- when deep query expansion fails, search aborts with error code `search_query_expansion_failed`
- `search` response shape: `search_mode/results`
- `answer`/`research` response payloads no longer include `errors`
- `fetch` response adds `statuses[]` (one item per input URL, ordered by input URL order, `success|error`)
- `fetch.statuses[].error`: present on `error` items, shape `{tag, detail}` where `tag` is one of
  `CRAWL_NOT_FOUND|CRAWL_TIMEOUT|CRAWL_LIVECRAWL_TIMEOUT|SOURCE_NOT_AVAILABLE|UNSUPPORTED_URL|CRAWL_UNKNOWN_ERROR`
- fetch/extract pipeline supports JS-rendered pages, PDF text extraction, and noisy layouts with boilerplate filtering
- `fetch.extract` uses an internal markdown renderer pipeline; no renderer backend toggle is exposed.
