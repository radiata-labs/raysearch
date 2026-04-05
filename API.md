# RaySearch API Documentation

This document describes the HTTP API endpoints, request/response models, and authentication requirements for the RaySearch personal API service.

## Overview

RaySearch provides an async-first search orchestration engine for AI-overview style workflows. The API exposes the following capabilities:

- **Search**: Multi-provider retrieval with optional fetch and rerank stages
- **Fetch**: Page crawling, extraction, abstracts, overview generation, and related links
- **Answer**: Grounded answer generation with citations
- **Research**: Multi-round research reports with synthesis and structured output

## Authentication

All endpoints except `/healthz` require Bearer token authentication.

```http
Authorization: Bearer <YOUR_API_KEY>
```

The token is configured via `api.bearer_token` in the config file or the `RAYSEARCH_API_KEY` environment variable.

## Base URL

Default: `http://localhost:8000`

Configure via `api.host` and `api.port` in settings.

---

## Endpoints

### Health Check

```http
GET /healthz
```

Returns service health status. No authentication required.

**Response**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status (`"ok"` or `"error"`) |
| `engine_ready` | boolean | Whether the engine is initialized |

---

### Search

```http
POST /v1/search
Authorization: Bearer <token>
```

Execute a search query across configured providers with optional content fetching.

**Request Body**

```json
{
  "query": "latest multimodal model papers",
  "user_location": "US",
  "mode": "deep",
  "max_results": 10,
  "start_published_date": "2024-01-01",
  "end_published_date": "2024-12-31",
  "include_domains": ["arxiv.org"],
  "exclude_domains": ["example.com"],
  "include_text": ["multimodal"],
  "exclude_text": ["advertisement"],
  "moderation": true,
  "additional_queries": ["vision language models", "multimodal reasoning"],
  "fetchs": {
    "crawl_mode": "fallback",
    "crawl_timeout": 30.0,
    "content": {
      "detail": "concise",
      "max_chars": 5000,
      "include_markdown_links": true
    },
    "abstracts": {
      "query": "multimodal models",
      "max_chars": 300
    },
    "subpages": {
      "max_subpages": 3,
      "subpage_keywords": [" methodology", "results"]
    },
    "overview": {
      "query": "summarize the key findings",
      "json_schema": {
        "type": "object",
        "properties": {
          "key_findings": {"type": "array"}
        }
      }
    },
    "others": {
      "max_links": 10,
      "max_image_links": 5
    }
  }
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query (cleaned, non-empty) |
| `user_location` | string | Yes | Two-letter ISO country code (e.g., `"US"`, `"JP"`, `"CN"`) |
| `mode` | string | No | Search mode: `"fast"`, `"auto"` (default), `"deep"` |
| `max_results` | integer | No | Maximum results to return (> 0) |
| `start_published_date` | string | No | ISO 8601 date string for filtering |
| `end_published_date` | string | No | ISO 8601 date string for filtering |
| `include_domains` | array | No | Domains to include (normalized, unique) |
| `exclude_domains` | array | No | Domains to exclude (normalized, unique) |
| `include_text` | array | No | Text phrases to include (max 5 words or 6 CJK chars per phrase) |
| `exclude_text` | array | No | Text phrases to exclude (max 5 words or 6 CJK chars per phrase) |
| `moderation` | boolean | No | Enable content moderation (default: `true`) |
| `additional_queries` | array | No | Additional queries (only supported when `mode="deep"`) |
| `fetchs` | object | Yes | Fetch configuration (see [Fetch Configuration](#fetch-configuration)) |

**Response**

```json
{
  "request_id": "abc123",
  "search_mode": "deep",
  "results": [
    {
      "url": "https://arxiv.org/abs/2401.00123",
      "title": "Multimodal Foundation Models",
      "published_date": "2024-01-15",
      "author": "John Doe",
      "image": "https://example.com/image.png",
      "favicon": "https://arxiv.org/favicon.ico",
      "content": "Abstract content...",
      "abstracts": ["Key finding 1", "Key finding 2"],
      "abstract_scores": [0.95, 0.88],
      "overview": {"key_findings": ["finding1", "finding2"]},
      "subpages": [
        {
          "url": "https://arxiv.org/abs/2401.00123/methodology",
          "title": "Methodology",
          "published_date": "",
          "author": "",
          "image": "",
          "favicon": "",
          "content": "Methodology content...",
          "abstracts": [],
          "abstract_scores": []
        }
      ],
      "others": {
        "links": ["https://related1.com", "https://related2.com"],
        "image_links": ["https://img1.png"]
      }
    }
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `search_mode` | string | The search mode used |
| `results` | array | Array of [FetchResultItem](#fetchresultitem) objects |

---

### Fetch

```http
POST /v1/fetch
Authorization: Bearer <token>
```

Fetch and process content from specified URLs.

**Request Body**

```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ],
  "crawl_mode": "fallback",
  "crawl_timeout": 30.0,
  "content": {
    "detail": "standard",
    "max_chars": 10000,
    "include_tags": ["body"],
    "exclude_tags": ["navigation", "footer"]
  },
  "abstracts": {
    "query": "main topic summary",
    "max_chars": 500
  },
  "subpages": {
    "max_subpages": 5,
    "subpage_keywords": ["details", "methodology"]
  },
  "overview": true,
  "others": {
    "max_links": 20
  }
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `urls` | array | Yes | List of URLs (must be `http://` or `https://`, non-empty) |
| `crawl_mode` | string | No | Crawl mode: `"never"`, `"fallback"` (default), `"preferred"`, `"always"` |
| `crawl_timeout` | float | No | Crawl timeout in seconds (> 0) |
| `content` | boolean/object | No | Content extraction config (at least one action required) |
| `abstracts` | boolean/object | No | Abstract generation config |
| `subpages` | object | No | Subpage extraction config |
| `overview` | boolean/object | No | Overview generation config |
| `others` | object | No | Related links extraction config |

**Response**

```json
{
  "request_id": "def456",
  "results": [
    {
      "url": "https://example.com/article1",
      "title": "Article Title",
      "published_date": "2024-03-15",
      "author": "Jane Smith",
      "image": "",
      "favicon": "https://example.com/favicon.ico",
      "content": "Full article content...",
      "abstracts": ["Summary point 1", "Summary point 2"],
      "abstract_scores": [0.92, 0.85],
      "overview": null,
      "subpages": [],
      "others": {
        "links": ["https://example.com/related"],
        "image_links": []
      }
    }
  ],
  "statuses": [
    {
      "url": "https://example.com/article1",
      "status": "success",
      "error": null
    },
    {
      "url": "https://example.com/article2",
      "status": "error",
      "error": {
        "tag": "CRAWL_TIMEOUT",
        "detail": "Page crawl exceeded timeout"
      }
    }
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `results` | array | Array of successful [FetchResultItem](#fetchresultitem) objects |
| `statuses` | array | Array of [FetchStatusItem](#fetchstatusitem) objects for each URL |

---

### Answer

```http
POST /v1/answer
Authorization: Bearer <token>
```

Generate a grounded answer with citations from provided content.

**Request Body**

```json
{
  "query": "What are the key findings about multimodal models?",
  "json_schema": {
    "type": "object",
    "properties": {
      "findings": {
        "type": "array",
        "items": {"type": "string"}
      },
      "conclusion": {"type": "string"}
    },
    "required": ["findings"]
  },
  "content": true
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Question or prompt (cleaned, non-empty) |
| `json_schema` | object | No | JSON Schema for structured output (Draft 2020-12) |
| `content` | boolean | No | Include content in citations (default: `false`) |

**Response**

```json
{
  "request_id": "ghi789",
  "answer": {
    "findings": [
      "Multimodal models show improved performance on vision-language tasks",
      "Cross-modal attention mechanisms enhance reasoning capabilities"
    ],
    "conclusion": "Multimodal architectures are advancing rapidly with significant improvements in cross-modal understanding."
  },
  "citations": [
    {
      "id": "cite1",
      "url": "https://arxiv.org/abs/2401.00123",
      "title": "Multimodal Foundation Models",
      "content": "The model achieves state-of-the-art results..."
    },
    {
      "id": "cite2",
      "url": "https://example.com/article",
      "title": "Cross-Modal Learning"
    }
  ]
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `answer` | string/object | Generated answer (string or structured JSON if `json_schema` provided) |
| `citations` | array | Array of [AnswerCitation](#answercitation) objects |

---

### Create Research Task

```http
POST /v1/research
Authorization: Bearer <token>
```

Create an asynchronous research task for multi-round investigation.

**Request Body**

```json
{
  "themes": "Current trends in multimodal AI research",
  "search_mode": "research-pro",
  "json_schema": {
    "type": "object",
    "properties": {
      "trends": {"type": "array"},
      "key_papers": {"type": "array"},
      "summary": {"type": "string"}
    }
  }
}
```

**Request Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `themes` | string | Yes | Research topic/theme (cleaned, non-empty) |
| `search_mode` | string | No | Search mode: `"research-fast"`, `"research"` (default), `"research-pro"` |
| `json_schema` | object | No | JSON Schema for structured output (Draft 2020-12) |

**Response**

```json
{
  "research_id": "a1b2c3d4e5f6",
  "create_at": 1712345678901,
  "themes": "Current trends in multimodal AI research",
  "search_mode": "research-pro",
  "json_schema": {
    "type": "object",
    "properties": {
      "trends": {"type": "array"}
    }
  },
  "status": "pending",
  "output": null,
  "finished_at": null,
  "error": null
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `research_id` | string | Unique task identifier (hex string) |
| `create_at` | integer | Creation timestamp (milliseconds since epoch) |
| `themes` | string | Research theme/topic |
| `search_mode` | string | Search mode used |
| `json_schema` | object | Provided JSON Schema (if any) |
| `status` | string | Task status: `"pending"`, `"running"`, `"completed"`, `"canceled"`, `"failed"` |
| `output` | object | Research output (only when `status="completed"`), see [ResearchResponse](#researchresponse) |
| `finished_at` | integer | Completion timestamp (only when finished) |
| `error` | string | Error message (only when `status="failed"`) |

---

### List Research Tasks

```http
GET /v1/research?cursor=0&limit=10
Authorization: Bearer <token>
```

List research tasks with pagination.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cursor` | string | No | Pagination cursor (non-negative integer string, default: `null`) |
| `limit` | integer | No | Number of results (1-50, default: `10`) |

**Response**

```json
{
  "data": [
    {
      "research_id": "a1b2c3d4e5f6",
      "create_at": 1712345678901,
      "themes": "Research topic 1",
      "search_mode": "research",
      "status": "completed",
      "output": {
        "content": "Research findings...",
        "structured": {"key": "value"}
      },
      "finished_at": 1712345789001
    },
    {
      "research_id": "b2c3d4e5f6a1",
      "create_at": 1712345500000,
      "themes": "Research topic 2",
      "search_mode": "research-fast",
      "status": "running"
    }
  ],
  "has_more": true,
  "next_cursor": "10"
}
```

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `data` | array | Array of [ResearchTaskResponse](#researchtaskresponse) objects |
| `has_more` | boolean | Whether more results are available |
| `next_cursor` | string | Cursor for next page (empty string if `has_more=false`) |

---

### Get Research Task

```http
GET /v1/research/{research_id}
Authorization: Bearer <token>
```

Get details of a specific research task.

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `research_id` | string | Yes | Task identifier (hex string) |

**Response**

Returns a [ResearchTaskResponse](#researchtaskresponse) object.

**Error Responses**

| Status Code | Description |
|-------------|-------------|
| 404 | Research task not found |

---

### Cancel Research Task

```http
DELETE /v1/research/{research_id}
Authorization: Bearer <token>
```

Cancel a running or pending research task.

**Path Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `research_id` | string | Yes | Task identifier (hex string) |

**Response**

Returns the canceled [ResearchTaskResponse](#researchtaskresponse) object with `status="canceled"`.

**Error Responses**

| Status Code | Description |
|-------------|-------------|
| 404 | Research task not found |

---

## Shared Types

### Fetch Configuration

The `fetchs` field in SearchRequest and the nested fields in FetchRequest share the same configuration structure.

#### FetchContentRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_chars` | integer | No | Maximum characters to extract (> 0) |
| `detail` | string | No | Detail level: `"concise"` (default), `"standard"`, `"full"` |
| `include_markdown_links` | boolean | No | Include markdown-formatted links (default: `false`) |
| `include_html_tags` | boolean | No | Preserve HTML tags in content (default: `false`) |
| `include_tags` | array | No | HTML tags to include: `"header"`, `"navigation"`, `"banner"`, `"body"`, `"sidebar"`, `"footer"`, `"metadata"` |
| `exclude_tags` | array | No | HTML tags to exclude (same options as `include_tags`) |

> Note: `include_tags` and `exclude_tags` must not overlap.

#### FetchAbstractsRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | No | Query for abstract relevance scoring |
| `max_chars` | integer | No | Maximum characters per abstract (> 0) |

#### FetchSubpagesRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_subpages` | integer | No | Maximum subpages to fetch (> 0) |
| `subpage_keywords` | string/array | No | Keywords to filter subpages (comma-separated string or array) |

#### FetchOverviewRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | No | Query guiding overview generation |
| `json_schema` | object | No | JSON Schema for structured overview output |

#### FetchOthersRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `max_links` | integer | No | Maximum related links to extract (> 0) |
| `max_image_links` | integer | No | Maximum image links to extract (> 0) |

> Note: At least one of `max_links` or `max_image_links` must be set.

---

### Response Types

#### FetchResultItem

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Source URL |
| `title` | string | Page title |
| `published_date` | string | Publication date (ISO 8601 format) |
| `author` | string | Author name |
| `image` | string | Main image URL |
| `favicon` | string | Favicon URL |
| `content` | string | Extracted content |
| `abstracts` | array | List of relevant abstracts |
| `abstract_scores` | array | Relevance scores for each abstract (aligned with `abstracts`) |
| `overview` | string/object | Generated overview (null if not requested) |
| `subpages` | array | Array of [FetchSubpagesResult](#fetchsubpagesresult) objects |
| `others` | object | [FetchOthersResult](#fetchothersresult) object (null if not requested) |

#### FetchSubpagesResult

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Subpage URL |
| `title` | string | Subpage title |
| `published_date` | string | Publication date (ISO 8601 format) |
| `author` | string | Author name |
| `image` | string | Image URL |
| `favicon` | string | Favicon URL |
| `content` | string | Extracted content |
| `abstracts` | array | List of abstracts |
| `abstract_scores` | array | Relevance scores |
| `overview` | string/object | Generated overview |

#### FetchOthersResult

| Field | Type | Description |
|-------|------|-------------|
| `links` | array | Related links found on the page |
| `image_links` | array | Image links found on the page |

#### FetchStatusItem

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | URL being processed |
| `status` | string | `"success"` or `"error"` |
| `error` | object | [FetchStatusError](#fetchstatuserror) (null if success) |

#### FetchStatusError

| Field | Type | Description |
|-------|------|-------------|
| `tag` | string | Error tag: `"CRAWL_NOT_FOUND"`, `"CRAWL_TIMEOUT"`, `"CRAWL_LIVECRAWL_TIMEOUT"`, `"SOURCE_NOT_AVAILABLE"`, `"UNSUPPORTED_URL"`, `"CRAWL_UNKNOWN_ERROR"` |
| `detail` | string | Error detail message (optional) |

#### AnswerCitation

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Citation identifier |
| `url` | string | Source URL |
| `title` | string | Source title |
| `content` | string | Cited content (optional, only if `content=true` in request) |

#### ResearchResponse

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Research report content |
| `structured` | object | Structured output matching provided `json_schema` (optional) |

#### ResearchTaskResponse

| Field | Type | Description |
|-------|------|-------------|
| `research_id` | string | Unique task identifier |
| `create_at` | integer | Creation timestamp (milliseconds) |
| `themes` | string | Research theme/topic |
| `search_mode` | string | Search mode: `"research-fast"`, `"research"`, `"research-pro"` |
| `json_schema` | object | JSON Schema for output (null if not provided) |
| `status` | string | `"pending"`, `"running"`, `"completed"`, `"canceled"`, `"failed"` |
| `output` | object | [ResearchResponse](#researchresponse) when completed |
| `finished_at` | integer | Completion timestamp (milliseconds, null if not finished) |
| `error` | string | Error message (null unless `status="failed"`) |

---

## Enums

### SearchMode

| Value | Description |
|-------|-------------|
| `fast` | Quick search, minimal processing |
| `auto` | Balanced search with automatic optimization (default) |
| `deep` | Deep search with multiple queries and comprehensive results |

### ResearchSearchMode

| Value | Description |
|-------|-------------|
| `research-fast` | Quick research with minimal rounds |
| `research` | Standard multi-round research (default) |
| `research-pro` | Comprehensive research with extensive analysis |

### CrawlMode

| Value | Description |
|-------|-------------|
| `never` | Never use crawling, only search provider results |
| `fallback` | Use crawl when search provider data is insufficient (default) |
| `preferred` | Prefer crawling over search provider data |
| `always` | Always crawl pages directly |

### FetchContentDetail

| Value | Description |
|-------|-------------|
| `concise` | Brief content extraction (default) |
| `standard` | Moderate content extraction |
| `full` | Full content extraction |

### FetchContentTag

HTML section tags for content filtering:

| Value | Description |
|-------|-------------|
| `header` | Page header content |
| `navigation` | Navigation elements |
| `banner` | Banner content |
| `body` | Main body content |
| `sidebar` | Sidebar content |
| `footer` | Footer content |
| `metadata` | Metadata content |

### ResearchTaskStatus

| Value | Description |
|-------|-------------|
| `pending` | Task created, waiting to start |
| `running` | Task is actively executing |
| `completed` | Task finished successfully |
| `canceled` | Task was canceled by user |
| `failed` | Task encountered an error |

---

## Error Responses

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (missing or invalid bearer token) |
| 404 | Not Found (resource does not exist) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (engine not ready) |

### Error Response Format

All error responses follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

---

## Usage Examples

### cURL Examples

#### Health Check

```bash
curl -X GET http://localhost:8000/healthz
```

**Response:**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

---

#### Search with Content

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI research papers",
    "user_location": "US",
    "mode": "auto",
    "max_results": 5,
    "fetchs": {
      "content": true,
      "abstracts": true
    }
  }'
```

**Response:**

```json
{
  "request_id": "req_abc123",
  "search_mode": "auto",
  "results": [
    {
      "url": "https://arxiv.org/abs/2401.00123",
      "title": "Multimodal Foundation Models Survey",
      "published_date": "2024-01-15",
      "author": "John Doe",
      "image": "https://arxiv.org/static/logo.png",
      "favicon": "https://arxiv.org/favicon.ico",
      "content": "This paper surveys recent advances in multimodal foundation models...",
      "abstracts": [
        "Proposes a unified architecture for vision-language tasks",
        "Achieves state-of-the-art results on multiple benchmarks"
      ],
      "abstract_scores": [0.95, 0.88],
      "overview": null,
      "subpages": [],
      "others": null
    }
  ]
}
```

---

#### Fetch URLs

```bash
curl -X POST http://localhost:8000/v1/fetch \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://arxiv.org/abs/2401.00123"],
    "content": {"detail": "standard"},
    "abstracts": {"query": "main findings"}
  }'
```

**Response:**

```json
{
  "request_id": "req_def456",
  "results": [
    {
      "url": "https://arxiv.org/abs/2401.00123",
      "title": "Multimodal Foundation Models Survey",
      "published_date": "2024-01-15",
      "author": "John Doe",
      "image": "",
      "favicon": "https://arxiv.org/favicon.ico",
      "content": "Full paper content extracted here...",
      "abstracts": [
        "The paper proposes a unified transformer architecture",
        "Experiments show 15% improvement over baselines"
      ],
      "abstract_scores": [0.92, 0.85],
      "overview": null,
      "subpages": [],
      "others": null
    }
  ],
  "statuses": [
    {
      "url": "https://arxiv.org/abs/2401.00123",
      "status": "success",
      "error": null
    }
  ]
}
```

---

#### Generate Answer

```bash
curl -X POST http://localhost:8000/v1/answer \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key contributions",
    "content": true
  }'
```

**Response:**

```json
{
  "request_id": "req_ghi789",
  "answer": "The key contributions of this research include: (1) a novel unified architecture for multimodal learning, (2) significant performance improvements on vision-language benchmarks, and (3) efficient training methods that reduce computational costs.",
  "citations": [
    {
      "id": "cite_001",
      "url": "https://arxiv.org/abs/2401.00123",
      "title": "Multimodal Foundation Models Survey",
      "content": "The proposed architecture achieves 15% improvement..."
    },
    {
      "id": "cite_002",
      "url": "https://arxiv.org/abs/2401.00045",
      "title": "Efficient Multimodal Training"
    }
  ]
}
```

---

#### Create Research Task

```bash
curl -X POST http://localhost:8000/v1/research \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "themes": "State of multimodal AI in 2024",
    "search_mode": "research"
  }'
```

**Response:**

```json
{
  "research_id": "a1b2c3d4e5f6g7h8",
  "create_at": 1712345678901,
  "themes": "State of multimodal AI in 2024",
  "search_mode": "research",
  "json_schema": null,
  "status": "pending",
  "output": null,
  "finished_at": null,
  "error": null
}
```

---

#### Check Research Status

```bash
curl -X GET http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response (running):**

```json
{
  "research_id": "a1b2c3d4e5f6",
  "create_at": 1712345678901,
  "themes": "State of multimodal AI in 2024",
  "search_mode": "research",
  "json_schema": null,
  "status": "running",
  "output": null,
  "finished_at": null,
  "error": null
}
```

**Response (completed):**

```json
{
  "research_id": "a1b2c3d4e5f6",
  "create_at": 1712345678901,
  "themes": "State of multimodal AI in 2024",
  "search_mode": "research",
  "json_schema": null,
  "status": "completed",
  "output": {
    "content": "## Research Summary\n\nMultimodal AI has seen significant advances in 2024...",
    "structured": null
  },
  "finished_at": 1712345999001,
  "error": null
}
```

---

#### Cancel Research

```bash
curl -X DELETE http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**

```json
{
  "research_id": "a1b2c3d4e5f6",
  "create_at": 1712345678901,
  "themes": "State of multimodal AI in 2024",
  "search_mode": "research",
  "json_schema": null,
  "status": "canceled",
  "output": null,
  "finished_at": 1712345712345,
  "error": null
}
```

---

#### List Research Tasks

```bash
curl -X GET "http://localhost:8000/v1/research?limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**

```json
{
  "data": [
    {
      "research_id": "a1b2c3d4e5f6",
      "create_at": 1712345678901,
      "themes": "State of multimodal AI in 2024",
      "search_mode": "research",
      "json_schema": null,
      "status": "completed",
      "output": {
        "content": "Research summary...",
        "structured": null
      },
      "finished_at": 1712345999001,
      "error": null
    },
    {
      "research_id": "b2c3d4e5f6a7",
      "create_at": 1712345000000,
      "themes": "LLM inference optimization",
      "search_mode": "research-fast",
      "json_schema": null,
      "status": "running",
      "output": null,
      "finished_at": null,
      "error": null
    }
  ],
  "has_more": false,
  "next_cursor": ""
}
```

---

## OpenAPI Documentation

When `api.enable_docs` is set to `true` in configuration, interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## Rate Limiting

This is a personal API intended for single-user or trusted private environments. No built-in rate limiting is enforced. Configure external rate limiting if exposing to broader networks.

---

## Versioning

All API endpoints use the `/v1/` prefix. Future versions will use `/v2/`, etc., maintaining backward compatibility where possible.