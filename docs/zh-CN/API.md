# RaySearch API 文档

本文档说明 RaySearch 个人 API 服务的 HTTP 接口、请求/响应模型以及认证要求。

## 概览

RaySearch 是一个面向 AI Overview 类工作流的异步优先搜索编排引擎，提供以下能力：

- **Search**：跨多个提供商检索，并可选执行抓取与重排
- **Fetch**：页面抓取、内容抽取、摘要、概览生成与相关链接提取
- **Answer**：带引用的受约束回答生成
- **Research**：多轮研究报告与结构化输出

## 认证

除 `/healthz` 外，所有接口都需要 Bearer Token 认证。

```http
Authorization: Bearer <YOUR_API_KEY>
```

Token 通过配置文件中的 `api.bearer_token` 或环境变量 `RAYSEARCH_API_KEY` 设置。

## 基础 URL

默认值：`http://localhost:8000`

可通过设置中的 `api.host` 和 `api.port` 配置。

---

## 接口

### 健康检查

```http
GET /healthz
```

返回服务健康状态，无需认证。

**响应**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

**字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `status` | string | 服务状态，`"ok"` 或 `"error"` |
| `engine_ready` | boolean | 引擎是否已初始化 |

---

### Search

```http
POST /v1/search
Authorization: Bearer <token>
```

在已配置的提供商之间执行搜索，并可选抓取正文内容。

**请求体**

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

**字段**

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `query` | string | 是 | 搜索词，必须清理后非空 |
| `user_location` | string | 是 | 两位 ISO 国家代码，例如 `"US"`、`"JP"`、`"CN"` |
| `mode` | string | 否 | 搜索模式：`"fast"`、`"auto"`（默认）、`"deep"` |
| `max_results` | integer | 否 | 最大返回结果数，大于 0 |
| `start_published_date` | string | 否 | 用于过滤的 ISO 8601 日期 |
| `end_published_date` | string | 否 | 用于过滤的 ISO 8601 日期 |
| `include_domains` | array | 否 | 仅包含这些域名，自动标准化并去重 |
| `exclude_domains` | array | 否 | 排除这些域名，自动标准化并去重 |
| `include_text` | array | 否 | 需要包含的文本短语 |
| `exclude_text` | array | 否 | 需要排除的文本短语 |
| `moderation` | boolean | 否 | 是否启用内容审核，默认 `true` |
| `additional_queries` | array | 否 | 额外查询，仅在 `mode="deep"` 时支持 |
| `fetchs` | object | 是 | 抓取配置，见 [获取配置](#fetch-configuration) |

**响应**

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

**响应字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `request_id` | string | 请求唯一标识 |
| `search_mode` | string | 实际使用的搜索模式 |
| `results` | array | [FetchResultItem](#fetchresultitem) 对象数组 |

---

### Fetch

```http
POST /v1/fetch
Authorization: Bearer <token>
```

抓取并处理指定 URL 的内容。

**请求体**

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

**字段**

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `urls` | array | 是 | URL 列表，必须是 `http://` 或 `https://`，且非空 |
| `crawl_mode` | string | 否 | 抓取模式：`"never"`、`"fallback"`（默认）、`"preferred"`、`"always"` |
| `crawl_timeout` | float | 否 | 抓取超时时间，单位秒，大于 0 |
| `content` | boolean/object | 否 | 内容抽取配置，至少要有一个动作 |
| `abstracts` | boolean/object | 否 | 摘要生成配置 |
| `subpages` | object | 否 | 子页面抽取配置 |
| `overview` | boolean/object | 否 | 概览生成配置 |
| `others` | object | 否 | 相关链接抽取配置 |

**响应**

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

**响应字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `request_id` | string | 请求唯一标识 |
| `results` | array | 成功的 [FetchResultItem](#fetchresultitem) 数组 |
| `statuses` | array | 每个 URL 对应的 [FetchStatusItem](#fetchstatusitem) 数组 |

---

### Answer

```http
POST /v1/answer
Authorization: Bearer <token>
```

基于提供的内容生成带引用的回答。

**请求体**

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

**字段**

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `query` | string | 是 | 问题或提示词，必须清理后非空 |
| `json_schema` | object | 否 | 用于结构化输出的 JSON Schema（Draft 2020-12） |
| `content` | boolean | 否 | 是否在引用中包含内容，默认 `false` |

**响应**

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

**响应字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `request_id` | string | 请求唯一标识 |
| `answer` | string/object | 生成结果；如果提供了 `json_schema`，则返回结构化 JSON |
| `citations` | array | [AnswerCitation](#answercitation) 对象数组 |

---

### Create Research Task

```http
POST /v1/research
Authorization: Bearer <token>
```

创建一个用于多轮研究的异步任务。

**请求体**

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

**字段**

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `themes` | string | 是 | 研究主题，必须清理后非空 |
| `search_mode` | string | 否 | 搜索模式：`"research-fast"`、`"research"`（默认）、`"research-pro"` |
| `json_schema` | object | 否 | 用于结构化输出的 JSON Schema（Draft 2020-12） |

**响应**

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

**响应字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `research_id` | string | 任务唯一标识（十六进制字符串） |
| `create_at` | integer | 创建时间戳（毫秒） |
| `themes` | string | 研究主题 |
| `search_mode` | string | 使用的搜索模式 |
| `json_schema` | object | 提供的 JSON Schema（如果有） |
| `status` | string | 任务状态：`"pending"`、`"running"`、`"completed"`、`"canceled"`、`"failed"` |
| `output` | object | 研究输出，仅在 `status="completed"` 时返回，见 [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完成时间戳，仅在任务结束时返回 |
| `error` | string | 错误信息，仅在 `status="failed"` 时返回 |

---

### List Research Tasks

```http
GET /v1/research?cursor=0&limit=10
Authorization: Bearer <token>
```

分页列出研究任务。

**查询参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `cursor` | string | 否 | 分页游标，非负整数字符串，默认 `null` |
| `limit` | integer | 否 | 返回数量，范围 1-50，默认 `10` |

**响应**

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

**响应字段**

| 字段 | 类型 | 说明 |
|-------|------|------|
| `data` | array | [ResearchTaskResponse](#researchtaskresponse) 对象数组 |
| `has_more` | boolean | 是否还有更多结果 |
| `next_cursor` | string | 下一页游标，若 `has_more=false` 则为空字符串 |

---

### Get Research Task

```http
GET /v1/research/{research_id}
Authorization: Bearer <token>
```

获取某个研究任务的详细信息。

**路径参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `research_id` | string | 是 | 任务标识（十六进制字符串） |

**响应**

返回一个 [ResearchTaskResponse](#researchtaskresponse) 对象。

**错误响应**

| 状态码 | 说明 |
|--------|------|
| 404 | 未找到研究任务 |

---

### Cancel Research Task

```http
DELETE /v1/research/{research_id}
Authorization: Bearer <token>
```

取消正在运行或等待中的研究任务。

**路径参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `research_id` | string | 是 | 任务标识（十六进制字符串） |

**响应**

返回已取消的 [ResearchTaskResponse](#researchtaskresponse) 对象，且 `status="canceled"`。

**错误响应**

| 状态码 | 说明 |
|--------|------|
| 404 | 未找到研究任务 |

---

## 共享类型

<a id="fetch-configuration"></a>
### 获取配置

`SearchRequest` 中的 `fetchs` 字段，以及 `FetchRequest` 中的嵌套字段，共享同一套配置结构。

#### FetchContentRequest

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `max_chars` | integer | 否 | 最大抽取字符数，大于 0 |
| `detail` | string | 否 | 细节级别：`"concise"`（默认）、`"standard"`、`"full"` |
| `include_markdown_links` | boolean | 否 | 是否保留 Markdown 链接，默认 `false` |
| `include_html_tags` | boolean | 否 | 是否保留 HTML 标签，默认 `false` |
| `include_tags` | array | 否 | 要包含的 HTML 标签 |
| `exclude_tags` | array | 否 | 要排除的 HTML 标签 |

> 注意：`include_tags` 和 `exclude_tags` 不能重叠。

#### FetchAbstractsRequest

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `query` | string | 否 | 用于摘要相关性评分的查询词 |
| `max_chars` | integer | 否 | 每条摘要的最大字符数，大于 0 |

#### FetchSubpagesRequest

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `max_subpages` | integer | 否 | 最大子页面数量，大于 0 |
| `subpage_keywords` | string/array | 否 | 用于筛选子页面的关键词，支持逗号分隔字符串或数组 |

#### FetchOverviewRequest

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `query` | string | 否 | 用于引导概览生成的查询词 |
| `json_schema` | object | 否 | 用于结构化概览输出的 JSON Schema |

#### FetchOthersRequest

| 字段 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `max_links` | integer | 否 | 最大相关链接数，大于 0 |
| `max_image_links` | integer | 否 | 最大图片链接数，大于 0 |

> 注意：`max_links` 和 `max_image_links` 至少要设置一个。

---

### 响应类型

#### FetchResultItem

| 字段 | 类型 | 说明 |
|-------|------|------|
| `url` | string | 源 URL |
| `title` | string | 页面标题 |
| `published_date` | string | 发表日期，ISO 8601 格式 |
| `author` | string | 作者名 |
| `image` | string | 主图 URL |
| `favicon` | string | 网站图标 URL |
| `content` | string | 抽取的内容 |
| `abstracts` | array | 相关摘要列表 |
| `abstract_scores` | array | 每条摘要的相关性评分，与 `abstracts` 对齐 |
| `overview` | string/object | 生成的概览；如果未请求则为 `null` |
| `subpages` | array | [FetchSubpagesResult](#fetchsubpagesresult) 对象数组 |
| `others` | object | [FetchOthersResult](#fetchothersresult) 对象；如果未请求则为 `null` |

#### FetchSubpagesResult

| 字段 | 类型 | 说明 |
|-------|------|------|
| `url` | string | 子页面 URL |
| `title` | string | 子页面标题 |
| `published_date` | string | 发表日期，ISO 8601 格式 |
| `author` | string | 作者名 |
| `image` | string | 图片 URL |
| `favicon` | string | 网站图标 URL |
| `content` | string | 抽取的内容 |
| `abstracts` | array | 摘要列表 |
| `abstract_scores` | array | 相关性评分 |
| `overview` | string/object | 生成的概览 |

#### FetchOthersResult

| 字段 | 类型 | 说明 |
|-------|------|------|
| `links` | array | 页面中找到的相关链接 |
| `image_links` | array | 页面中找到的图片链接 |

#### FetchStatusItem

| 字段 | 类型 | 说明 |
|-------|------|------|
| `url` | string | 正在处理的 URL |
| `status` | string | `"success"` 或 `"error"` |
| `error` | object | [FetchStatusError](#fetchstatuserror)，成功时为 `null` |

#### FetchStatusError

| 字段 | 类型 | 说明 |
|-------|------|------|
| `tag` | string | 错误标签：`"CRAWL_NOT_FOUND"`、`"CRAWL_TIMEOUT"`、`"CRAWL_LIVECRAWL_TIMEOUT"`、`"SOURCE_NOT_AVAILABLE"`、`"UNSUPPORTED_URL"`、`"CRAWL_UNKNOWN_ERROR"` |
| `detail` | string | 错误详情，可选 |

#### AnswerCitation

| 字段 | 类型 | 说明 |
|-------|------|------|
| `id` | string | 引用标识 |
| `url` | string | 来源 URL |
| `title` | string | 来源标题 |
| `content` | string | 被引用内容，仅当请求中 `content=true` 时可选返回 |

#### ResearchResponse

| 字段 | 类型 | 说明 |
|-------|------|------|
| `content` | string | 研究报告正文 |
| `structured` | object | 与提供的 `json_schema` 匹配的结构化输出，可选 |

#### ResearchTaskResponse

| 字段 | 类型 | 说明 |
|-------|------|------|
| `research_id` | string | 任务唯一标识 |
| `create_at` | integer | 创建时间戳（毫秒） |
| `themes` | string | 研究主题 |
| `search_mode` | string | 搜索模式：`"research-fast"`、`"research"`、`"research-pro"` |
| `json_schema` | object | 输出用 JSON Schema，未提供时为 `null` |
| `status` | string | `pending`、`running`、`completed`、`canceled`、`failed` |
| `output` | object | 完成时的 [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完成时间戳，未完成时为 `null` |
| `error` | string | 错误信息，仅在 `status="failed"` 时非空 |

---

## 枚举

### SearchMode

| 值 | 说明 |
|----|------|
| `fast` | 快速搜索，处理最少 |
| `auto` | 平衡型搜索，自动优化，默认值 |
| `deep` | 深度搜索，使用多个查询并返回更全面的结果 |

### ResearchSearchMode

| 值 | 说明 |
|----|------|
| `research-fast` | 快速研究，轮次最少 |
| `research` | 标准多轮研究，默认值 |
| `research-pro` | 深度研究，分析更全面 |

### CrawlMode

| 值 | 说明 |
|----|------|
| `never` | 从不抓取，只使用搜索结果 |
| `fallback` | 当搜索结果不足时再抓取，默认值 |
| `preferred` | 优先抓取页面 |
| `always` | 始终直接抓取页面 |

### FetchContentDetail

| 值 | 说明 |
|----|------|
| `concise` | 简洁抽取，默认值 |
| `standard` | 标准抽取 |
| `full` | 完整抽取 |

### FetchContentTag

用于内容过滤的 HTML 区块标签：

| 值 | 说明 |
|----|------|
| `header` | 页面头部 |
| `navigation` | 导航元素 |
| `banner` | 横幅内容 |
| `body` | 主体内容 |
| `sidebar` | 侧边栏 |
| `footer` | 页脚 |
| `metadata` | 元数据 |

### ResearchTaskStatus

| 值 | 说明 |
|----|------|
| `pending` | 任务已创建，等待开始 |
| `running` | 任务正在执行 |
| `completed` | 任务已成功完成 |
| `canceled` | 任务已被用户取消 |
| `failed` | 任务执行失败 |

---

## 错误响应

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求错误，参数无效 |
| 401 | 未授权，缺少或无效的 Bearer Token |
| 404 | 未找到资源 |
| 500 | 服务器内部错误 |
| 503 | 服务不可用，引擎尚未就绪 |

### 错误响应格式

所有错误响应都使用以下格式：

```json
{
  "detail": "Error message describing the issue"
}
```

---

## 使用示例

### cURL 示例

#### 健康检查

```bash
curl -X GET http://localhost:8000/healthz
```

**响应：**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

---

#### 带内容的搜索

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

**响应：**

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

#### 抓取 URL

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

**响应：**

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

#### 生成回答

```bash
curl -X POST http://localhost:8000/v1/answer \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key contributions",
    "content": true
  }'
```

**响应：**

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

#### 创建研究任务

```bash
curl -X POST http://localhost:8000/v1/research \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "themes": "State of multimodal AI in 2024",
    "search_mode": "research"
  }'
```

**响应：**

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

#### 检查研究状态

```bash
curl -X GET http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**响应（running）：**

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

**响应（completed）：**

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

#### 取消研究

```bash
curl -X DELETE http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**响应：**

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

#### 列出研究任务

```bash
curl -X GET "http://localhost:8000/v1/research?limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**响应：**

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

## OpenAPI 文档

当配置中的 `api.enable_docs` 设置为 `true` 时，可访问交互式 API 文档：

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## 限流

这是一个面向单用户或受信任私有环境的个人 API，不内置速率限制。若对外暴露，请配置外部限流。

---

## 版本控制

所有接口都使用 `/v1/` 前缀。未来版本将使用 `/v2/` 等前缀，并尽可能保持向后兼容。
