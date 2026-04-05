# RaySearch API 文件

本文說明 RaySearch 個人 API 服務的 HTTP 端點、請求/回應模型與驗證要求。

## 概覽

RaySearch 是一個面向 AI Overview 類工作流程的非同步優先搜尋編排引擎，提供以下能力：

- **Search**：跨多個供應商進行檢索，並可選擇執行抓取與重排
- **Fetch**：頁面抓取、內容擷取、摘要、概覽生成與相關連結擷取
- **Answer**：帶引用的受約束回答生成
- **Research**：多輪研究報告與結構化輸出

## 驗證

除 `/healthz` 外，所有端點都需要 Bearer Token 驗證。

```http
Authorization: Bearer <YOUR_API_KEY>
```

Token 可透過設定檔中的 `api.bearer_token` 或環境變數 `RAYSEARCH_API_KEY` 設定。

## 基礎 URL

預設值：`http://localhost:8000`

可透過設定中的 `api.host` 與 `api.port` 配置。

---

## 端點

### 健康檢查

```http
GET /healthz
```

回傳服務健康狀態，無需驗證。

**回應**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

**欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `status` | string | 服務狀態，`"ok"` 或 `"error"` |
| `engine_ready` | boolean | 引擎是否已初始化 |

---

### Search

```http
POST /v1/search
Authorization: Bearer <token>
```

在已配置的供應商之間執行搜尋，並可選擇抓取正文內容。

**請求本文**

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

**欄位**

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `query` | string | 是 | 搜尋詞，清理後不得為空 |
| `user_location` | string | 是 | 兩位元 ISO 國家代碼，例如 `"US"`、`"JP"`、`"CN"` |
| `mode` | string | 否 | 搜尋模式：`"fast"`、`"auto"`（預設）、`"deep"` |
| `max_results` | integer | 否 | 最大回傳筆數，大於 0 |
| `start_published_date` | string | 否 | 用於過濾的 ISO 8601 日期字串 |
| `end_published_date` | string | 否 | 用於過濾的 ISO 8601 日期字串 |
| `include_domains` | array | 否 | 只包含這些網域，會自動標準化並去重 |
| `exclude_domains` | array | 否 | 排除這些網域，會自動標準化並去重 |
| `include_text` | array | 否 | 需要包含的文字片語 |
| `exclude_text` | array | 否 | 需要排除的文字片語 |
| `moderation` | boolean | 否 | 是否啟用內容審核，預設 `true` |
| `additional_queries` | array | 否 | 額外查詢，僅在 `mode="deep"` 時支援 |
| `fetchs` | object | 是 | 抓取設定，見 [擷取設定](#fetch-configuration) |

**回應**

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

**回應欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `request_id` | string | 請求識別碼 |
| `search_mode` | string | 實際使用的搜尋模式 |
| `results` | array | [FetchResultItem](#fetchresultitem) 陣列 |

---

### Fetch

```http
POST /v1/fetch
Authorization: Bearer <token>
```

抓取並處理指定 URL 的內容。

**請求本文**

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

**欄位**

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `urls` | array | 是 | URL 清單，必須是 `http://` 或 `https://`，且不得為空 |
| `crawl_mode` | string | 否 | 抓取模式：`"never"`、`"fallback"`（預設）、`"preferred"`、`"always"` |
| `crawl_timeout` | float | 否 | 抓取逾時秒數，大於 0 |
| `content` | boolean/object | 否 | 內容擷取設定，至少需要一個動作 |
| `abstracts` | boolean/object | 否 | 摘要生成設定 |
| `subpages` | object | 否 | 子頁面擷取設定 |
| `overview` | boolean/object | 否 | 概覽生成設定 |
| `others` | object | 否 | 相關連結擷取設定 |

**回應**

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

**回應欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `request_id` | string | 請求識別碼 |
| `results` | array | 成功的 [FetchResultItem](#fetchresultitem) 陣列 |
| `statuses` | array | 每個 URL 對應的 [FetchStatusItem](#fetchstatusitem) 陣列 |

---

### Answer

```http
POST /v1/answer
Authorization: Bearer <token>
```

根據提供的內容生成帶引用的回答。

**請求本文**

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

**欄位**

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `query` | string | 是 | 問題或提示詞，清理後不得為空 |
| `json_schema` | object | 否 | 用於結構化輸出的 JSON Schema（Draft 2020-12） |
| `content` | boolean | 否 | 是否在引用中包含內容，預設 `false` |

**回應**

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

**回應欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `request_id` | string | 請求識別碼 |
| `answer` | string/object | 生成的回答；若提供 `json_schema`，則為結構化 JSON |
| `citations` | array | [AnswerCitation](#answercitation) 陣列 |

---

### Create Research Task

```http
POST /v1/research
Authorization: Bearer <token>
```

建立一個用於多輪研究的非同步任務。

**請求本文**

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

**欄位**

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `themes` | string | 是 | 研究主題，清理後不得為空 |
| `search_mode` | string | 否 | 搜尋模式：`"research-fast"`、`"research"`（預設）、`"research-pro"` |
| `json_schema` | object | 否 | 用於結構化輸出的 JSON Schema（Draft 2020-12） |

**回應**

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

**回應欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `research_id` | string | 任務識別碼（十六進位字串） |
| `create_at` | integer | 建立時間戳記（毫秒） |
| `themes` | string | 研究主題 |
| `search_mode` | string | 使用的搜尋模式 |
| `json_schema` | object | 提供的 JSON Schema（若有） |
| `status` | string | `pending`、`running`、`completed`、`canceled`、`failed` |
| `output` | object | 完成時的 [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完成時間戳記，僅在結束時填入 |
| `error` | string | 錯誤訊息，僅在 `status="failed"` 時存在 |

---

### List Research Tasks

```http
GET /v1/research?cursor=0&limit=10
Authorization: Bearer <token>
```

以分頁方式列出研究任務。

**查詢參數**

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `cursor` | string | 否 | 分頁游標，非負整數字串，預設 `null` |
| `limit` | integer | 否 | 回傳筆數，範圍 1-50，預設 `10` |

**回應**

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

**回應欄位**

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `data` | array | [ResearchTaskResponse](#researchtaskresponse) 陣列 |
| `has_more` | boolean | 是否還有更多結果 |
| `next_cursor` | string | 下一頁游標。若 `has_more=false` 則為空字串 |

---

### Get Research Task

```http
GET /v1/research/{research_id}
Authorization: Bearer <token>
```

取得特定研究任務的詳細資訊。

**路徑參數**

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `research_id` | string | 是 | 任務識別碼（十六進位字串） |

**回應**

回傳一個 [ResearchTaskResponse](#researchtaskresponse) 物件。

**錯誤回應**

| 狀態碼 | 說明 |
|--------|------|
| 404 | 找不到研究任務 |

---

### Cancel Research Task

```http
DELETE /v1/research/{research_id}
Authorization: Bearer <token>
```

取消正在執行或等待中的研究任務。

**路徑參數**

| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| `research_id` | string | 是 | 任務識別碼（十六進位字串） |

**回應**

回傳已取消的 [ResearchTaskResponse](#researchtaskresponse) 物件，且 `status="canceled"`。

**錯誤回應**

| 狀態碼 | 說明 |
|--------|------|
| 404 | 找不到研究任務 |

---

## 共用型別

<a id="fetch-configuration"></a>
### 擷取設定

`SearchRequest` 的 `fetchs` 欄位，以及 `FetchRequest` 的巢狀欄位，共用相同的設定結構。

#### FetchContentRequest

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `max_chars` | integer | 否 | 最大擷取字元數，大於 0 |
| `detail` | string | 否 | 詳細程度：`"concise"`（預設）、`"standard"`、`"full"` |
| `include_markdown_links` | boolean | 否 | 是否包含 Markdown 連結，預設 `false` |
| `include_html_tags` | boolean | 否 | 是否保留 HTML 標籤，預設 `false` |
| `include_tags` | array | 否 | 要包含的 HTML 標籤 |
| `exclude_tags` | array | 否 | 要排除的 HTML 標籤 |

> 注意：`include_tags` 與 `exclude_tags` 不可重疊。

#### FetchAbstractsRequest

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `query` | string | 否 | 用於摘要相關性評分的查詢詞 |
| `max_chars` | integer | 否 | 每則摘要的最大字元數，大於 0 |

#### FetchSubpagesRequest

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `max_subpages` | integer | 否 | 最大子頁面數量，大於 0 |
| `subpage_keywords` | string/array | 否 | 用於篩選子頁面的關鍵字，支援逗號分隔字串或陣列 |

#### FetchOverviewRequest

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `query` | string | 否 | 引導概覽生成的查詢詞 |
| `json_schema` | object | 否 | 用於結構化概覽輸出的 JSON Schema |

#### FetchOthersRequest

| 欄位 | 類型 | 必填 | 說明 |
|-------|------|------|------|
| `max_links` | integer | 否 | 最大相關連結數，大於 0 |
| `max_image_links` | integer | 否 | 最大圖片連結數，大於 0 |

> 注意：`max_links` 與 `max_image_links` 至少要設定一項。

---

### 回應型別

#### FetchResultItem

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `url` | string | 來源 URL |
| `title` | string | 頁面標題 |
| `published_date` | string | 發布日期，ISO 8601 格式 |
| `author` | string | 作者名稱 |
| `image` | string | 主圖 URL |
| `favicon` | string | 網站圖示 URL |
| `content` | string | 擷取的內容 |
| `abstracts` | array | 相關摘要列表 |
| `abstract_scores` | array | 每則摘要的相關性分數，與 `abstracts` 對應 |
| `overview` | string/object | 生成的概覽；若未請求則為 `null` |
| `subpages` | array | [FetchSubpagesResult](#fetchsubpagesresult) 陣列 |
| `others` | object | [FetchOthersResult](#fetchothersresult) 物件；若未請求則為 `null` |

#### FetchSubpagesResult

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `url` | string | 子頁面 URL |
| `title` | string | 子頁面標題 |
| `published_date` | string | 發布日期，ISO 8601 格式 |
| `author` | string | 作者名稱 |
| `image` | string | 圖片 URL |
| `favicon` | string | 網站圖示 URL |
| `content` | string | 擷取的內容 |
| `abstracts` | array | 摘要列表 |
| `abstract_scores` | array | 相關性分數 |
| `overview` | string/object | 生成的概覽 |

#### FetchOthersResult

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `links` | array | 在頁面中找到的相關連結 |
| `image_links` | array | 在頁面中找到的圖片連結 |

#### FetchStatusItem

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `url` | string | 正在處理的 URL |
| `status` | string | `"success"` 或 `"error"` |
| `error` | object | [FetchStatusError](#fetchstatuserror)，成功時為 `null` |

#### FetchStatusError

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `tag` | string | 錯誤標籤：`"CRAWL_NOT_FOUND"`、`"CRAWL_TIMEOUT"`、`"CRAWL_LIVECRAWL_TIMEOUT"`、`"SOURCE_NOT_AVAILABLE"`、`"UNSUPPORTED_URL"`、`"CRAWL_UNKNOWN_ERROR"` |
| `detail` | string | 錯誤詳細訊息，可選 |

#### AnswerCitation

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `id` | string | 引用識別碼 |
| `url` | string | 來源 URL |
| `title` | string | 來源標題 |
| `content` | string | 被引用內容，僅在請求 `content=true` 時可選回傳 |

#### ResearchResponse

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `content` | string | 研究報告正文 |
| `structured` | object | 與提供的 `json_schema` 相符的結構化輸出，可選 |

#### ResearchTaskResponse

| 欄位 | 類型 | 說明 |
|-------|------|------|
| `research_id` | string | 任務識別碼 |
| `create_at` | integer | 建立時間戳記（毫秒） |
| `themes` | string | 研究主題 |
| `search_mode` | string | 搜尋模式：`"research-fast"`、`"research"`、`"research-pro"` |
| `json_schema` | object | 輸出用 JSON Schema；未提供時為 `null` |
| `status` | string | `pending`、`running`、`completed`、`canceled`、`failed` |
| `output` | object | 完成時的 [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完成時間戳記，未完成時為 `null` |
| `error` | string | 錯誤訊息，僅在 `status="failed"` 時非空 |

---

## 列舉

### SearchMode

| 值 | 說明 |
|----|------|
| `fast` | 快速搜尋，處理最少 |
| `auto` | 平衡型搜尋，自動最佳化，預設值 |
| `deep` | 深度搜尋，使用多個查詢並回傳更完整的結果 |

### ResearchSearchMode

| 值 | 說明 |
|----|------|
| `research-fast` | 以最少輪次進行快速研究 |
| `research` | 標準多輪研究，預設值 |
| `research-pro` | 具更完整分析的進階研究 |

### CrawlMode

| 值 | 說明 |
|----|------|
| `never` | 永不抓取，只使用搜尋供應商結果 |
| `fallback` | 當搜尋結果不足時才抓取，預設值 |
| `preferred` | 優先抓取頁面 |
| `always` | 永遠直接抓取頁面 |

### FetchContentDetail

| 值 | 說明 |
|----|------|
| `concise` | 簡潔擷取，預設值 |
| `standard` | 標準擷取 |
| `full` | 完整擷取 |

### FetchContentTag

用於內容過濾的 HTML 區塊標籤：

| 值 | 說明 |
|----|------|
| `header` | 頁首內容 |
| `navigation` | 導覽元素 |
| `banner` | 橫幅內容 |
| `body` | 主要內容 |
| `sidebar` | 側邊欄 |
| `footer` | 頁尾 |
| `metadata` | 中繼資料 |

### ResearchTaskStatus

| 值 | 說明 |
|----|------|
| `pending` | 任務已建立，等待開始 |
| `running` | 任務正在執行 |
| `completed` | 任務已成功完成 |
| `canceled` | 任務已由使用者取消 |
| `failed` | 任務發生錯誤 |

---

## 錯誤回應

### HTTP 狀態碼

| 狀態碼 | 說明 |
|--------|------|
| 200 | 成功 |
| 400 | 請求錯誤，參數無效 |
| 401 | 未授權，缺少或無效的 Bearer Token |
| 404 | 找不到資源 |
| 500 | 伺服器內部錯誤 |
| 503 | 服務不可用，引擎尚未就緒 |

### 錯誤回應格式

所有錯誤回應都使用以下格式：

```json
{
  "detail": "Error message describing the issue"
}
```

---

## 使用範例

### cURL 範例

#### 健康檢查

```bash
curl -X GET http://localhost:8000/healthz
```

**回應：**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

---

#### 帶內容的搜尋

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

**回應：**

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

#### 擷取 URL

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

**回應：**

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

**回應：**

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

#### 建立研究任務

```bash
curl -X POST http://localhost:8000/v1/research \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "themes": "State of multimodal AI in 2024",
    "search_mode": "research"
  }'
```

**回應：**

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

#### 檢查研究狀態

```bash
curl -X GET http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**回應（running）：**

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

**回應（completed）：**

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

**回應：**

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

#### 列出研究任務

```bash
curl -X GET "http://localhost:8000/v1/research?limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**回應：**

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

## OpenAPI 文件

當設定中的 `api.enable_docs` 設為 `true` 時，可使用互動式 API 文件：

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## 速率限制

這是面向單一使用者或可信任私有環境的個人 API，不內建速率限制。若對外暴露，請配置外部限流。

---

## 版本管理

所有 API 端點都使用 `/v1/` 前綴。未來版本將使用 `/v2/` 等前綴，並盡可能保持向後相容。
