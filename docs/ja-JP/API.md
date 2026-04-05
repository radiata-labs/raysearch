# RaySearch API ドキュメント

このドキュメントでは、RaySearch の個人向け API サービスにおける HTTP エンドポイント、リクエスト/レスポンスモデル、認証要件を説明します。

## 概要

RaySearch は、AI Overview 系ワークフロー向けの非同期ファーストな検索オーケストレーションエンジンです。主な機能は次のとおりです。

- **Search**: 複数プロバイダを横断した検索。必要に応じて取得と再ランキングを実行
- **Fetch**: ページのクロール、抽出、要約、概要生成、関連リンク取得
- **Answer**: 引用付きの根拠ある回答生成
- **Research**: 複数ラウンドの調査レポートと構造化出力

## 認証

`/healthz` を除くすべてのエンドポイントは Bearer Token 認証が必要です。

```http
Authorization: Bearer <YOUR_API_KEY>
```

トークンは設定ファイルの `api.bearer_token` または環境変数 `RAYSEARCH_API_KEY` で指定します。

## ベース URL

デフォルト: `http://localhost:8000`

`api.host` と `api.port` で設定できます。

---

## エンドポイント

### ヘルスチェック

```http
GET /healthz
```

サービスの稼働状態を返します。認証は不要です。

**レスポンス**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

**フィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `status` | string | サービス状態。`"ok"` または `"error"` |
| `engine_ready` | boolean | エンジンが初期化済みかどうか |

---

### Search

```http
POST /v1/search
Authorization: Bearer <token>
```

設定済みプロバイダを横断して検索し、必要に応じて本文の取得を行います。

**リクエスト本文**

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

**フィールド**

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `query` | string | はい | 検索クエリ。整形後に空であってはならない |
| `user_location` | string | はい | 2文字の ISO 国コード。例: `"US"`、`"JP"`、`"CN"` |
| `mode` | string | いいえ | 検索モード。`"fast"`、`"auto"`（既定）、`"deep"` |
| `max_results` | integer | いいえ | 最大件数。0 より大きいこと |
| `start_published_date` | string | いいえ | フィルタ用 ISO 8601 日付文字列 |
| `end_published_date` | string | いいえ | フィルタ用 ISO 8601 日付文字列 |
| `include_domains` | array | いいえ | 含めるドメイン。正規化され重複排除されます |
| `exclude_domains` | array | いいえ | 除外するドメイン。正規化され重複排除されます |
| `include_text` | array | いいえ | 含めるテキスト句 |
| `exclude_text` | array | いいえ | 除外するテキスト句 |
| `moderation` | boolean | いいえ | コンテンツモデレーションを有効化。既定は `true` |
| `additional_queries` | array | いいえ | 追加クエリ。`mode="deep"` のときのみ対応 |
| `fetchs` | object | はい | 取得設定。[フェッチ設定](#fetch-configuration) を参照 |

**レスポンス**

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

**レスポンスフィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `request_id` | string | リクエスト識別子 |
| `search_mode` | string | 使用された検索モード |
| `results` | array | [FetchResultItem](#fetchresultitem) の配列 |

---

### Fetch

```http
POST /v1/fetch
Authorization: Bearer <token>
```

指定 URL の内容を取得して処理します。

**リクエスト本文**

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

**フィールド**

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `urls` | array | はい | URL の一覧。`http://` または `https://` で、空でないこと |
| `crawl_mode` | string | いいえ | クロールモード。`"never"`、`"fallback"`（既定）、`"preferred"`、`"always"` |
| `crawl_timeout` | float | いいえ | クロールのタイムアウト秒数。0 より大きいこと |
| `content` | boolean/object | いいえ | コンテンツ抽出設定。少なくとも 1 つの操作が必要 |
| `abstracts` | boolean/object | いいえ | 要約生成設定 |
| `subpages` | object | いいえ | サブページ抽出設定 |
| `overview` | boolean/object | いいえ | 概要生成設定 |
| `others` | object | いいえ | 関連リンク抽出設定 |

**レスポンス**

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

**レスポンスフィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `request_id` | string | リクエスト識別子 |
| `results` | array | 成功した [FetchResultItem](#fetchresultitem) の配列 |
| `statuses` | array | 各 URL に対応する [FetchStatusItem](#fetchstatusitem) の配列 |

---

### Answer

```http
POST /v1/answer
Authorization: Bearer <token>
```

提供されたコンテンツをもとに、引用付きの根拠ある回答を生成します。

**リクエスト本文**

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

**フィールド**

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `query` | string | はい | 質問またはプロンプト。整形後に空であってはならない |
| `json_schema` | object | いいえ | 構造化出力用 JSON Schema（Draft 2020-12） |
| `content` | boolean | いいえ | 引用に本文を含めるかどうか。既定は `false` |

**レスポンス**

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

**レスポンスフィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `request_id` | string | リクエスト識別子 |
| `answer` | string/object | 生成された回答。`json_schema` がある場合は構造化 JSON |
| `citations` | array | [AnswerCitation](#answercitation) の配列 |

---

### Create Research Task

```http
POST /v1/research
Authorization: Bearer <token>
```

複数ラウンドの調査用に、非同期の研究タスクを作成します。

**リクエスト本文**

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

**フィールド**

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `themes` | string | はい | 研究テーマ。整形後に空であってはならない |
| `search_mode` | string | いいえ | 検索モード。`"research-fast"`、`"research"`（既定）、`"research-pro"` |
| `json_schema` | object | いいえ | 構造化出力用 JSON Schema（Draft 2020-12） |

**レスポンス**

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

**レスポンスフィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `research_id` | string | タスク識別子（16 進文字列） |
| `create_at` | integer | 作成時刻（ミリ秒） |
| `themes` | string | 研究テーマ |
| `search_mode` | string | 使用した検索モード |
| `json_schema` | object | 提供された JSON Schema（ある場合） |
| `status` | string | `pending`、`running`、`completed`、`canceled`、`failed` |
| `output` | object | 完了時の [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完了時刻。終了時のみ設定 |
| `error` | string | エラー内容。`status="failed"` のときのみ |

---

### List Research Tasks

```http
GET /v1/research?cursor=0&limit=10
Authorization: Bearer <token>
```

研究タスクをページネーション付きで一覧取得します。

**クエリパラメータ**

| パラメータ | 型 | 必須 | 説明 |
|------|------|------|------|
| `cursor` | string | いいえ | ページングカーソル。0 以上の整数文字列。既定は `null` |
| `limit` | integer | いいえ | 取得件数。1-50、既定は `10` |

**レスポンス**

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

**レスポンスフィールド**

| フィールド | 型 | 説明 |
|-------|------|------|
| `data` | array | [ResearchTaskResponse](#researchtaskresponse) の配列 |
| `has_more` | boolean | さらに結果があるかどうか |
| `next_cursor` | string | 次ページのカーソル。`has_more=false` の場合は空文字 |

---

### Get Research Task

```http
GET /v1/research/{research_id}
Authorization: Bearer <token>
```

指定した研究タスクの詳細を取得します。

**パスパラメータ**

| パラメータ | 型 | 必須 | 説明 |
|------|------|------|------|
| `research_id` | string | はい | タスク識別子（16 進文字列） |

**レスポンス**

[ResearchTaskResponse](#researchtaskresponse) オブジェクトを返します。

**エラー**

| ステータスコード | 説明 |
|-------------|-------------|
| 404 | 研究タスクが見つからない |

---

### Cancel Research Task

```http
DELETE /v1/research/{research_id}
Authorization: Bearer <token>
```

実行中または待機中の研究タスクをキャンセルします。

**パスパラメータ**

| パラメータ | 型 | 必須 | 説明 |
|------|------|------|------|
| `research_id` | string | はい | タスク識別子（16 進文字列） |

**レスポンス**

`status="canceled"` の [ResearchTaskResponse](#researchtaskresponse) を返します。

**エラー**

| ステータスコード | 説明 |
|-------------|-------------|
| 404 | 研究タスクが見つからない |

---

## 共有型

<a id="fetch-configuration"></a>
### フェッチ設定

`SearchRequest` の `fetchs` フィールドと `FetchRequest` のネストされたフィールドは、同じ設定構造を共有します。

#### FetchContentRequest

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `max_chars` | integer | いいえ | 抽出する最大文字数。0 より大きいこと |
| `detail` | string | いいえ | 詳細レベル。`"concise"`（既定）、`"standard"`、`"full"` |
| `include_markdown_links` | boolean | いいえ | Markdown 形式のリンクを含めるかどうか。既定は `false` |
| `include_html_tags` | boolean | いいえ | HTML タグを保持するかどうか。既定は `false` |
| `include_tags` | array | いいえ | 含める HTML タグ |
| `exclude_tags` | array | いいえ | 除外する HTML タグ |

> 注意: `include_tags` と `exclude_tags` は重複できません。

#### FetchAbstractsRequest

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `query` | string | いいえ | 要約の関連性評価に使うクエリ |
| `max_chars` | integer | いいえ | 要約 1 件あたりの最大文字数。0 より大きいこと |

#### FetchSubpagesRequest

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `max_subpages` | integer | いいえ | 取得する最大サブページ数。0 より大きいこと |
| `subpage_keywords` | string/array | いいえ | サブページを絞り込むキーワード。カンマ区切り文字列または配列 |

#### FetchOverviewRequest

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `query` | string | いいえ | 概要生成を誘導するクエリ |
| `json_schema` | object | いいえ | 構造化概要出力用 JSON Schema |

#### FetchOthersRequest

| フィールド | 型 | 必須 | 説明 |
|-------|------|------|------|
| `max_links` | integer | いいえ | 取得する関連リンクの最大数。0 より大きいこと |
| `max_image_links` | integer | いいえ | 取得する画像リンクの最大数。0 より大きいこと |

> 注意: `max_links` と `max_image_links` の少なくとも一方を設定する必要があります。

---

### レスポンス型

#### FetchResultItem

| フィールド | 型 | 説明 |
|-------|------|------|
| `url` | string | ソース URL |
| `title` | string | ページタイトル |
| `published_date` | string | 公開日。ISO 8601 形式 |
| `author` | string | 著者名 |
| `image` | string | メイン画像 URL |
| `favicon` | string | ファビコン URL |
| `content` | string | 抽出されたコンテンツ |
| `abstracts` | array | 関連する要約の一覧 |
| `abstract_scores` | array | 各要約の関連スコア。`abstracts` と対応 |
| `overview` | string/object | 生成された概要。未指定なら `null` |
| `subpages` | array | [FetchSubpagesResult](#fetchsubpagesresult) の配列 |
| `others` | object | [FetchOthersResult](#fetchothersresult) 。未指定なら `null` |

#### FetchSubpagesResult

| フィールド | 型 | 説明 |
|-------|------|------|
| `url` | string | サブページ URL |
| `title` | string | サブページタイトル |
| `published_date` | string | 公開日。ISO 8601 形式 |
| `author` | string | 著者名 |
| `image` | string | 画像 URL |
| `favicon` | string | ファビコン URL |
| `content` | string | 抽出されたコンテンツ |
| `abstracts` | array | 要約一覧 |
| `abstract_scores` | array | 関連スコア |
| `overview` | string/object | 生成された概要 |

#### FetchOthersResult

| フィールド | 型 | 説明 |
|-------|------|------|
| `links` | array | ページ内で見つかった関連リンク |
| `image_links` | array | ページ内で見つかった画像リンク |

#### FetchStatusItem

| フィールド | 型 | 説明 |
|-------|------|------|
| `url` | string | 処理中の URL |
| `status` | string | `"success"` または `"error"` |
| `error` | object | [FetchStatusError](#fetchstatuserror)。成功時は `null` |

#### FetchStatusError

| フィールド | 型 | 説明 |
|-------|------|------|
| `tag` | string | エラータグ。`"CRAWL_NOT_FOUND"`、`"CRAWL_TIMEOUT"`、`"CRAWL_LIVECRAWL_TIMEOUT"`、`"SOURCE_NOT_AVAILABLE"`、`"UNSUPPORTED_URL"`、`"CRAWL_UNKNOWN_ERROR"` |
| `detail` | string | エラー詳細メッセージ。任意 |

#### AnswerCitation

| フィールド | 型 | 説明 |
|-------|------|------|
| `id` | string | 引用識別子 |
| `url` | string | ソース URL |
| `title` | string | ソースタイトル |
| `content` | string | 引用本文。リクエストで `content=true` の場合のみ任意で返却 |

#### ResearchResponse

| フィールド | 型 | 説明 |
|-------|------|------|
| `content` | string | 研究レポート本文 |
| `structured` | object | 提供された `json_schema` に一致する構造化出力。任意 |

#### ResearchTaskResponse

| フィールド | 型 | 説明 |
|-------|------|------|
| `research_id` | string | タスク識別子 |
| `create_at` | integer | 作成時刻（ミリ秒） |
| `themes` | string | 研究テーマ |
| `search_mode` | string | 検索モード。`"research-fast"`、`"research"`、`"research-pro"` |
| `json_schema` | object | 出力用 JSON Schema。未提供なら `null` |
| `status` | string | `pending`、`running`、`completed`、`canceled`、`failed` |
| `output` | object | 完了時の [ResearchResponse](#researchresponse) |
| `finished_at` | integer | 完了時刻。未完了時は `null` |
| `error` | string | エラーメッセージ。`status="failed"` の場合のみ |

---

## 列挙型

### SearchMode

| 値 | 説明 |
|----|------|
| `fast` | 迅速な検索。処理は最小限 |
| `auto` | バランス型検索。自動最適化。既定値 |
| `deep` | 複数クエリを使う深い検索。結果も詳細 |

### ResearchSearchMode

| 値 | 説明 |
|----|------|
| `research-fast` | 最小ラウンドの高速調査 |
| `research` | 標準のマルチラウンド調査。既定値 |
| `research-pro` | 分析を厚くした包括的な調査 |

### CrawlMode

| 値 | 説明 |
|----|------|
| `never` | クロールしない。検索プロバイダの結果のみ使用 |
| `fallback` | 検索結果が不十分な場合のみクロール。既定値 |
| `preferred` | 検索よりクロールを優先 |
| `always` | 常にページを直接クロール |

### FetchContentDetail

| 値 | 説明 |
|----|------|
| `concise` | 簡潔な抽出。既定値 |
| `standard` | 標準的な抽出 |
| `full` | 完全な抽出 |

### FetchContentTag

コンテンツフィルタリング用の HTML セクションタグです。

| 値 | 説明 |
|----|------|
| `header` | ページヘッダー |
| `navigation` | ナビゲーション要素 |
| `banner` | バナー内容 |
| `body` | 本文 |
| `sidebar` | サイドバー |
| `footer` | フッター |
| `metadata` | メタデータ |

### ResearchTaskStatus

| 値 | 説明 |
|----|------|
| `pending` | タスクが作成され、開始待ち |
| `running` | タスクが実行中 |
| `completed` | タスクが正常終了 |
| `canceled` | タスクがユーザーによりキャンセル |
| `failed` | タスクがエラーで終了 |

---

## エラー応答

### HTTP ステータスコード

| ステータスコード | 説明 |
|-------------|-------------|
| 200 | 成功 |
| 400 | 不正なリクエスト（パラメータが無効） |
| 401 | 認証失敗（Bearer Token の欠如または不正） |
| 404 | リソースが存在しない |
| 500 | サーバー内部エラー |
| 503 | サービス利用不可（エンジン未準備） |

### エラー応答形式

すべてのエラー応答は次の形式です。

```json
{
  "detail": "Error message describing the issue"
}
```

---

## 使用例

### cURL 例

#### ヘルスチェック

```bash
curl -X GET http://localhost:8000/healthz
```

**レスポンス:**

```json
{
  "status": "ok",
  "engine_ready": true
}
```

---

#### コンテンツ付き検索

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

**レスポンス:**

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

#### URL を取得

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

**レスポンス:**

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

#### 回答を生成

```bash
curl -X POST http://localhost:8000/v1/answer \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key contributions",
    "content": true
  }'
```

**レスポンス:**

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

#### 研究タスクを作成

```bash
curl -X POST http://localhost:8000/v1/research \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "themes": "State of multimodal AI in 2024",
    "search_mode": "research"
  }'
```

**レスポンス:**

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

#### 研究ステータスを確認

```bash
curl -X GET http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**レスポンス（running）:**

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

**レスポンス（completed）:**

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

#### 研究をキャンセル

```bash
curl -X DELETE http://localhost:8000/v1/research/a1b2c3d4e5f6 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**レスポンス:**

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

#### 研究タスク一覧

```bash
curl -X GET "http://localhost:8000/v1/research?limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**レスポンス:**

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

## OpenAPI ドキュメント

設定の `api.enable_docs` を `true` にすると、対話式の API ドキュメントを利用できます。

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## レート制限

これは単一ユーザーまたは信頼できるプライベート環境向けの個人 API です。内蔵のレート制限はありません。広く公開する場合は外部で制限を設定してください。

---

## バージョニング

すべての API エンドポイントは `/v1/` プレフィックスを使います。将来のバージョンは `/v2/` などを使用し、可能な限り後方互換を維持します。
