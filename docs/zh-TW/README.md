![cover-v5-optimized](../../images/GitHub_README.png)

<div align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</div>

# RaySearch

RaySearch 是一個異步優先的搜尋編排引擎，用於在多個提供商、爬蟲、提取器、排序器和 LLM 後端之上構建 AI 概覽式工作流。

它提供四個高層管道：

- `search`：多提供商檢索，可選抓取和重排序階段
- `fetch`：頁面爬取、提取、摘要、概覽生成和相關連結
- `answer`：搜尋加帶引用的答案生成
- `research`：多輪研究報告生成，包含綜合和結構化輸出

## 為什麼選擇 RaySearch

- 基於元件的架構，支援可插拔的提供商、爬蟲、提取器、排序器、快取和 LLM 客戶端
- 純異步執行時，單一 `Engine` 入口點
- YAML/JSON 設定載入器，支援環境變數注入提供商和模型密鑰
- 內建追蹤和計量輸出，便於可觀測性
- 專為搜尋密集型和研究密集型代理工作流設計，而非純聊天場景

## 安裝

核心安裝：

```bash
uv pip install raysearch
```

常用完整安裝：

```bash
uv pip install "raysearch[extract,extract_pdf,crawl,rank,cache,api,overview,tracking]"
```

使用 Playwright 爬取時，需單獨安裝瀏覽器二進位：

```bash
playwright install
```

## 公開 API

```python
from raysearch import Engine, SearchRequest, load_settings
```

主要入口點：

- `load_settings(path=None, env=None)`
- `Engine.from_settings(setting_file=None, *, settings=None, overrides=None)`
- `await engine.search(request)`
- `await engine.fetch(request)`
- `await engine.answer(request)`
- `await engine.research(request)`

## 快速開始

```python
from raysearch import Engine, SearchRequest

async def main() -> None:
    async with Engine.from_settings("demo/search_config_example.yaml") as engine:
        response = await engine.search(
            SearchRequest(
                query="latest multimodal model papers",
                mode="deep",
                max_results=8,
            )
        )
        for item in response.results:
            print(item.title, item.url)
```

## 配置

RaySearch 按以下順序載入設定：

1. 傳遞給 `load_settings(...)` 的顯式 `path`
2. `RAYSEARCH_CONFIG_PATH` 環境變數
3. `raysearch.yaml` 檔案
4. 程式碼內預設值

主要配置組：

- `components`：provider、crawl、extract、rank、llm、cache、tracking、metering、http 和速率限制
- `telemetry`：追蹤和計量發射器行為
- `search`：搜尋模式配置和查詢擴展行為
- `fetch`：提取、摘要和概覽調優
- `answer`：規劃和生成模型選擇
- `research`：報告生成預算和模型路由
- `runner`：並發和佇列限制

元件族採用簡單的預設加實例結構：

```yaml
components:
  provider:
    default: google
    google:
      enabled: true
      cookies:
        CONSENT: "YES+"
    duckduckgo:
      enabled: true
      base_url: https://html.duckduckgo.com/html
      allow_redirects: false
```

參考配置：

- `demo/search_config_example.yaml`

## 提供商與管道

內建提供商覆蓋：

- `google`
- `google_news`
- `duckduckgo`
- `searxng`
- `github`
- `reddit`
- `reuters`
- `openalex`
- `semantic_scholar`
- `wikidata`
- `wikipedia`
- `arxiv`
- `marginalia`
- `blend` 用於組合多個提供商

內建管道支援：

- 搜尋結果擴展和重排序
- Markdown 優先的抓取提取
- 摘要生成和頁面概覽綜合
- 帶引用的答案生成
- 多輪研究報告生成

## 環境變數

載入器在 `AppSettings.runtime_env` 中保留完整的進程環境，元件配置模型按需從中提取值。

常用示例：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- 提供商特定覆蓋，如 `GITHUB_TOKEN` 或 `SEARXNG_BASE_URL`

## 追蹤與計量

追蹤和計量獨立於請求管道配置。

預設產物名稱遵循套件名：

- 追蹤 JSONL：`.raysearch_tracking.jsonl`
- 計量 JSONL：`.raysearch_metering.jsonl`
- 計量 SQLite：`.raysearch_metering.sqlite3`
- 快取 SQLite：`.raysearch_cache.sqlite3`

## 開發

倉庫包含可執行的範例：

- `demo/search.py`
- `demo/fetch.py`
- `demo/answer.py`
- `demo/research.py`

範例配置：

- `demo/search_config_example.yaml`

## 注意事項

- `search.mode` 支援 `fast`、`auto` 和 `deep`
- RaySearch僅支援異步
- 元件發現從 `raysearch.components` 載入
- JS 密集型爬取需要 Playwright 和已安裝的瀏覽器