![cover-v5-optimized](../../images/GitHub_README.png)

<div align="center">
  <a href="../../README.md"><img alt="README in English" src="https://img.shields.io/badge/English-d9d9d9"></a>
  <a href="../zh-TW/README.md"><img alt="繁體中文文件" src="https://img.shields.io/badge/繁體中文-d9d9d9"></a>
  <a href="../zh-CN/README.md"><img alt="简体中文文件" src="https://img.shields.io/badge/简体中文-d9d9d9"></a>
  <a href="../ja-JP/README.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-d9d9d9"></a>
</div>

# RaySearch

RaySearch は、複数のプロバイダー、クローラー、抽出器、ランカー、LLM バックエンドを組み合わせて AI オーバービュー形式のワークフローを構築するための、非同期ファーストの検索オーケストレーションエンジンです。

4つの高レベルパイプラインを提供します：

- `search`：複数プロバイダー検索、オプションのフェッチとリランキングステージ
- `fetch`：ページクローリング、抽出、要約、オーバービュー生成、関連リンク
- `answer`：検索と引用付き回答生成
- `research`：統合と構造化出力を含む多回合計レポート生成

## RaySearchを選ぶ理由

- コンポーネントベースのアーキテクチャ、プラグイン可能なプロバイダー、クローラー、抽出器、ランカー、キャッシュ、LLM クライアント
- 完全非同期ランタイム、単一の `Engine` エントリーポイント
- YAML/JSON 設定ローダー、プロバイダーとモデルのシークレットに環境変数注入
- 可観測性のための内蔵トラッキングとメータリング出力
- 検索集中型・研究集中型エージェントワークフロー向け設計（チャットのみの用途ではない）

## インストール

コアインストール：

```bash
uv pip install raysearch
```

一般的な完全インストール：

```bash
uv pip install "raysearch[extract,extract_pdf,crawl,rank,cache,api,overview,tracking]"
```

Playwright ベースのクローリング使用時は、ブラウザバイナリを別途インストール：

```bash
playwright install
```

## 公開 API

```python
from raysearch import Engine, SearchRequest, load_settings
```

主要エントリーポイント：

- `load_settings(path=None, env=None)`
- `Engine.from_settings(setting_file=None, *, settings=None, overrides=None)`
- `await engine.search(request)`
- `await engine.fetch(request)`
- `await engine.answer(request)`
- `await engine.research(request)`

## クイックスタート

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

## 設定

RaySearch は以下の順序で設定を読み込みます：

1. `load_settings(...)` に渡された明示的な `path`
2. `RAYSEARCH_CONFIG_PATH` 環境変数
3. `raysearch.yaml` ファイル
4. コード内デフォルト値

主要設定グループ：

- `components`：provider、crawl、extract、rank、llm、cache、tracking、metering、http、レート制限
- `telemetry`：トラッキングとメータリングエミッターの動作
- `search`：検索モードプロファイルとクエリ拡張動作
- `fetch`：抽出、要約、オーバービュー調整
- `answer`：計画と生成モデル選択
- `research`：レポート生成バジェットとモデルルーティング
- `runner`：並行性とキューリミット

コンポーネントファミリーはシンプルなデフォルト＋インスタンス構造を使用：

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

参考設定：

- `demo/search_config_example.yaml`

## プロバイダーとパイプライン

内蔵プロバイダー対応：

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
- `blend`（複数プロバイダー統合用）

内蔵パイプライン対応：

- 検索結果拡張とリランキング
- Markdownファーストのフェッチ抽出
- 要約生成とページオーバービュー統合
- 引用付き回答生成
- 多回合計レポート生成

## 環境変数

ローダーは `AppSettings.runtime_env` に完全なプロセス環境を保持し、コンポーネント設定モデルが必要時にそこから値を取得します。

一般的な例：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- プロバイダー固有のオーバーライド（`GITHUB_TOKEN`、`SEARXNG_BASE_URL` 等）

## トラッキングとメータリング

トラッキングとメータリングはリクエストパイプラインから独立して設定されます。

デフォルトアーティファクト名はパッケージ名に従います：

- トラッキング JSONL：`.raysearch_tracking.jsonl`
- メータリング JSONL：`.raysearch_metering.jsonl`
- メータリング SQLite：`.raysearch_metering.sqlite3`
- キャッシュ SQLite：`.raysearch_cache.sqlite3`

## 開発

リポジトリには実行可能なデモが含まれます：

- `demo/search.py`
- `demo/fetch.py`
- `demo/answer.py`
- `demo/research.py`

サンプル設定：

- `demo/search_config_example.yaml`

## 注意事項

- `search.mode` は `fast`、`auto`、`deep` をサポート
- RaySearch は非同期専用
- コンポーネント探索は `raysearch.components` から読み込み
- JS集中型クローリングには Playwright とインストール済みブラウザが必要