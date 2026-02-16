from __future__ import annotations

from serpsage.components.extract.markdown.dom import cleanup_dom, parse_html_document
from serpsage.components.extract.markdown.postprocess import markdown_to_text
from serpsage.components.extract.markdown.render import render_markdown


def _render_markdown(html_doc: str, *, base_url: str = "https://example.com") -> str:
    soup = parse_html_document(html_doc)
    cleanup_dom(soup)
    root = soup.body if soup.body is not None else soup
    markdown, _ = render_markdown(root=root, base_url=base_url)
    return markdown


def _render_text(html_doc: str, *, base_url: str = "https://example.com") -> str:
    markdown = _render_markdown(html_doc, base_url=base_url)
    return markdown_to_text(markdown)


def test_cleanup_keeps_nested_api_like_div_text() -> None:
    html_doc = """
    <div class="api-section">
      <div class="api-section-heading flex flex-col gap-y-4 w-full">
        <div class="flex items-baseline border-b pb-2.5 border-gray-100 dark:border-gray-800 w-full">
          <h4 class="api-section-heading-title flex-1 mb-0">Authorizations</h4>
          <div class="flex items-center"></div>
        </div>
      </div>
      <div class="primitive-param-field border-gray-100 dark:border-gray-800 border-b last:border-b-0">
        <div class="py-6">
          <div class="flex font-mono text-sm group/param-head param-head break-all relative" id="authorization-x-api-key">
            <div class="flex-1 flex flex-col content-start py-0.5 mr-5">
              <div class="flex items-center flex-wrap gap-2">
                <div class="font-semibold text-primary dark:text-primary-light" data-component-part="field-name">x-api-key</div>
                <div class="inline items-center gap-2 text-xs font-medium" data-component-part="field-meta">
                  <div class="flex items-center px-2 py-0.5 rounded-md" data-component-part="field-info-pill"><span>string</span></div>
                  <div class="flex items-center px-2 py-0.5 rounded-md" data-component-part="field-info-pill"><span>header</span></div>
                  <div class="px-2 py-0.5 rounded-md" data-component-part="field-required-pill">required</div>
                </div>
              </div>
            </div>
          </div>
          <div class="mt-4">
            <div class="space-y-4 whitespace-normal prose prose-sm prose-gray dark:prose-invert overflow-wrap-anywhere">
              <p class="whitespace-pre-line">API key can be provided either via x-api-key header or Authorization header with Bearer scheme</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    """

    text = _render_text(html_doc)

    assert "Authorizations" in text
    assert "x-api-key" in text
    assert "string" in text
    assert "header" in text
    assert "required" in text
    assert (
        "API key can be provided either via x-api-key header or Authorization header with Bearer scheme"
        in text
    )


def test_cleanup_still_removes_explicit_ad_blocks() -> None:
    html_doc = """
    <div class="article-body">
      <p>Main body content stays.</p>
    </div>
    <div class="ad-slot sponsored">Buy now ad copy</div>
    <div data-ad-slot="top-banner">Sponsored widget</div>
    """

    text = _render_text(html_doc)

    assert "Main body content stays." in text
    assert "Buy now ad copy" not in text
    assert "Sponsored widget" not in text


def test_cleanup_still_removes_search_blocks() -> None:
    html_doc = """
    <main>
      <div role="search">Search input and filters</div>
      <div class="site-search-form">Type to search docs</div>
      <p>Primary content paragraph.</p>
    </main>
    """

    text = _render_text(html_doc)

    assert "Primary content paragraph." in text
    assert "Search input and filters" not in text
    assert "Type to search docs" not in text


def test_heading_token_not_treated_as_ad() -> None:
    html_doc = """
    <div class="doc-shell">
      <div class="heading">Heading should stay visible</div>
      <div class="param-head">Parameter head should stay visible</div>
      <div class="content">Body paragraph text.</div>
    </div>
    """

    text = _render_text(html_doc)

    assert "Heading should stay visible" in text
    assert "Parameter head should stay visible" in text
    assert "Body paragraph text." in text


def test_empty_same_page_anchor_not_rendered_as_autolink() -> None:
    html_doc = """
    <div class="api-section">
      <h4>Authorizations</h4>
      <a href="#authorization-x-api-key" aria-label="Navigate to header"></a>
      <div data-component-part="field-name">x-api-key</div>
      <div data-component-part="field-meta">
        <div data-component-part="field-info-pill"><span>string</span></div>
        <div data-component-part="field-info-pill"><span>header</span></div>
        <div data-component-part="field-required-pill">required</div>
      </div>
    </div>
    """

    markdown = _render_markdown(
        html_doc,
        base_url="https://exa.ai/docs/reference/get-contents",
    )

    assert "#authorization-x-api-key" not in markdown
    assert (
        "<https://exa.ai/docs/reference/get-contents#authorization-x-api-key>"
        not in markdown
    )


def test_compact_metadata_divs_render_inline() -> None:
    html_doc = """
    <div class="api-section">
      <div class="api-section-heading"><h4>Authorizations</h4></div>
      <div class="primitive-param-field">
        <div class="param-head" id="authorization-x-api-key">
          <div data-component-part="field-name">x-api-key</div>
          <div class="inline items-center gap-2" data-component-part="field-meta">
            <div data-component-part="field-info-pill"><span>string</span></div>
            <div data-component-part="field-info-pill"><span>header</span></div>
            <div data-component-part="field-required-pill">required</div>
          </div>
        </div>
      </div>
    </div>
    """

    markdown = _render_markdown(html_doc)
    text = markdown_to_text(markdown)

    assert "x-api-key string header required" in text
    assert "x-api-key\nstring" not in text
