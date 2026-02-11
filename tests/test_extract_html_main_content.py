from __future__ import annotations

from serpsage.components.extract.html_main import MainContentHtmlExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


def test_main_content_extractor_prefers_mediawiki_content_and_drops_noise():
    html = b"""<!doctype html>
    <html><body>
      <div id="mw-navigation">MENU SHOULD DROP</div>
      <div id="toc">TOC SHOULD DROP</div>
      <div id="mw-content-text">
        <p>Main content here.</p>
        <p>Second paragraph.</p>
      </div>
      <footer>FOOTER SHOULD DROP</footer>
    </body></html>"""
    settings = AppSettings.model_validate(
        {"enrich": {"extractor": {"backend": "main_content"}}}
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    ex = MainContentHtmlExtractor(rt=rt)
    out = ex.extract(url="https://x", content=html, content_type="text/html")
    text = out.text.lower()
    assert "main content here" in text
    assert "menu should drop" not in text
    assert "toc should drop" not in text
    assert "footer should drop" not in text


def test_main_content_extractor_applies_fixed_max_chars_budget():
    body = "<p>" + ("hello " * 20_000) + "</p>"
    html = f"<html><body><main>{body}</main></body></html>".encode("utf-8")
    settings = AppSettings.model_validate(
        {"enrich": {"extractor": {"backend": "main_content"}}}
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    ex = MainContentHtmlExtractor(rt=rt)
    out = ex.extract(url="https://x", content=html, content_type="text/html")
    assert len(out.text) <= 50_000
