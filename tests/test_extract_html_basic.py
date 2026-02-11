from __future__ import annotations

from serpsage.components.extract.html_basic import BasicHtmlExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


def test_basic_html_extractor_drops_nav_footer():
    html = b"""<!doctype html>
    <html><body>
      <nav>MENU SHOULD DROP</nav>
      <article><h1>Hello</h1><p>Main content here.</p></article>
      <footer>FOOTER SHOULD DROP</footer>
    </body></html>"""
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    ex = BasicHtmlExtractor(rt=rt)
    out = ex.extract(url="https://x", content=html, content_type="text/html")
    text = out.text.lower()
    assert "main content" in text
    assert "menu should drop" not in text
    assert "footer should drop" not in text
