from __future__ import annotations

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return 0


def _runtime() -> Runtime:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "use_model": "gpt-4.1-mini",
                "models": [{"name": "gpt-4.1-mini", "backend": "openai"}],
            }
        }
    )
    clock = _Clock()
    return Runtime(
        settings=settings,
        telemetry=NoopTelemetry(),
        clock=clock,
    )


def test_markdown_extracts_main_content() -> None:
    html = (
        b"<html><body>"
        b"<header>Header</header><nav>Menu</nav>"
        b"<main><h1>Main Title</h1><p>Hello world body.</p><p>More content.</p></main>"
        b"<footer>Footer</footer>"
        b"</body></html>"
    )
    ex = MarkdownExtractor(rt=_runtime())
    doc = ex.extract(
        url="https://example.com/x", content=html, content_type="text/html"
    )
    assert doc.content_kind == "html"
    assert "Main Title" in doc.markdown
    assert "Hello world body." in doc.plain_text
    assert "Menu" not in doc.markdown


def test_pdf_extension_is_not_default_noise_extension() -> None:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "use_model": "gpt-4.1-mini",
                "models": [{"name": "gpt-4.1-mini", "backend": "openai"}],
            }
        }
    )
    assert "pdf" not in settings.pipeline.profiles["general"].noise_extensions
