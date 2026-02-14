from __future__ import annotations

import time

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return MarkdownExtractor(rt=rt)


def _first_fenced_block(markdown: str) -> tuple[str, str, str]:
    lines = markdown.splitlines()
    in_code = False
    fence = ""
    info = ""
    body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            ticks = len(stripped) - len(stripped.lstrip("`"))
            if not in_code:
                in_code = True
                fence = "`" * ticks
                info = stripped[ticks:].strip()
                continue
            if ticks >= len(fence):
                return fence, info, "\n".join(body)
        if in_code:
            body.append(line)

    return "", "", ""


def test_fenced_code_preserves_whitespace_and_duplicate_lines() -> None:
    extractor = _build_extractor()
    code = (
        "if True:\n"
        "    print('x')\n"
        "    print('x')\n"
        "print('``` marker')\n"
        "value  =  42"
    )
    html_doc = (
        "<html><body><main><h1>Code</h1><pre><code class='language-python'>"
        f"{code}"
        "</code></pre></main></body></html>"
    ).encode()

    out = extractor.extract(
        url="https://example.com/code",
        content=html_doc,
        content_type="text/html",
        include_secondary_content=False,
        collect_links=False,
    )

    fence, info, body = _first_fenced_block(out.markdown)
    assert fence == "````"
    assert info == "python"
    assert body == code
    assert "    print('x')" in out.markdown
    assert "value  =  42" in out.markdown
