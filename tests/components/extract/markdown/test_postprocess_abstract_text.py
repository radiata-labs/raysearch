from __future__ import annotations

from serpsage.components.extract.markdown.postprocess import markdown_to_abstract_text


def test_markdown_to_abstract_text_removes_links_images_urls_and_noise() -> None:
    markdown = """
# DeepSeek V3.2

[Skip to main content](https://example.com/main)

Section Title
====

Please read [API Docs](https://api.example.com/docs) and <https://example.com/raw-link>.

![](https://example.com/image.png)

https://example.com/standalone

---

Here is **bold** and __under__ with `inline_code`.

```python
print("should be removed")
```
"""

    result = markdown_to_abstract_text(markdown)
    lines = result.splitlines()

    assert "DeepSeek V3.2" in lines
    assert "Section Title" in lines
    assert "Please read API Docs and ." in lines
    assert "Here is bold and under with inline_code." in lines
    assert "Skip to main content" not in result
    assert "https://" not in result
    assert "image.png" not in result
    assert "print(\"should be removed\")" not in result
    assert "\n" in result
