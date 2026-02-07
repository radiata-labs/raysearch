from __future__ import annotations

from search_core.utils import TextUtils


def test_normalize_nfkc():
    text = "ＡＢＣ１２３"
    assert TextUtils.normalize_text(text) == "abc123"


def test_tokenize_chinese_mixed():
    tokens = TextUtils.tokenize("中文A/B测试2025")
    assert "中文" in tokens or "测试" in tokens
    assert "a" in tokens
    assert "2025" in tokens


def test_stopwords_filtered():
    tokens = TextUtils.tokenize("这是的了呢")
    assert "的" not in tokens
    assert "了" not in tokens
    assert "呢" not in tokens


def test_ngrams_added():
    tokens = TextUtils.tokenize("中文测试用例")
    # Expect some 2-gram/3-gram tokens
    assert any(len(t) == 2 for t in tokens)
    assert any(len(t) == 3 for t in tokens)

