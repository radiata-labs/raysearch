from __future__ import annotations

from pathlib import Path

STOPWORDS_PATH = Path(__file__).parent / "files"

try:
    from marisa_trie import Trie  # type: ignore[import-not-found]

    _TRIE_AVAILABLE = True
except Exception:  # noqa: BLE001
    Trie = None  # type: ignore[assignment]
    _TRIE_AVAILABLE = False


def _load_stopwords() -> list[str]:
    words: list[str] = []
    for file_path in STOPWORDS_PATH.glob("*.txt"):
        content: list[str] | None = None
        for enc in ("utf-8", "gbk", "gb2312", "gb18030", "big5"):
            try:
                with file_path.open(encoding=enc) as f:
                    content = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            continue
        for line in content:
            word = line.strip()
            if word:
                words.append(word)
    return words


_stopwords = _load_stopwords()

if _TRIE_AVAILABLE and Trie is not None:
    stopwords = Trie(_stopwords)
else:
    stopwords = set(_stopwords)


def is_stopword(word: str) -> bool:
    return bool(word and word in stopwords)


def filter_stopwords(words: list[str]) -> list[str]:
    return [w for w in words if w and not is_stopword(w)]


__all__ = ["stopwords", "is_stopword", "filter_stopwords"]
