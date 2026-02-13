import time
from pathlib import Path

from marisa_trie import Trie

STOPWORDS_PATH = Path(__file__).parent / "files"
_stopwords: list[str] = []
_time_start = time.process_time()
print(f"Loading stopwords from {STOPWORDS_PATH.as_posix()}")
for file_path in STOPWORDS_PATH.glob("*.txt"):
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "big5"]
    content = None
    for enc in encodings:
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
            _stopwords.append(word)
_time_end = time.process_time()
if _stopwords:
    print(
        f"Loading {_stopwords.__len__()} stopwords took {_time_end - _time_start:.3f} seconds"
    )
else:
    print("No stopwords loaded")

stopwords = Trie(_stopwords)


def is_stopword(word: str) -> bool:
    return word in stopwords


def filter_stopwords(words: list[str]) -> list[str]:
    return [w for w in words if w and not is_stopword(w)]


__all__ = ["stopwords", "is_stopword", "filter_stopwords"]
