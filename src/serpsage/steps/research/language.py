from __future__ import annotations

import re

LanguageCode = str

LANGUAGE_CODES: set[LanguageCode] = {
    "zh-Hans",
    "zh-Hant",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "ru",
    "ar",
    "hi",
    "tr",
    "other",
}

_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_KANA = re.compile(r"[\u3040-\u30ff]")
_RE_HANGUL = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_RE_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_RE_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_RE_ARABIC = re.compile(r"[\u0600-\u06ff]")
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097f]")

_TRADITIONAL_HINT_CHARS = set(
    "繁體臺萬與為於後發說點電體學習驗證應該實際權限開啟設定資源導覽機會"
)
_SIMPLIFIED_HINT_CHARS = set(
    "简体台万与为于后发说点电体学习验证应该实际权限开启设置资源导航机会"
)

_LANG_ALIAS: dict[str, LanguageCode] = {
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-hans": "zh-Hans",
    "zh-sg": "zh-Hans",
    "chinese": "zh-Hans",
    "simplified chinese": "zh-Hans",
    "chinese simplified": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hk": "zh-Hant",
    "zh-hant": "zh-Hant",
    "traditional chinese": "zh-Hant",
    "chinese traditional": "zh-Hant",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "ja": "ja",
    "ja-jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "ko-kr": "ko",
    "korean": "ko",
    "fr": "fr",
    "fr-fr": "fr",
    "french": "fr",
    "de": "de",
    "de-de": "de",
    "german": "de",
    "es": "es",
    "es-es": "es",
    "spanish": "es",
    "pt": "pt",
    "pt-pt": "pt",
    "pt-br": "pt",
    "portuguese": "pt",
    "it": "it",
    "it-it": "it",
    "italian": "it",
    "ru": "ru",
    "ru-ru": "ru",
    "russian": "ru",
    "ar": "ar",
    "ar-sa": "ar",
    "arabic": "ar",
    "hi": "hi",
    "hi-in": "hi",
    "hindi": "hi",
    "tr": "tr",
    "tr-tr": "tr",
    "turkish": "tr",
    "other": "other",
}

_FR_HINTS = (
    " le ",
    " la ",
    " les ",
    " de ",
    " des ",
    " est ",
    " avec ",
    " pour ",
    " guide ",
)
_DE_HINTS = (
    " der ",
    " die ",
    " das ",
    " und ",
    " mit ",
    " für ",
    " wie ",
    " anleitung ",
)
_ES_HINTS = (
    " el ",
    " la ",
    " los ",
    " las ",
    " de ",
    " para ",
    " como ",
    " guía ",
)
_PT_HINTS = (
    " o ",
    " a ",
    " os ",
    " as ",
    " de ",
    " para ",
    " como ",
    " guia ",
)
_IT_HINTS = (
    " il ",
    " lo ",
    " la ",
    " gli ",
    " delle ",
    " per ",
    " come ",
    " guida ",
)
_TR_HINTS = (
    " ve ",
    " için ",
    " nasıl ",
    " rehber ",
    " kullan ",
)

_GLOBAL_TECH_HINTS = (
    "spacex",
    "openai",
    "anthropic",
    "google",
    "meta",
    "microsoft",
    "apple",
    "nvidia",
    "tesla",
    "github",
    "arxiv",
    "paper",
    "论文",
    "技術",
    "技术",
    "official docs",
    "documentation",
    "doc",
    "api",
    "sdk",
    "framework",
    "model",
    "llm",
    "benchmark",
    "release note",
    "open source",
    "开源",
    "標準",
    "标准",
    "spec",
    "repository",
    "kubernetes",
    "docker",
    "python",
    "java",
    "rust",
)

_LOCAL_LIFE_HINTS = (
    "旅行",
    "旅游",
    "旅遊",
    "攻略",
    "景点",
    "景點",
    "美食",
    "酒店",
    "住宿",
    "签证",
    "簽證",
    "交通",
    "city guide",
    "travel guide",
    "itinerary",
    "local food",
    "trip plan",
    "市政",
    "政务",
    "政務",
    "政策",
    "hospital",
    "school",
)


def normalize_language_code(raw: object, *, default: LanguageCode = "other") -> str:
    token = str(raw or "").strip().casefold()
    if not token:
        return default
    if token in _LANG_ALIAS:
        return _LANG_ALIAS[token]
    return default


def detect_input_language(text: str) -> str:
    token = text.strip()
    if not token:
        return "other"
    if _RE_KANA.search(token):
        return "ja"
    if _RE_HANGUL.search(token):
        return "ko"
    if _RE_CYRILLIC.search(token):
        return "ru"
    if _RE_ARABIC.search(token):
        return "ar"
    if _RE_DEVANAGARI.search(token):
        return "hi"
    if _RE_CJK.search(token):
        return _detect_chinese_variant(token)
    latin_guess = _detect_latin_language(token)
    if latin_guess:
        return latin_guess
    return "other"


def route_search_language(*, theme: str, input_language: str) -> str:
    normalized_input = normalize_language_code(input_language, default="other")
    content = theme.strip().casefold()
    if not content:
        return normalized_input if normalized_input != "other" else "en"
    if _looks_local_life(content):
        return normalized_input if normalized_input != "other" else "en"
    if _looks_global_tech(content):
        return "en"
    return normalized_input if normalized_input != "other" else "en"


def map_provider_language_param(
    *, provider_backend: str, search_language: str
) -> dict[str, str]:
    backend = provider_backend.strip().casefold()
    normalized = normalize_language_code(search_language, default="other")
    if backend != "searxng":
        return {}
    mapping: dict[str, str] = {
        "en": "en-US",
        "zh-Hans": "zh-CN",
        "zh-Hant": "zh-TW",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "pt": "pt-PT",
        "it": "it-IT",
        "ru": "ru-RU",
        "tr": "tr-TR",
        "ar": "ar-SA",
        "hi": "hi-IN",
    }
    target = mapping.get(normalized)
    if not target:
        return {}
    return {"language": target}


def language_alignment_score(*, text: str, target_language: str) -> float:
    target = normalize_language_code(target_language, default="other")
    detected = detect_input_language(text)
    if target == "other":
        return 1.0 if text.strip() else 0.0
    if detected == target:
        return 1.0
    if {detected, target} <= {"zh-Hans", "zh-Hant"}:
        return 0.85
    latin_family = {"en", "fr", "de", "es", "pt", "it", "tr"}
    if detected in latin_family and target in latin_family:
        if target == "en" and detected != "en":
            return 0.5
        return 0.65
    if detected == "other":
        return 0.4
    return 0.0


def document_language_alignment(*, text: str, target_language: str) -> float:
    content = text.strip()
    if not content:
        return 1.0
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return 1.0
    sample = lines[:24]
    scores = [
        language_alignment_score(text=item, target_language=target_language)
        for item in sample
    ]
    if not scores:
        return 1.0
    return float(sum(scores) / float(len(scores)))


def _detect_chinese_variant(text: str) -> str:
    trad_hits = sum(1 for ch in text if ch in _TRADITIONAL_HINT_CHARS)
    simp_hits = sum(1 for ch in text if ch in _SIMPLIFIED_HINT_CHARS)
    if trad_hits > simp_hits:
        return "zh-Hant"
    return "zh-Hans"


def _detect_latin_language(text: str) -> str | None:
    token = f" {text.strip().casefold()} "
    if not _RE_LATIN.search(token):
        return None
    if _contains_any(text=token, markers=_FR_HINTS):
        return "fr"
    if _contains_any(text=token, markers=_DE_HINTS):
        return "de"
    if _contains_any(text=token, markers=_ES_HINTS):
        return "es"
    if _contains_any(text=token, markers=_PT_HINTS):
        return "pt"
    if _contains_any(text=token, markers=_IT_HINTS):
        return "it"
    if _contains_any(text=token, markers=_TR_HINTS):
        return "tr"
    return "en"


def _looks_global_tech(content: str) -> bool:
    return _contains_any(text=content, markers=_GLOBAL_TECH_HINTS)


def _looks_local_life(content: str) -> bool:
    return _contains_any(text=content, markers=_LOCAL_LIFE_HINTS)


def _contains_any(*, text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


__all__ = [
    "LANGUAGE_CODES",
    "detect_input_language",
    "document_language_alignment",
    "language_alignment_score",
    "map_provider_language_param",
    "normalize_language_code",
    "route_search_language",
]
