from __future__ import annotations

from serpsage.text.normalize import normalize_text


def extract_intent_tokens(query: str, intent_terms: list[str]) -> list[str]:
    lowered = normalize_text(query)
    out: list[str] = []
    for term in intent_terms or []:
        t = normalize_text(term)
        if t and t in lowered:
            out.append(t)
    return out


__all__ = ["extract_intent_tokens"]
