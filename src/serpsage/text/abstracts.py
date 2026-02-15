from __future__ import annotations

import math
import re
from dataclasses import dataclass

from serpsage.text.normalize import clean_whitespace

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_INLINE_CODE_ONLY_RE = re.compile(r"^\s*`[^`]+`\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;.!])\s+")


@dataclass(slots=True, frozen=True)
class AbstractCandidate:
    text: str
    heading: str
    position: int


def extract_abstract_candidates(
    markdown: str,
    *,
    max_markdown_chars: int,
    max_abstracts: int,
    min_abstract_chars: int,
) -> list[AbstractCandidate]:
    text = (markdown or "").strip()
    if max_markdown_chars > 0 and len(text) > max_markdown_chars:
        text = text[:max_markdown_chars]
    if not text:
        return []

    candidates: list[AbstractCandidate] = []
    in_code = False
    current_heading = ""
    next_position = 0

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if _FENCE_RE.match(stripped):
            in_code = not in_code
            continue
        if in_code:
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            current_heading = clean_whitespace(heading_match.group(1) or "")
            continue

        if _INLINE_CODE_ONLY_RE.match(stripped):
            continue

        if _TABLE_ROW_RE.match(stripped):
            row = _parse_table_row(stripped)
            if row:
                if len(row) >= max(1, int(min_abstract_chars)):
                    candidates.append(
                        AbstractCandidate(
                            text=row,
                            heading=current_heading,
                            position=next_position,
                        )
                    )
                    next_position += 1
            if len(candidates) >= max(1, int(max_abstracts)):
                break
            continue

        normalized = _LIST_PREFIX_RE.sub("", stripped)
        normalized = clean_whitespace(normalized)
        if not normalized:
            continue

        for sentence in split_line_sentences(normalized):
            if len(sentence) < max(1, int(min_abstract_chars)):
                continue
            candidates.append(
                AbstractCandidate(
                    text=sentence,
                    heading=current_heading,
                    position=next_position,
                )
            )
            next_position += 1
            if len(candidates) >= max(1, int(max_abstracts)):
                break
        if len(candidates) >= max(1, int(max_abstracts)):
            break

    return candidates


def split_line_sentences(text: str) -> list[str]:
    normalized = clean_whitespace(text or "")
    if not normalized:
        return []
    parts = _SENTENCE_SPLIT_RE.split(normalized)
    out: list[str] = []
    for part in parts:
        sentence = clean_whitespace(part)
        if sentence:
            out.append(sentence)
    return out or [normalized]


def apply_title_logit_boost(
    *,
    abstract_score: float,
    title_score: float,
    alpha: float,
) -> float:
    eps = 1e-6
    sa = min(1.0 - eps, max(eps, float(abstract_score)))
    st = min(1.0 - eps, max(eps, float(title_score)))
    la = math.log(sa / (1.0 - sa))
    lt = max(0.0, math.log(st / (1.0 - st)))
    boosted = 1.0 / (1.0 + math.exp(-(la + float(alpha) * lt)))
    return min(1.0, max(sa, boosted))


def fit_abstract_budget(
    *,
    ranked: list[tuple[float, AbstractCandidate]],
    top_k_abstracts: int,
    max_chars: int | None,
) -> list[tuple[float, AbstractCandidate]]:
    out: list[tuple[float, AbstractCandidate]] = []
    total_chars = 0
    limit = max(1, int(top_k_abstracts))
    for score, candidate in ranked:
        if len(out) >= limit:
            break
        if max_chars is not None and max_chars > 0:
            extra_newline = 1 if out else 0
            next_total = total_chars + extra_newline + len(candidate.text)
            if next_total > int(max_chars):
                break
            total_chars = next_total
        out.append((score, candidate))
    return out


def _parse_table_row(line: str) -> str:
    body = line.strip()
    body = body.removeprefix("|")
    body = body.removesuffix("|")
    cells = [clean_whitespace(cell) for cell in body.split("|")]
    if not cells:
        return ""
    if all(_TABLE_SEP_CELL_RE.fullmatch(cell or "") for cell in cells if cell):
        return ""
    values = [cell for cell in cells if cell]
    return " | ".join(values)


__all__ = [
    "AbstractCandidate",
    "apply_title_logit_boost",
    "extract_abstract_candidates",
    "fit_abstract_budget",
    "split_line_sentences",
]
