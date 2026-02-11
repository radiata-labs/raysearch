from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from serpsage.text.normalize import normalize_text
from serpsage.text.similarity import is_duplicate_text
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens

if TYPE_CHECKING:
    from serpsage.settings.models import ProfileSettings


class EnrichScoringMixin:
    settings: Any
    _ranker: Any

    def _filter_blocks(
        self, blocks: list[str], *, profile: ProfileSettings, query: str
    ) -> tuple[list[str], dict[str, int]]:
        noise_words = list(profile.noise_words or [])
        title_patterns = self._compile_patterns(list(profile.title_tail_patterns or []))
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))

        stats: dict[str, int] = {
            "drop_noise_word": 0,
            "drop_template_hard": 0,
            "blocks_pruned_leading": 0,
            "blocks_deduped": 0,
        }

        kept: list[str] = []
        for block in blocks:
            if self._has_noise_word(block, noise_words):
                stats["drop_noise_word"] += 1
                continue
            tpl = self._template_score(
                block,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="block",
            )
            if tpl >= float(self.settings.enrich.select.block_hard_drop_threshold):
                stats["drop_template_hard"] += 1
                continue
            kept.append(block)
            if len(kept) >= int(self.settings.enrich.chunking.max_blocks):
                break

        deduped: list[str] = []
        th = max(0.92, float(profile.fuzzy_threshold))
        for b in kept:
            if is_duplicate_text(b, deduped, threshold=th):
                stats["blocks_deduped"] += 1
                continue
            deduped.append(b)
        kept = deduped

        scan_n = min(20, len(kept))
        start = 0
        for i in range(scan_n):
            b = kept[i]
            if len(b) < 120:
                continue
            tpl = self._template_score(
                b,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="block",
            )
            if tpl <= 0.55:
                start = i
                break
        if start > 0 and len(kept) > 1:
            stats["blocks_pruned_leading"] = int(start)
            kept = kept[start:]

        stats["blocks_kept"] = int(len(kept))
        return kept, stats

    def _score_chunks(
        self,
        *,
        chunks: list[str],
        query: str,
        query_tokens: list[str],
        profile: ProfileSettings,
    ) -> tuple[list[tuple[float, str]], dict[str, int | float]]:
        title_patterns = self._compile_patterns(list(profile.title_tail_patterns or []))
        query_has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", query))
        intent_tokens = extract_intent_tokens(query, list(profile.intent_terms or []))

        stats: dict[str, int | float] = {
            "drop_template_hard": 0,
            "drop_noise_word": 0,
            "drop_digits": 0,
            "drop_lang_mismatch": 0,
        }

        filtered: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks):
            dropped, reason = self._hard_drop_chunk(
                chunk,
                profile=profile,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
            )
            if dropped:
                if reason:
                    stats[reason] = int(stats.get(reason, 0)) + 1
                continue
            filtered.append((idx, chunk))
        if not filtered:
            stats["chunks_filtered"] = int(len(chunks))
            stats["chunks_candidates"] = 0
            return [], stats

        raw_scores = self._ranker.score_texts(
            texts=[c for _, c in filtered],
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        norm = self._ranker.normalize(scores=raw_scores)
        if norm and max(norm) <= 0.0 and max(raw_scores) > 0.0:
            norm = [0.5 for _ in norm]

        if raw_scores:
            stats["raw_min"] = float(min(raw_scores))
            stats["raw_max"] = float(max(raw_scores))
        if norm:
            stats["norm_min"] = float(min(norm))
            stats["norm_max"] = float(max(norm))

        scored, post_stats = self._post_process_chunk_scores(
            filtered,
            base_scores=norm,
            early_bonus=float(self.settings.enrich.select.early_bonus),
            dedupe_threshold=float(profile.fuzzy_threshold),
            min_score=float(self.settings.enrich.select.min_chunk_score),
            score_soft_gate_tau=float(self.settings.enrich.select.score_soft_gate_tau),
            template_penalty_weight=float(
                self.settings.enrich.select.template_penalty_weight
            ),
            query_has_cjk=query_has_cjk,
            title_patterns=title_patterns,
        )
        stats.update(post_stats)
        stats["chunks_candidates"] = int(len(filtered))
        stats["chunks_filtered"] = int(len(chunks) - len(filtered))
        return scored, stats

    def _compile_patterns(self, patterns: list[str]) -> list[re.Pattern[str]]:
        out: list[re.Pattern[str]] = []
        for p in patterns:
            if not p:
                continue
            try:
                out.append(re.compile(p, re.IGNORECASE))
            except re.error:
                continue
        return out

    def _has_noise_word(self, text: str, noise_words: list[str]) -> bool:
        lowered = normalize_text(text)
        for w in noise_words or []:
            wl = normalize_text(w)
            if wl and wl in lowered:
                return True
        return False

    def _chunk_stats(self, chunk: str) -> tuple[int, float, int, float, float]:
        tokens = tokenize(chunk)
        token_count = len(tokens)
        unique_ratio = len(set(tokens)) / token_count if tokens else 0.0
        digits = sum(ch.isdigit() for ch in chunk)
        digits_ratio = digits / max(1, len(chunk))
        cjk_count = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", chunk))
        cjk_ratio = cjk_count / max(1, len(chunk))
        return token_count, unique_ratio, digits, digits_ratio, cjk_ratio

    def _template_score(
        self,
        text: str,
        *,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
        mode: str,
    ) -> float:
        if not text:
            return 1.0
        lowered = normalize_text(text)
        if not lowered:
            return 1.0

        kw_strong = {
            "archive",
            "archives",
            "sitemap",
            "privacy",
            "terms",
            "cookie",
            "cookies",
        }
        kw_weak = {
            "next",
            "previous",
            "older",
            "newer",
            "page",
            "pages",
            "category",
            "categories",
            "tag",
            "tags",
        }
        kw_cjk_strong = {
            "欢迎",
            "参与完善",
            "编辑前请阅读",
            "条目",
            "模板",
            "著作权",
            "版权",
            "登录",
            "注册",
            "目录",
        }
        kw_cjk_weak = {
            "编辑",
            "规范",
            "帮助",
            "本站",
            "维基",
            "wiki",
        }

        kw_hits = 0.0
        for kw in kw_strong:
            if kw in lowered:
                kw_hits += 0.35
        for kw in kw_weak:
            if kw in lowered:
                kw_hits += 0.15
        for kw in kw_cjk_strong:
            if kw and kw in text:
                kw_hits += 0.30
        for kw in kw_cjk_weak:
            if kw and kw in text:
                kw_hits += 0.12
        kw_component = min(1.0, kw_hits)

        date_component = min(1.0, 0.15 * len(re.findall(r"\b(?:19|20)\d{2}\b", text)))

        token_count, unique_ratio, _, digits_ratio, cjk_ratio = self._chunk_stats(text)
        uniq_component = 0.0
        if token_count >= 8 and unique_ratio < 0.45:
            uniq_component = min(1.0, (0.45 - unique_ratio) / 0.45)

        digit_component = 0.0
        if digits_ratio > 0.18:
            digit_component = min(1.0, (digits_ratio - 0.18) / 0.32)

        sep_ratio = len(re.findall(r"[\|/>\u00bb]", text)) / max(1, len(text))
        sep_component = 0.0
        if mode == "block" and sep_ratio > 0.03:
            sep_component = min(1.0, (sep_ratio - 0.03) / 0.10)
        elif mode != "block" and sep_ratio > 0.05:
            sep_component = min(1.0, (sep_ratio - 0.05) / 0.10)

        link_hits = len(
            re.findall(r"(https?://|www\\.|\\b\\w+\\.(?:com|org|net|cn)\\b|@)", text)
        )
        link_component = 0.0
        if link_hits >= 2:
            link_component = min(1.0, 0.2 * float(link_hits))

        lang_component = 0.0
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.10:
            lang_component = 0.35

        title_component = 0.0
        for pat in title_patterns:
            if pat.search(text):
                title_component = 0.6
                break

        tpl = 0.0
        for comp in (
            kw_component,
            date_component,
            digit_component,
            sep_component,
            link_component,
            uniq_component,
            lang_component,
            title_component,
        ):
            comp = max(0.0, min(1.0, float(comp)))
            tpl = 1.0 - (1.0 - tpl) * (1.0 - comp)

        if mode == "block" and len(lowered) <= 64 and kw_component >= 0.35:
            tpl = max(tpl, 0.9)

        return float(max(0.0, min(1.0, tpl)))

    def _hard_drop_chunk(
        self,
        chunk: str,
        *,
        profile: ProfileSettings,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
    ) -> tuple[bool, str | None]:
        if not chunk:
            return True, "drop_empty"
        normalized = normalize_text(chunk)
        if not normalized:
            return True, "drop_empty"
        if self._has_noise_word(chunk, list(profile.noise_words or [])):
            return True, "drop_noise_word"

        _, _, digits, digits_ratio, cjk_ratio = self._chunk_stats(chunk)
        if digits >= 18 and digits_ratio > 0.2:
            return True, "drop_digits"
        if query_has_cjk and cjk_ratio > 0 and cjk_ratio < 0.06:
            return True, "drop_lang_mismatch"

        tpl = self._template_score(
            chunk,
            query_has_cjk=query_has_cjk,
            title_patterns=title_patterns,
            mode="chunk",
        )
        hard_th = float(self.settings.enrich.select.template_hard_drop_threshold)
        if tpl < hard_th:
            return False, None

        sep_ratio = len(re.findall(r"[\|/>\u00bb]", chunk)) / max(1, len(chunk))
        if tpl >= 0.98 and (len(normalized) < 420 or sep_ratio > 0.08):
            return True, "drop_template_hard"
        return False, None

    def _post_process_chunk_scores(
        self,
        filtered_chunks: list[tuple[int, str]],
        *,
        base_scores: list[float],
        early_bonus: float,
        dedupe_threshold: float,
        min_score: float,
        score_soft_gate_tau: float,
        template_penalty_weight: float,
        query_has_cjk: bool,
        title_patterns: list[re.Pattern[str]],
    ) -> tuple[list[tuple[float, str]], dict[str, int]]:
        stats: dict[str, int] = {"drop_min_score": 0, "drop_duplicate": 0}

        eps = 1e-6

        def sigmoid(x: float) -> float:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        def logit(p: float) -> float:
            p = max(eps, min(1.0 - eps, float(p)))
            return math.log(p / (1.0 - p))

        scored: list[tuple[float, str]] = []
        for (pos, chunk), base_score in zip(filtered_chunks, base_scores, strict=False):
            p0 = max(0.0, min(1.0, float(base_score)))
            lg = logit(p0)

            if early_bonus > 1.0:
                lg -= float(pos) * math.log(float(early_bonus))

            tpl = self._template_score(
                chunk,
                query_has_cjk=query_has_cjk,
                title_patterns=title_patterns,
                mode="chunk",
            )
            lg -= float(template_penalty_weight) * float(tpl)

            final = float(sigmoid(lg))

            tau = float(score_soft_gate_tau)
            if tau > 0:
                final *= float(sigmoid((final - float(min_score)) / tau))

            scored.append((float(final), chunk))

        scored.sort(key=lambda t: t[0], reverse=True)

        kept: list[tuple[float, str]] = []
        th = float(dedupe_threshold)
        for score, chunk in scored:
            if float(score) <= 0.0:
                continue
            if float(score) < float(min_score) and float(score_soft_gate_tau) <= 0:
                stats["drop_min_score"] += 1
                continue
            if is_duplicate_text(chunk, [c for _, c in kept], threshold=th):
                stats["drop_duplicate"] += 1
                continue
            kept.append((score, chunk))
        return kept, stats


__all__ = ["EnrichScoringMixin"]
