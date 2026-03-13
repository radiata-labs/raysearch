from __future__ import annotations

from serpsage.settings.models import AppSettings
from serpsage.utils import clean_whitespace


def resolve_research_model(
    *,
    settings: AppSettings,
    stage: str,
    fallback: str,
) -> str:
    model_settings = settings.research.models
    stage_to_model = {
        "plan": model_settings.plan,
        "link_select": model_settings.link_select,
        "overview": model_settings.abstract_analyze,
        "abstract": model_settings.abstract_analyze,
        "content": model_settings.content_analyze,
        "synthesize": model_settings.synthesize,
        "markdown": model_settings.markdown,
    }
    configured_model = clean_whitespace(stage_to_model.get(stage, ""))
    return configured_model or fallback


__all__ = ["resolve_research_model"]
