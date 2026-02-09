from __future__ import annotations

import importlib
import inspect
import pathlib
import pkgutil
from dataclasses import is_dataclass

import pytest

import serpsage
from serpsage.contracts.base import WorkUnit

INCLUDED_PREFIXES = (
    "serpsage.app.engine",
    "serpsage.pipeline.",  # steps excluded below
    "serpsage.domain.",
    "serpsage.provider.",
    "serpsage.fetch.",
    "serpsage.extract.",
    "serpsage.rank.",
    "serpsage.cache.",
    "serpsage.overview.openai_compat",
)

EXCLUDED_MODULES = {
    "serpsage.pipeline.steps",
}

NAME_PATTERN = (
    "Engine",
    "Step",
    "Provider",
    "Fetcher",
    "Extractor",
    "Ranker",
    "Cache",
    "Limiter",
    "LLM",
    "Builder",
    "Deduper",
    "Enricher",
    "Filterer",
    "Normalizer",
    "Reranker",
)


def _iter_serpsage_modules() -> list[str]:
    return [
        m.name
        for m in pkgutil.walk_packages(serpsage.__path__, prefix="serpsage.")
        if not m.ispkg
    ]


def _included(module_name: str) -> bool:
    if module_name in EXCLUDED_MODULES:
        return False
    return any(module_name == p or module_name.startswith(p) for p in INCLUDED_PREFIXES)


def _is_work_class(name: str) -> bool:
    return any(tok in name for tok in NAME_PATTERN)


@pytest.mark.parametrize("module_name", [m for m in _iter_serpsage_modules() if _included(m)])
def test_work_classes_inherit_workunit_and_require_rt(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    for name, obj in vars(mod).items():
        if not inspect.isclass(obj):
            continue
        if obj.__module__ != module_name:
            continue
        if is_dataclass(obj):
            continue
        if not _is_work_class(name):
            continue

        assert issubclass(obj, WorkUnit), f"{module_name}.{name} must inherit WorkUnit"

        sig = inspect.signature(obj.__init__)
        rt_param = sig.parameters.get("rt")
        assert rt_param is not None, f"{module_name}.{name}.__init__ must accept rt"
        assert (
            rt_param.kind == inspect.Parameter.KEYWORD_ONLY
        ), f"{module_name}.{name}.__init__ rt must be keyword-only"


def test_no_env_access_outside_settings_loader() -> None:
    # Lightweight text scan that enforces the convention.
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "serpsage"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.as_posix()
        if rel.endswith("serpsage/settings/load.py"):
            continue
        txt = path.read_text(encoding="utf-8")
        if "os.environ" in txt or "os.getenv" in txt:
            offenders.append(rel)
    assert offenders == [], f"env access found outside settings loader: {offenders}"
