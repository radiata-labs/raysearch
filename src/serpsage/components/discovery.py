from __future__ import annotations

import importlib
import pkgutil

_BUILTIN_FAMILIES = (
    "http",
    "provider",
    "fetch",
    "extract",
    "rank",
    "cache",
    "llm",
    "rate_limit",
    "telemetry",
)
_EXCLUDED_MODULES = {"__init__", "base", "utils"}


class BuiltinComponentDiscovery:
    @staticmethod
    def discover() -> None:
        for family in _BUILTIN_FAMILIES:
            BuiltinComponentDiscovery._discover_family(family)

    @staticmethod
    def _discover_family(family: str) -> None:
        package_name = f"serpsage.components.{family}"
        package = importlib.import_module(package_name)
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            return
        for module_info in pkgutil.iter_modules(package_path):
            if module_info.ispkg:
                continue
            module_name = str(module_info.name)
            if module_name.startswith("_") or module_name in _EXCLUDED_MODULES:
                continue
            importlib.import_module(f"{package_name}.{module_name}")


__all__ = ["BuiltinComponentDiscovery"]
