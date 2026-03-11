from __future__ import annotations

import inspect
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

from pydantic import ConfigDict

from serpsage.core.workunit import WorkUnit
from serpsage.settings.models import SettingModel

ConfigT = TypeVar("ConfigT", bound="ComponentConfigBase")
WorkUnitT = TypeVar("WorkUnitT", bound=WorkUnit)


class ComponentConfigBase(SettingModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False

    __setting_family__: str = ""
    __setting_name__: str = ""

    @classmethod
    def inject_env(
        cls,
        raw: dict[str, Any],
        *,
        env: dict[str, str],
    ) -> dict[str, Any]:
        _ = env
        return dict(raw)

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any] | None = None,
        *,
        env: dict[str, str],
    ) -> ComponentConfigBase:
        payload = cls.inject_env(dict(raw or {}), env=env)
        return cls.model_validate(payload)


class ComponentBase(WorkUnit, Generic[ConfigT]):
    Config: type[ConfigT]

    def __init_subclass__(
        cls,
        config: type[ConfigT] | None = None,
        **_kwargs: Any,
    ) -> None:
        super().__init_subclass__()

        orig_bases: tuple[type, ...] = getattr(cls, "__orig_bases__", ())
        for orig_base in orig_bases:
            origin_class = get_origin(orig_base)
            if not (
                inspect.isclass(origin_class)
                and issubclass(origin_class, ComponentBase)
            ):
                continue
            try:
                config_t = cast("tuple[ConfigT]", get_args(orig_base))[0]
            except ValueError:  # pragma: no cover
                continue
            if (
                config is None
                and inspect.isclass(config_t)
                and issubclass(config_t, ComponentConfigBase)
            ):
                config = config_t
        if config is not None:
            family_name = str(getattr(config, "__setting_family__", "")).strip()
            setting_name = str(getattr(config, "__setting_name__", "")).strip()
            if not family_name or not setting_name:
                raise TypeError(
                    f"{cls.__name__} config `{config.__name__}` must declare "
                    "non-empty `__setting_family__` and `__setting_name__`"
                )
            cls.Config = config

    @property
    def config(self) -> ConfigT:
        default: Any = None
        config_class = getattr(self, "Config", None)
        if inspect.isclass(config_class) and issubclass(
            config_class, ComponentConfigBase
        ):
            value = getattr(
                getattr(
                    self.rt.settings.components, config_class.__setting_family__, None
                ),
                config_class.__setting_name__,
                default,
            )
            return cast("ConfigT", value)
        return cast("ConfigT", default)


__all__ = [
    "ComponentBase",
    "ComponentConfigBase",
    "ConfigT",
]
