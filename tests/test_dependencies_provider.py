from __future__ import annotations

import pytest

from serpsage.dependencies import (
    Inject,
    InjectToken,
    ServiceCollection,
    ServiceResolutionError,
)


class SampleService:
    def __init__(self) -> None:
        self.value = "sample"


class Consumer:
    def __init__(self, *, sample: SampleService = Inject()) -> None:
        self.sample = sample


class OptionalConsumer:
    def __init__(self, *, sample: SampleService | None = Inject()) -> None:
        self.sample = sample


class PluginBase:
    pass


class PluginA(PluginBase):
    pass


class PluginB(PluginBase):
    pass


class TupleCollector:
    def __init__(self, *, plugins: tuple[PluginBase, ...] = Inject()) -> None:
        self.plugins = plugins


class ListCollector:
    def __init__(self, *, plugins: list[PluginBase]) -> None:
        self.plugins = plugins


class AliasConsumer:
    def __init__(
        self, *, value: str = Inject(InjectToken[str]("sample.alias"))
    ) -> None:
        self.value = value


class CycleA:
    def __init__(self, *, item: CycleB = Inject()) -> None:
        self.item = item


class CycleB:
    def __init__(self, *, item: CycleA = Inject()) -> None:
        self.item = item


def test_singleton_and_alias_resolution() -> None:
    alias_key = InjectToken[str]("sample.alias")
    services = ServiceCollection()
    services.bind_instance(alias_key, "ok")
    services.bind_class(SampleService, SampleService)
    services.bind_class(Consumer, Consumer)
    services.bind_class(AliasConsumer, AliasConsumer)

    provider = services.build_provider()

    first = provider.require(SampleService)
    second = provider.require(SampleService)
    consumer = provider.require(Consumer)
    alias_consumer = provider.require(AliasConsumer)

    assert first is second
    assert consumer.sample is first
    assert alias_consumer.value == "ok"


def test_optional_and_multibinding_resolution() -> None:
    services = ServiceCollection()
    services.bind_class(OptionalConsumer, OptionalConsumer)
    services.bind_many(PluginBase, PluginA, order=20)
    services.bind_many(PluginBase, PluginB, order=10)
    services.bind_class(TupleCollector, TupleCollector)
    services.bind_class(
        ListCollector,
        ListCollector,
        init_kwargs={"plugins": Inject(PluginBase)},
    )

    provider = services.build_provider()

    optional_consumer = provider.require(OptionalConsumer)
    tuple_collector = provider.require(TupleCollector)
    list_collector = provider.require(ListCollector)

    assert optional_consumer.sample is None
    assert [type(item) for item in tuple_collector.plugins] == [PluginB, PluginA]
    assert [type(item) for item in list_collector.plugins] == [PluginB, PluginA]


def test_cycle_validation_reports_path() -> None:
    services = ServiceCollection()
    services.bind_class(CycleA, CycleA)
    services.bind_class(CycleB, CycleB)

    with pytest.raises(ServiceResolutionError) as exc_info:
        services.build_provider()

    message = str(exc_info.value)
    assert "CycleA" in message
    assert "CycleB" in message
    assert "path:" in message
