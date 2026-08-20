import pytest

from pypanelsim import DGPRegistry, classic_factor_design


def test_registry_creates_named_simulators() -> None:
    registry = DGPRegistry()
    registry.register("factor", classic_factor_design)

    simulator = registry.create("factor", overlap=1.0)
    assert simulator.name == "classic_factor"
    assert registry.names() == ("factor",)
    assert "factor" in registry
    assert len(registry) == 1


def test_registry_rejects_invalid_duplicate_and_unknown_names() -> None:
    registry = DGPRegistry()
    registry.register("factor", classic_factor_design)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("factor", classic_factor_design)
    with pytest.raises(ValueError, match="lowercase"):
        registry.register("Bad-Name", classic_factor_design)
    with pytest.raises(KeyError, match="available designs: factor"):
        registry.create("missing")


def test_registry_allows_explicit_replacement() -> None:
    registry = DGPRegistry()
    registry.register("factor", classic_factor_design)
    registry.register("factor", classic_factor_design, replace=True)
    assert registry.create("factor").name == "classic_factor"
