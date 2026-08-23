"""Contract tests for the descriptive public namespaces."""

import pypanelsim
from pypanelsim import core, designs, primitives


def test_core_namespace_exposes_simulation_contract() -> None:
    assert core.PanelDataset is pypanelsim.PanelDataset
    assert core.PanelDimensions is pypanelsim.PanelDimensions
    assert core.PanelSimulator is pypanelsim.PanelSimulator
    assert core.SimulationSeeds is pypanelsim.SimulationSeeds


def test_primitives_namespace_exposes_probability_laws() -> None:
    assert primitives.TwoWayFixedEffectsOutcome is pypanelsim.TwoWayFixedEffectsOutcome
    assert primitives.LowRankFactorOutcome is pypanelsim.LowRankFactorOutcome
    assert primitives.RandomizedSingleCohortAssignment is (
        pypanelsim.RandomizedSingleCohortAssignment
    )


def test_designs_namespace_covers_all_public_families() -> None:
    canonical = designs.classic_factor(seed=42)
    baker = designs.baker(seed=42)
    lepskii = designs.lepskii(
        "complete_homogeneity",
        config=designs.LepskiiPanelConfig(n_units=120),
        seed=42,
    )

    assert canonical.name == "classic_factor"
    assert baker.name == "baker_event_study"
    assert lepskii.name == "lepskii_complete_homogeneity"
