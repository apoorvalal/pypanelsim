import numpy as np

from pypanelsim import (
    ARMAErrorOutcome,
    ClusteredTrendOutcome,
    CorrelatedGaussianFeatures,
    GaussianUnitFeatures,
    LowRankFactorOutcome,
    PanelDimensions,
    PanelSimulator,
    RandomizedSingleCohortAssignment,
    SimulationContext,
    SimulationSeeds,
    TwoWayFixedEffectsOutcome,
    UnitTrendOutcome,
    att_dml_nonlinear_basis,
)
from pypanelsim.components import CallableOutcomeModel, ConstantEffect


def test_correlated_features_keep_raw_estimator_covariates() -> None:
    dimensions = PanelDimensions(4000, 2)
    draw = CorrelatedGaussianFeatures(
        n_features=6,
        correlation=0.5,
        transform=att_dml_nonlinear_basis,
        estimator_uses_raw=True,
    ).generate(dimensions, np.random.default_rng(4))

    assert draw.observables.shape == (4000, 9)
    assert draw.estimator_observables.shape == (4000, 6)
    correlation = np.corrcoef(draw.estimator_observables[:, :2], rowvar=False)[0, 1]
    assert abs(correlation - 0.5) < 0.05


def test_two_way_fixed_effects_and_rank_k_factor_expose_components() -> None:
    dimensions = PanelDimensions(30, 20)
    context = SimulationContext(dimensions, np.zeros((30, 20)))
    twfe = TwoWayFixedEffectsOutcome(1.0, 0.5, 0.0).generate(
        context, np.random.default_rng(5)
    )
    factor = LowRankFactorOutcome(rank=3, noise_scale=0.0).generate(
        context, np.random.default_rng(6)
    )

    assert twfe.metadata["kind"] == "two_way_fixed_effects"
    assert np.linalg.matrix_rank(twfe.values) <= 2
    assert factor.metadata["loadings"].shape == (30, 3)
    assert factor.metadata["factors"].shape == (20, 3)
    assert np.linalg.matrix_rank(factor.values) == 3


def test_arma_unit_trend_and_cluster_trend_have_panel_shape() -> None:
    dimensions = PanelDimensions(20, 30)
    context = SimulationContext(dimensions, np.zeros((20, 30)))
    models = (
        ARMAErrorOutcome((0.4,), (0.2,), burn_in=20),
        UnitTrendOutcome(0.5),
        ClusteredTrendOutcome(4, 0.5),
    )
    for model in models:
        draw = model.generate(context, np.random.default_rng(7))
        assert draw.values.shape == (20, 30)
        assert np.all(np.isfinite(draw.values))


def test_named_streams_isolate_assignment_from_unrelated_feature_draws() -> None:
    dimensions = PanelDimensions(50, 8)
    common = {
        "dimensions": dimensions,
        "assignment": RandomizedSingleCohortAssignment(15, 5),
        "outcome_model": CallableOutcomeModel(
            lambda context, rng: rng.normal(size=(50, 8))
        ),
        "effect_model": ConstantEffect(),
    }
    without_features = PanelSimulator(name="plain", **common)
    with_features = PanelSimulator(
        name="features", feature_model=GaussianUnitFeatures(3, 2), **common
    )

    first = without_features.simulate(streams=SimulationSeeds.from_seed(11))
    second = with_features.simulate(streams=SimulationSeeds.from_seed(11))
    np.testing.assert_array_equal(first.treatment, second.treatment)
    assert set(second.unit_annotations) >= {"ever_treated", "adoption_period"}
