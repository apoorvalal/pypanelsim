import numpy as np

from pypanelsim import (
    ARMAErrorOutcome,
    ClusteredTrendOutcome,
    CorrelatedGaussianFeatures,
    EmpiricalPanelOutcome,
    GaussianUnitFeatures,
    LowRankFactorOutcome,
    PanelDimensions,
    PanelSimulator,
    RandomARMAErrorOutcome,
    RandomizedSingleCohortAssignment,
    SimulationContext,
    SimulationSeeds,
    TwoWayFixedEffectsOutcome,
    UnitPositionOutcome,
    UnitTrendOutcome,
    att_dml_nonlinear_basis,
)
from pypanelsim.components import CallableOutcomeModel, ConstantEffect


def test_empirical_panel_outcome_preserves_baseline_and_adds_iid_noise() -> None:
    baseline = np.arange(24, dtype=float).reshape(4, 6)
    model = EmpiricalPanelOutcome(
        baseline,
        noise_scale=0.2,
        source="illustrative panel",
    )
    baseline[:] = -99.0
    context = SimulationContext(PanelDimensions(4, 6), np.zeros((4, 6)))

    draw = model.generate(context, np.random.default_rng(42))
    expected_errors = np.random.default_rng(42).normal(scale=0.2, size=(4, 6))

    np.testing.assert_allclose(draw.values, model.baseline + expected_errors)
    np.testing.assert_array_equal(draw.metadata["errors"], expected_errors)
    assert draw.metadata["noise_model"] == "iid_gaussian"
    assert draw.metadata["source"] == "illustrative panel"
    assert model.baseline.flags.writeable is False


def test_empirical_panel_outcome_supports_correlated_time_noise() -> None:
    baseline = np.zeros((3, 4))
    covariance = 0.25 * 0.6 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4)))
    model = EmpiricalPanelOutcome(baseline, noise_covariance=covariance)
    context = SimulationContext(PanelDimensions(3, 4), np.zeros((3, 4)))

    draw = model.generate(context, np.random.default_rng(17))
    expected = np.random.default_rng(17).multivariate_normal(
        np.zeros(4),
        covariance,
        size=3,
        check_valid="raise",
        tol=1e-10,
    )

    np.testing.assert_allclose(draw.values, expected)
    assert draw.metadata["noise_model"] == "correlated_gaussian"
    np.testing.assert_array_equal(draw.metadata["noise_covariance"], covariance)


def test_empirical_panel_outcome_validates_inputs_and_dimensions() -> None:
    with np.testing.assert_raises_regex(ValueError, "two-dimensional"):
        EmpiricalPanelOutcome(np.ones(4))
    with np.testing.assert_raises_regex(ValueError, "not both"):
        EmpiricalPanelOutcome(
            np.ones((2, 3)), noise_scale=0.1, noise_covariance=np.eye(3)
        )
    with np.testing.assert_raises_regex(ValueError, "shape"):
        EmpiricalPanelOutcome(np.ones((2, 3)), noise_covariance=np.eye(2))
    with np.testing.assert_raises_regex(ValueError, "symmetric"):
        EmpiricalPanelOutcome(
            np.ones((2, 2)),
            noise_covariance=np.array([[1.0, 0.5], [0.0, 1.0]]),
        )
    with np.testing.assert_raises_regex(ValueError, "positive semidefinite"):
        EmpiricalPanelOutcome(
            np.ones((2, 2)),
            noise_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )

    model = EmpiricalPanelOutcome(np.ones((2, 3)))
    context = SimulationContext(PanelDimensions(3, 3), np.zeros((3, 3)))
    with np.testing.assert_raises_regex(ValueError, "baseline must have shape"):
        model.generate(context, np.random.default_rng(1))


def test_empirical_panel_outcome_composes_with_assignment_and_known_effects() -> None:
    baseline = np.arange(40, dtype=float).reshape(5, 8) / 10.0
    simulator = PanelSimulator(
        name="semi_synthetic",
        dimensions=PanelDimensions(*baseline.shape),
        assignment=RandomizedSingleCohortAssignment(1, 5),
        outcome_model=EmpiricalPanelOutcome(baseline, noise_scale=0.1),
        effect_model=ConstantEffect(0.75),
    )

    panel = simulator.simulate(streams=SimulationSeeds.from_seed(29))

    np.testing.assert_allclose(
        panel.untreated_outcome,
        panel.metadata["outcome"]["baseline"] + panel.metadata["outcome"]["errors"],
    )
    np.testing.assert_allclose(
        panel.outcome,
        panel.untreated_outcome + panel.treatment_effect,
    )
    assert panel.true_att == 0.75
    assert panel.metadata["outcome"]["model"] == "EmpiricalPanelOutcome"


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
        RandomARMAErrorOutcome(2, 2, burn_in=20),
        UnitTrendOutcome(0.5),
        UnitPositionOutcome(0.0, 2.0, 0.5),
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
