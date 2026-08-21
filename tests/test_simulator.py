import numpy as np
import pytest

from pypanelsim import (
    BinaryLogitAssignment,
    CallableOutcomeModel,
    CallableUnitEffect,
    ComponentDraw,
    ConstantEffect,
    GaussianUnitFeatures,
    PanelDimensions,
    PanelSimulator,
    SingleCohortAssignment,
    resolve_rng,
)


def gaussian_outcome(context, rng):
    values = rng.normal(size=(context.dimensions.n_units, context.dimensions.n_periods))
    return ComponentDraw(values, {"distribution": "normal"})


def make_simulator() -> PanelSimulator:
    return PanelSimulator(
        name="gaussian",
        dimensions=PanelDimensions(8, 6),
        assignment=SingleCohortAssignment(n_treated=2, adoption_period=4),
        outcome_model=CallableOutcomeModel(gaussian_outcome),
        effect_model=ConstantEffect(1.5),
    )


def test_composed_simulator_is_seeded_and_estimator_neutral() -> None:
    simulator = make_simulator()
    first = simulator.simulate(seed=91)
    second = simulator.simulate(seed=91)
    third = simulator.simulate(seed=92)

    np.testing.assert_array_equal(first.outcome, second.outcome)
    assert not np.array_equal(first.outcome, third.outcome)
    assert first.metadata["assignment"]["model"] == "SingleCohortAssignment"
    assert first.metadata["outcome"]["distribution"] == "normal"
    assert first.true_att == 1.5

    class MeanDifferenceEstimator:
        def fit(self, outcome, treatment):
            self.value = float(outcome[treatment == 1].mean())
            return self

    estimator = MeanDifferenceEstimator().fit(*first.as_arrays())
    assert np.isfinite(estimator.value)


def test_iter_simulations_spawns_reproducible_independent_streams() -> None:
    simulator = make_simulator()
    first = list(simulator.iter_simulations(3, seed=101))
    second = list(simulator.iter_simulations(3, seed=101))

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.outcome, right.outcome)
    assert not np.array_equal(first[0].outcome, first[1].outcome)


def test_resolve_rng_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="seed or rng"):
        resolve_rng(seed=1, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match="Generator"):
        resolve_rng(rng="not a generator")


def test_simulator_validates_component_shape_and_effect_support() -> None:
    dimensions = PanelDimensions(4, 5)

    def wrong_shape(context, rng):
        del context, rng
        return np.zeros((2, 2))

    simulator = PanelSimulator(
        name="bad_shape",
        dimensions=dimensions,
        assignment=SingleCohortAssignment(1, 3),
        outcome_model=CallableOutcomeModel(wrong_shape),
        effect_model=ConstantEffect(),
    )
    with pytest.raises(ValueError, match="untreated outcome must have shape"):
        simulator.simulate(seed=1)


def test_feature_model_is_shared_by_assignment_outcome_and_dataset() -> None:
    dimensions = PanelDimensions(20, 6)

    def feature_outcome(context, rng):
        noise = rng.normal(scale=0.01, size=(dimensions.n_units, dimensions.n_periods))
        signal = context.observables[:, :1] + 2.0 * context.unobservables[:, :1]
        return ComponentDraw(signal + noise, {"shared_features": True})

    simulator = PanelSimulator(
        name="feature_selection",
        dimensions=dimensions,
        assignment=BinaryLogitAssignment(
            adoption_period=4,
            intercept=-0.5,
            observable_coefficients=(1.0,),
            unobservable_coefficients=(1.0,),
        ),
        outcome_model=CallableOutcomeModel(feature_outcome),
        effect_model=ConstantEffect(1.0),
        feature_model=GaussianUnitFeatures(n_observables=1, n_unobservables=1),
    )
    panel = simulator.simulate(seed=12)

    np.testing.assert_allclose(
        panel.unit_covariates,
        panel.metadata["assignment"]["linear_predictor"][:, None]
        + 0.5
        - panel.metadata["features"]["unobservables"],
    )
    assert panel.unit_covariate_names == ("x1",)
    assert panel.metadata["outcome"]["shared_features"] is True


def test_simulator_accepts_unit_effect_lambda() -> None:
    dimensions = PanelDimensions(8, 6)
    simulator = PanelSimulator(
        name="heterogeneous_effects",
        dimensions=dimensions,
        assignment=SingleCohortAssignment(n_treated=3, adoption_period=4),
        outcome_model=CallableOutcomeModel(gaussian_outcome),
        effect_model=lambda x: (
            1.0 + 0.5 * x.observables[:, 0] + 0.75 * x.unobservables[:, 0]
        ),
        feature_model=GaussianUnitFeatures(n_observables=1, n_unobservables=1),
    )
    panel = simulator.simulate(seed=14)

    assert isinstance(simulator.effect_model, CallableUnitEffect)
    expected = (
        1.0
        + 0.5 * panel.unit_covariates[:, 0]
        + 0.75 * panel.metadata["features"]["unobservables"][:, 0]
    )
    np.testing.assert_allclose(panel.metadata["effect"]["unit_effects"], expected)
    np.testing.assert_allclose(
        panel.treatment_effect,
        expected[:, None] * panel.treatment,
    )
