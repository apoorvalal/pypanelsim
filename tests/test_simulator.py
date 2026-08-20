import numpy as np
import pytest

from pypanelsim import (
    CallableOutcomeModel,
    ComponentDraw,
    ConstantEffect,
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
