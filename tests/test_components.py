import numpy as np
import pytest

from pypanelsim import (
    ConstantEffect,
    LinearRampEffect,
    PanelDimensions,
    SimulationContext,
    SingleCohortAssignment,
    StaggeredAdoption,
)


def test_single_cohort_defaults_to_last_units() -> None:
    dimensions = PanelDimensions(6, 5)
    draw = SingleCohortAssignment(n_treated=2, adoption_period=3).assign(
        dimensions, np.random.default_rng(1)
    )

    np.testing.assert_array_equal(draw.values[:4], 0.0)
    np.testing.assert_array_equal(draw.values[4:, :3], 0.0)
    np.testing.assert_array_equal(draw.values[4:, 3:], 1.0)


def test_single_cohort_accepts_explicit_nonterminal_units() -> None:
    dimensions = PanelDimensions(5, 4)
    draw = SingleCohortAssignment(
        n_treated=2, adoption_period=2, treated_units=(1, 3)
    ).assign(dimensions, np.random.default_rng(1))

    np.testing.assert_array_equal(np.flatnonzero(draw.values[:, -1]), [1, 3])


def test_staggered_adoption_and_linear_effect_use_unit_event_time() -> None:
    dimensions = PanelDimensions(4, 6)
    assignment = StaggeredAdoption({1: 2, 3: 4}).assign(
        dimensions, np.random.default_rng(2)
    )
    context = SimulationContext(dimensions, assignment.values)
    effect = LinearRampEffect(0.5).generate(context, np.random.default_rng(3))

    np.testing.assert_array_equal(context.adoption_times, [6, 2, 6, 4])
    np.testing.assert_allclose(effect.values[1], [0, 0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(effect.values[3], [0, 0, 0, 0, 0.5, 1.0])


def test_constant_effect_is_zero_outside_treatment() -> None:
    dimensions = PanelDimensions(3, 4)
    treatment = StaggeredAdoption({2: 1}).assign(dimensions, np.random.default_rng(4))
    context = SimulationContext(dimensions, treatment.values)
    effect = ConstantEffect(-2.0).generate(context, np.random.default_rng(5))

    np.testing.assert_array_equal(effect.values, -2.0 * treatment.values)


@pytest.mark.parametrize(
    "assignment, message",
    [
        (SingleCohortAssignment(n_treated=0), "n_treated"),
        (SingleCohortAssignment(n_treated=1, adoption_period=5), "adoption_period"),
        (
            SingleCohortAssignment(
                n_treated=2, adoption_period=2, treated_units=(1, 1)
            ),
            "unique",
        ),
        (StaggeredAdoption({6: 1}), "invalid treated unit"),
    ],
)
def test_invalid_assignment_fails(assignment: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        assignment.assign(PanelDimensions(5, 5), np.random.default_rng(1))
