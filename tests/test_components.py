import numpy as np
import pytest

from pypanelsim import (
    AdditiveFactorOutcome,
    AssignmentContext,
    BinaryLogitAssignment,
    CallableEffect,
    CallableUnitEffect,
    CohortEventTimeEffect,
    ConstantEffect,
    GaussianTimeFeatures,
    GeneralizedPropensityAssignment,
    LinearRampEffect,
    PanelDimensions,
    RandomizedSingleCohortAssignment,
    RandomizedStaggeredAdoption,
    SimulationContext,
    SingleCohortAssignment,
    StaggeredAdoption,
    SumOutcomeModel,
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


def test_staggered_assignment_copies_its_configuration() -> None:
    adoption = {1: 2}
    assignment = StaggeredAdoption(adoption)
    adoption[2] = 3

    draw = assignment.assign(PanelDimensions(4, 5), np.random.default_rng(2))
    np.testing.assert_array_equal(np.flatnonzero(draw.values[:, -1]), [1])


def test_constant_effect_is_zero_outside_treatment() -> None:
    dimensions = PanelDimensions(3, 4)
    treatment = StaggeredAdoption({2: 1}).assign(dimensions, np.random.default_rng(4))
    context = SimulationContext(dimensions, treatment.values)
    effect = ConstantEffect(-2.0).generate(context, np.random.default_rng(5))

    np.testing.assert_array_equal(effect.values, -2.0 * treatment.values)


def test_callable_unit_effect_uses_observed_and_latent_features() -> None:
    dimensions = PanelDimensions(3, 4)
    treatment = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    context = SimulationContext(
        dimensions,
        treatment,
        observables=np.array([[-1.0], [0.5], [2.0]]),
        unobservables=np.array([[0.25], [-0.5], [1.0]]),
    )
    model = CallableUnitEffect(
        lambda x: 1.0 + 0.4 * x.observables[:, 0] + 0.8 * x.unobservables[:, 0]
    )
    draw = model.generate(context, np.random.default_rng(6))

    expected_unit_effects = np.array([0.8, 0.8, 2.6])
    np.testing.assert_allclose(
        draw.values,
        expected_unit_effects[:, None] * treatment,
    )
    np.testing.assert_allclose(draw.metadata["unit_effects"], expected_unit_effects)
    np.testing.assert_allclose(
        draw.metadata["effect_surface"],
        np.repeat(expected_unit_effects[:, None], dimensions.n_periods, axis=1),
    )


@pytest.mark.parametrize(
    "result, expected",
    [
        (2.0, np.array([2.0, 2.0, 2.0])),
        (np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_callable_unit_effect_normalizes_scalar_and_column(result, expected) -> None:
    dimensions = PanelDimensions(3, 2)
    context = SimulationContext(dimensions, np.ones((3, 2)))

    draw = CallableUnitEffect(lambda context: result).generate(
        context, np.random.default_rng(1)
    )

    np.testing.assert_allclose(draw.metadata["unit_effects"], expected)
    np.testing.assert_allclose(draw.values, np.repeat(expected[:, None], 2, axis=1))


@pytest.mark.parametrize(
    "function, message",
    [
        (lambda context: np.zeros((context.dimensions.n_units, 2)), "shape"),
        (lambda context: np.full(context.dimensions.n_units, np.nan), "finite"),
    ],
)
def test_callable_unit_effect_validates_output(function, message: str) -> None:
    dimensions = PanelDimensions(3, 4)
    context = SimulationContext(dimensions, np.zeros((3, 4)))

    with pytest.raises(ValueError, match=message):
        CallableUnitEffect(function).generate(context, np.random.default_rng(1))


def test_callable_effect_accepts_time_vector_and_unit_time_surface() -> None:
    dimensions = PanelDimensions(3, 4)
    treatment = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    context = SimulationContext(dimensions, treatment)
    time_effects = np.array([0.5, 1.0, 1.5, 2.0])

    time_draw = CallableEffect(lambda x: time_effects[None, :]).generate(
        context, np.random.default_rng(1)
    )
    np.testing.assert_allclose(time_draw.values, time_effects[None, :] * treatment)
    np.testing.assert_allclose(time_draw.metadata["time_effects"], time_effects)
    assert time_draw.metadata["normalization"] == "time"

    vector_draw = CallableEffect(lambda x: time_effects).generate(
        context, np.random.default_rng(1)
    )
    np.testing.assert_allclose(vector_draw.values, time_draw.values)

    surface = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    surface_draw = CallableEffect(lambda x: surface).generate(
        context, np.random.default_rng(1)
    )
    np.testing.assert_allclose(surface_draw.values, surface * treatment)
    np.testing.assert_allclose(surface_draw.metadata["effect_surface"], surface)
    assert surface_draw.metadata["normalization"] == "surface"


def test_gaussian_time_features_have_one_row_per_period() -> None:
    dimensions = PanelDimensions(5, 7)
    model = GaussianTimeFeatures(n_features=2)
    first = model.generate(dimensions, np.random.default_rng(9))
    second = model.generate(dimensions, np.random.default_rng(9))

    assert first.values.shape == (7, 2)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata["feature_names"] == ("v1", "v2")

    with pytest.raises(ValueError, match="nonnegative"):
        GaussianTimeFeatures(n_features=-1)


def test_additive_factor_outcome_exposes_exact_drawn_components() -> None:
    dimensions = PanelDimensions(4, 5)
    context = SimulationContext(dimensions, np.zeros((4, 5)))
    model = AdditiveFactorOutcome(
        unit_effect_scale=1.2,
        time_effect_scale=0.4,
        noise_scale=0.3,
    )
    first = model.generate(context, np.random.default_rng(11))
    second = model.generate(context, np.random.default_rng(11))

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_allclose(
        first.values,
        first.metadata["unit_effects"][:, None]
        + first.metadata["time_effects"][None, :]
        + first.metadata["errors"],
    )
    assert first.metadata["kind"] == "additive_factor"


@pytest.mark.parametrize(
    "parameter",
    ("unit_effect_scale", "time_effect_scale", "noise_scale"),
)
def test_additive_factor_outcome_rejects_invalid_scales(parameter: str) -> None:
    with pytest.raises(ValueError, match=parameter):
        AdditiveFactorOutcome(**{parameter: -0.1})


def test_sum_outcome_model_combines_and_records_components() -> None:
    dimensions = PanelDimensions(3, 4)
    context = SimulationContext(dimensions, np.zeros((3, 4)))
    first = AdditiveFactorOutcome(
        unit_effect_scale=1.0,
        time_effect_scale=0.0,
        noise_scale=0.0,
    )
    second = AdditiveFactorOutcome(
        unit_effect_scale=0.0,
        time_effect_scale=1.0,
        noise_scale=0.0,
    )
    combined = SumOutcomeModel((first, second), weights=(2.0, 0.5)).generate(
        context,
        np.random.default_rng(12),
    )
    components = combined.metadata["components"]

    np.testing.assert_allclose(
        combined.values,
        2.0
        * (
            components[0]["unit_effects"][:, None]
            + components[0]["time_effects"][None, :]
            + components[0]["errors"]
        )
        + 0.5
        * (
            components[1]["unit_effects"][:, None]
            + components[1]["time_effects"][None, :]
            + components[1]["errors"]
        ),
    )
    assert [component["model"] for component in components] == [
        "AdditiveFactorOutcome",
        "AdditiveFactorOutcome",
    ]


def test_sum_outcome_model_validates_models_and_weights() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SumOutcomeModel(())
    with pytest.raises(TypeError, match="OutcomeModel"):
        SumOutcomeModel((object(),))
    with pytest.raises(ValueError, match="equal length"):
        SumOutcomeModel((AdditiveFactorOutcome(),), weights=(1.0, 2.0))


def test_simulation_context_validates_time_feature_rows() -> None:
    dimensions = PanelDimensions(3, 4)
    with pytest.raises(ValueError, match="n_periods"):
        SimulationContext(
            dimensions,
            np.zeros((3, 4)),
            time_features=np.zeros((3, 1)),
        )


def test_randomized_single_cohort_samples_fixed_count_from_eligible_units() -> None:
    dimensions = PanelDimensions(8, 6)
    model = RandomizedSingleCohortAssignment(
        n_treated=3,
        adoption_period=4,
        eligible_units=(1, 2, 4, 6, 7),
    )
    first = model.assign(dimensions, np.random.default_rng(10))
    second = model.assign(dimensions, np.random.default_rng(10))

    np.testing.assert_array_equal(first.values, second.values)
    treated = np.flatnonzero(first.values[:, -1])
    assert treated.size == 3
    assert set(treated).issubset({1, 2, 4, 6, 7})
    np.testing.assert_allclose(
        first.metadata["propensity_scores"],
        [0.0, 0.6, 0.6, 0.0, 0.6, 0.0, 0.6, 0.6],
    )


def test_randomized_staggered_adoption_fixes_each_cohort_size() -> None:
    dimensions = PanelDimensions(20, 8)
    model = RandomizedStaggeredAdoption(
        adoption_periods=(2, 5),
        cohort_sizes=(4, 6),
        eligible_units=tuple(range(1, 19)),
    )
    first = model.assign(dimensions, np.random.default_rng(10))
    second = model.assign(dimensions, np.random.default_rng(10))

    np.testing.assert_array_equal(first.values, second.values)
    adoption = first.metadata["adoption_times"]
    assert np.sum(adoption == 2) == 4
    assert np.sum(adoption == 5) == 6
    assert np.sum(adoption == 8) == 10
    np.testing.assert_array_equal(first.values[:, :2], 0.0)
    np.testing.assert_allclose(
        first.metadata["generalized_propensity_scores"][1],
        [4 / 18, 6 / 18, 8 / 18],
    )
    np.testing.assert_allclose(
        first.metadata["generalized_propensity_scores"][[0, 19]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )


def test_cohort_event_time_effect_applies_supplied_profiles() -> None:
    dimensions = PanelDimensions(4, 5)
    treatment = StaggeredAdoption({0: 1, 2: 3}).assign(
        dimensions, np.random.default_rng(2)
    )
    context = SimulationContext(dimensions, treatment.values)
    model = CohortEventTimeEffect(
        {
            1: (0.5, 1.0, 1.5, 2.0),
            3: (-1.0, -2.0),
        }
    )
    draw = model.generate(context, np.random.default_rng(3))

    np.testing.assert_allclose(draw.values[0], [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(draw.values[2], [0.0, 0.0, 0.0, -1.0, -2.0])
    np.testing.assert_array_equal(draw.values[[1, 3]], 0.0)


def test_cohort_event_time_effect_rejects_missing_or_short_profiles() -> None:
    dimensions = PanelDimensions(2, 5)
    treatment = StaggeredAdoption({0: 1}).assign(dimensions, np.random.default_rng(2))
    context = SimulationContext(dimensions, treatment.values)

    with pytest.raises(ValueError, match="no event-time profile"):
        CohortEventTimeEffect({2: (1.0, 2.0, 3.0)}).generate(
            context, np.random.default_rng(3)
        )
    with pytest.raises(ValueError, match="at least 4 values"):
        CohortEventTimeEffect({1: (1.0, 2.0)}).generate(
            context, np.random.default_rng(3)
        )


def test_binary_logit_uses_observed_and_latent_features() -> None:
    dimensions = PanelDimensions(4, 5)
    context = AssignmentContext(
        dimensions,
        observables=np.array([[-1.0], [0.0], [1.0], [2.0]]),
        unobservables=np.array([[0.5], [-0.5], [1.5], [0.0]]),
    )
    model = BinaryLogitAssignment(
        adoption_period=3,
        intercept=0.2,
        observable_coefficients=(0.8,),
        unobservable_coefficients=(-0.4,),
    )
    draw = model.assign(context, np.random.default_rng(3))

    linear_predictor = 0.2 + 0.8 * context.observables[:, 0]
    linear_predictor -= 0.4 * context.unobservables[:, 0]
    expected = 1.0 / (1.0 + np.exp(-linear_predictor))
    np.testing.assert_allclose(draw.metadata["propensity_scores"], expected)
    np.testing.assert_array_equal(draw.values[:, :3], 0.0)
    np.testing.assert_array_equal(draw.values[:, 3], draw.values[:, 4])


def test_generalized_propensity_maps_units_to_adoption_cohorts() -> None:
    dimensions = PanelDimensions(2000, 8)
    context = AssignmentContext(
        dimensions,
        observables=np.zeros((dimensions.n_units, 1)),
        unobservables=np.zeros((dimensions.n_units, 0)),
    )
    model = GeneralizedPropensityAssignment(
        adoption_periods=(3, 5),
        intercepts=(np.log(2.0), np.log(3.0)),
        observable_coefficients=((0.0,), (0.0,)),
    )
    draw = model.assign(context, np.random.default_rng(4))

    scores = draw.metadata["generalized_propensity_scores"]
    np.testing.assert_allclose(scores, np.tile([2 / 6, 3 / 6, 1 / 6], (2000, 1)))
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)
    assert set(np.unique(draw.metadata["adoption_times"])).issubset({3, 5, 8})
    assert np.all(np.diff(draw.values, axis=1) >= 0.0)
    empirical = (
        np.bincount(draw.metadata["assigned_categories"], minlength=3)
        / dimensions.n_units
    )
    np.testing.assert_allclose(empirical, [2 / 6, 3 / 6, 1 / 6], atol=0.035)


def test_selection_assignments_validate_feature_dimensions() -> None:
    dimensions = PanelDimensions(5, 5)
    context = AssignmentContext(
        dimensions,
        observables=np.zeros((5, 2)),
        unobservables=np.zeros((5, 1)),
    )

    with pytest.raises(ValueError, match="observable_coefficients"):
        BinaryLogitAssignment(
            adoption_period=3,
            observable_coefficients=(1.0,),
            unobservable_coefficients=(1.0,),
        ).assign(context, np.random.default_rng(1))
    with pytest.raises(ValueError, match="unobservable_coefficients"):
        GeneralizedPropensityAssignment(
            adoption_periods=(2, 3),
            observable_coefficients=((1.0, 1.0), (1.0, 1.0)),
            unobservable_coefficients=((1.0,),),
        ).assign(context, np.random.default_rng(1))


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
