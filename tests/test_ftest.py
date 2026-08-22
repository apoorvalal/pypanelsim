import numpy as np
import pytest

from pypanelsim import (
    FTestCohortConfig,
    FTestTemporalConfig,
    available_ftest_cohort_designs,
    available_ftest_temporal_designs,
    ftest_cohort,
    ftest_cohort_profiles,
    ftest_temporal,
    ftest_temporal_profile,
)

TEMPORAL_NAMES = (
    "constant",
    "linear",
    "concave",
    "positive_then_negative",
    "exponential",
    "sinusoidal",
    "random_walk",
)

COHORT_NAMES = (
    "homogeneous",
    "log_vs_linear_vs_sin",
    "small_differences",
    "large_differences",
    "selection_on_gains",
    "novelty_effects",
    "activity_bias",
)


def test_ftest_catalogs_match_both_paper_suites() -> None:
    assert available_ftest_temporal_designs() == TEMPORAL_NAMES
    assert available_ftest_cohort_designs() == COHORT_NAMES


@pytest.mark.parametrize("name", TEMPORAL_NAMES)
def test_temporal_profiles_match_paper_formulas(name: str) -> None:
    config = FTestTemporalConfig(
        n_units=20,
        n_periods=9,
        adoption_period=3,
        max_effect=2.0,
    )
    length = 6
    profile = ftest_temporal_profile(name, config=config, seed=123)

    expected = {
        "constant": np.full(length, 2.0),
        "linear": np.linspace(0.0, 2.0, length),
        "concave": 2.0 * 0.5 * np.log(2.0 * np.arange(1, length + 1) / length + 1.0),
        "positive_then_negative": np.concatenate(
            (
                np.linspace(0.0, 2.0, length // 2),
                np.linspace(2.0, -2.0, length - length // 2),
            )
        ),
        "exponential": 2.0 * (1.0 - np.exp(-np.linspace(0.0, 5.0, length))),
        "sinusoidal": 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, length)),
        "random_walk": 2.0 * np.cumsum(np.random.default_rng(123).normal(size=length)),
    }[name]
    np.testing.assert_allclose(profile, expected)
    assert not profile.flags.writeable


def test_three_cohort_profiles_match_paper_formulas() -> None:
    config = FTestCohortConfig(n_units=80)
    homogeneous = ftest_cohort_profiles("homogeneous", config=config)
    np.testing.assert_allclose(homogeneous[10], np.log(np.arange(1, 21)))
    np.testing.assert_allclose(homogeneous[15], np.log(np.arange(1, 16)))
    np.testing.assert_allclose(homogeneous[20], np.log(np.arange(1, 11)))

    mixed = ftest_cohort_profiles("log_vs_linear_vs_sin", config=config)
    np.testing.assert_allclose(
        mixed[10], np.concatenate((np.linspace(2.0, 0.0, 10), np.zeros(10)))
    )
    np.testing.assert_allclose(mixed[15], np.log(2.0 * np.arange(1, 16)))
    np.testing.assert_allclose(mixed[20], np.sin(np.arange(1, 11)))

    small = ftest_cohort_profiles("small_differences", config=config)
    large = ftest_cohort_profiles("large_differences", config=config)
    selected = ftest_cohort_profiles("selection_on_gains", config=config)
    for index, period in enumerate(config.adoption_periods):
        np.testing.assert_allclose(
            small[period], homogeneous[period] * (1.0 + 0.1 * index)
        )
        np.testing.assert_allclose(large[period], homogeneous[period] * (period / 10.0))
        np.testing.assert_allclose(
            selected[period], homogeneous[period] * (1.0 - 0.1 * index)
        )

    novelty = ftest_cohort_profiles("novelty_effects", config=config)
    for period in config.adoption_periods:
        length = config.n_periods - period
        np.testing.assert_allclose(
            novelty[period], 2.0 * np.exp(-0.3 * np.arange(length)) + 0.5
        )

    activity = ftest_cohort_profiles("activity_bias", config=config)
    np.testing.assert_allclose(activity[10], 2.5)
    np.testing.assert_allclose(activity[15], homogeneous[15])
    np.testing.assert_allclose(activity[20], homogeneous[20])


@pytest.mark.parametrize("name", TEMPORAL_NAMES)
def test_every_temporal_design_simulates_its_declared_profile(name: str) -> None:
    config = FTestTemporalConfig(
        n_units=40,
        n_periods=12,
        adoption_period=5,
        n_treated=16,
    )
    panel = ftest_temporal(
        name,
        config=config,
        profile_seed=17,
        seed=23,
    )
    profile = ftest_temporal_profile(name, config=config, seed=17)

    assert panel.shape == (40, 12)
    assert panel.treated_units.size == 16
    np.testing.assert_array_equal(panel.treatment[:, :5], 0.0)
    np.testing.assert_allclose(
        panel.treatment_effect[panel.treated_units, 5:],
        np.tile(profile, (16, 1)),
    )
    np.testing.assert_array_equal(panel.treatment_effect[panel.control_units], 0.0)


@pytest.mark.parametrize("name", COHORT_NAMES)
def test_every_cohort_design_simulates_its_declared_profiles(name: str) -> None:
    config = FTestCohortConfig(n_units=80)
    panel = ftest_cohort(name, config=config, seed=29)
    profiles = ftest_cohort_profiles(name, config=config)

    assert panel.shape == (80, 30)
    adoption = panel.adoption_times
    for period, size in zip(config.adoption_periods, config.cohort_sizes, strict=True):
        units = np.flatnonzero(adoption == period)
        assert units.size == size
        np.testing.assert_allclose(
            panel.treatment_effect[units, period:],
            np.tile(profiles[period], (size, 1)),
        )
    assert panel.control_units.size == 40
    np.testing.assert_array_equal(panel.treatment_effect[panel.control_units], 0.0)
    np.testing.assert_allclose(
        panel.outcome,
        panel.untreated_outcome + panel.treatment_effect,
    )


def test_ftest_draws_are_reproducible_and_names_are_validated() -> None:
    config = FTestCohortConfig(n_units=80)
    first = ftest_cohort("novelty_effects", config=config, seed=31)
    second = ftest_cohort("novelty_effects", config=config, seed=31)
    np.testing.assert_array_equal(first.outcome, second.outcome)
    np.testing.assert_array_equal(first.treatment, second.treatment)

    with pytest.raises(KeyError, match="available"):
        ftest_temporal_profile("not_a_design")
    with pytest.raises(KeyError, match="available"):
        ftest_cohort_profiles("not_a_design")
