from __future__ import annotations

import numpy as np
import pytest

import pypanelsim as pps


def test_baker_default_design_matches_cohorts_and_dynamic_effects() -> None:
    config = pps.BakerPanelConfig()
    panel = pps.baker(config=config, seed=20260821)

    assert panel.shape == (1_000, 36)
    assert panel.unit_covariate_names == ("state",)
    np.testing.assert_array_equal(
        np.unique(panel.adoption_times, return_counts=True)[0],
        config.adoption_periods,
    )
    assignment = panel.metadata["assignment"]
    np.testing.assert_array_equal(
        np.unique(assignment["cohort_period_by_state"], return_counts=True)[1],
        config.cohort_state_counts,
    )

    for cohort, slope in zip(
        config.adoption_periods, config.cohort_effect_slopes, strict=True
    ):
        unit = np.flatnonzero(panel.adoption_times == cohort)[0]
        np.testing.assert_allclose(panel.treatment_effect[unit, :cohort], 0.0)
        np.testing.assert_allclose(
            panel.treatment_effect[unit, cohort : cohort + 3],
            np.asarray([1.0, 2.0, 3.0]) * slope,
        )


def test_baker_is_seed_reproducible_and_randomizes_state_cohorts() -> None:
    config = pps.BakerPanelConfig(n_units=100)
    first = pps.baker(config=config, seed=7)
    repeated = pps.baker(config=config, seed=7)
    other = pps.baker(config=config, seed=8)

    np.testing.assert_array_equal(first.outcome, repeated.outcome)
    np.testing.assert_array_equal(first.treatment, repeated.treatment)
    assert not np.array_equal(first.treatment, other.treatment)

    states = first.unit_covariates[:, 0].astype(int)
    for state in np.unique(states):
        state_units = np.flatnonzero(states == state)
        assert np.unique(first.adoption_times[state_units]).size == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_units": 0}, "n_units"),
        ({"adoption_years": (1998, 1989, 2007)}, "strictly increasing"),
        ({"cohort_state_counts": (17, 18)}, "equal length"),
        ({"cohort_effect_slopes": (0.1, np.nan, 0.01)}, "finite"),
        ({"noise_scale": -1.0}, "nonnegative"),
    ],
)
def test_baker_config_rejects_invalid_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pps.BakerPanelConfig(**kwargs)


def test_baker_design_accepts_custom_panel_and_effect_law() -> None:
    config = pps.BakerPanelConfig(
        n_units=15,
        start_year=2000,
        end_year=2008,
        adoption_years=(2002, 2005),
        cohort_state_counts=(2, 3),
        cohort_effect_slopes=(0.4, -0.2),
        effect_scale=2.0,
        noise_scale=0.0,
    )
    panel = pps.baker_design(config).simulate(seed=11)

    assert panel.shape == (15, 9)
    assert panel.is_absorbing
    for cohort, slope in zip(
        config.adoption_periods, config.cohort_effect_slopes, strict=True
    ):
        unit = np.flatnonzero(panel.adoption_times == cohort)[0]
        assert panel.treatment_effect[unit, cohort] == pytest.approx(
            slope * config.effect_scale
        )
