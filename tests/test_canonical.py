import numpy as np
import pytest

import pypanelsim as pps

SMALL = pps.CanonicalPanelConfig(n_control=12, n_treated=4, n_pre=8, n_post=3)


@pytest.mark.parametrize(
    "generator, kwargs",
    [
        (pps.classic_factor, {"overlap": 1.0}),
        (pps.weak_factor, {"overlap": 1.0}),
        (pps.synthetic_control, {"active_share": 0.25}),
        (pps.factor_synthetic, {"overlap": 1.0}),
        (pps.time_series, {"coefficient": 0.9, "integrated": False}),
        (pps.time_series, {"coefficient": 0.2, "integrated": True}),
        (pps.mixed_factor, {"overlap": 1.0}),
    ],
)
def test_canonical_panel_contract(generator, kwargs) -> None:
    panel = generator(config=SMALL, seed=101, **kwargs)

    assert panel.shape == (16, 11)
    np.testing.assert_allclose(
        panel.outcome, panel.untreated_outcome + panel.treatment_effect
    )
    np.testing.assert_array_equal(panel.treatment[:, : SMALL.n_pre], 0.0)
    np.testing.assert_array_equal(panel.treatment[: SMALL.n_control], 0.0)
    np.testing.assert_array_equal(
        panel.treatment[SMALL.n_control :, SMALL.n_pre :], 1.0
    )
    assert panel.true_att == pytest.approx(0.4)
    assert not panel.outcome.flags.writeable


@pytest.mark.parametrize(
    "generator, kwargs",
    [
        (pps.classic_factor, {}),
        (pps.weak_factor, {}),
        (pps.synthetic_control, {}),
        (pps.factor_synthetic, {}),
        (pps.time_series, {}),
        (pps.mixed_factor, {}),
    ],
)
def test_canonical_seed_reproduces_complete_draw(generator, kwargs) -> None:
    first = generator(config=SMALL, seed=202, **kwargs)
    second = generator(config=SMALL, seed=202, **kwargs)
    third = generator(config=SMALL, seed=203, **kwargs)

    np.testing.assert_array_equal(first.outcome, second.outcome)
    assert not np.array_equal(first.outcome, third.outcome)


def test_registry_exposes_all_canonical_designs() -> None:
    assert pps.available_canonical_designs() == (
        "classic_factor",
        "factor_synthetic",
        "mixed_factor",
        "synthetic_control",
        "time_series",
        "weak_factor",
    )
    panel = pps.make_canonical("classic_factor", overlap=1.0, config=SMALL).simulate(
        seed=1
    )
    assert panel.name == "classic_factor"


def test_sparse_donor_weights_are_aligned_with_panel_units() -> None:
    panel = pps.synthetic_control(active_share=0.25, config=SMALL, seed=303)
    metadata = panel.metadata["outcome"]
    weights = metadata["donor_weights"]
    control_weights = metadata["control_donor_weights"]

    assert weights.shape == (SMALL.n_units,)
    np.testing.assert_array_equal(weights[SMALL.n_control :], 0.0)
    assert np.count_nonzero(control_weights) == 3
    np.testing.assert_allclose(control_weights.sum(), 1.0)
    np.testing.assert_allclose(
        control_weights[control_weights > 0.0], np.full(3, 1.0 / 3.0)
    )


def test_integrated_process_accumulates_stationary_increments() -> None:
    stationary = pps.time_series(
        coefficient=0.2, integrated=False, config=SMALL, seed=404
    )
    integrated = pps.time_series(
        coefficient=0.2, integrated=True, config=SMALL, seed=404
    )

    np.testing.assert_allclose(
        np.diff(integrated.metadata["outcome"]["process"], axis=1),
        stationary.metadata["outcome"]["process"][:, 1:],
    )


def test_weak_factors_use_r_sample_standard_deviation() -> None:
    panel = pps.weak_factor(config=SMALL, seed=405)
    factors = panel.metadata["outcome"]["factors"]
    np.testing.assert_allclose(
        factors.std(axis=1, ddof=1), np.full(10, 0.2), atol=1e-12
    )


def test_stationary_treated_mean_uses_innovation_mean() -> None:
    config = pps.CanonicalPanelConfig(
        n_control=200, n_treated=200, n_pre=190, n_post=10
    )
    coefficient = 0.5
    panel = pps.time_series(coefficient=coefficient, config=config, seed=406)
    process = panel.metadata["outcome"]["process"]
    treated_process = process[config.n_control :, 50:]
    expected = 0.25 / (1.0 - coefficient)
    assert abs(treated_process.mean() - expected) < 0.04


def test_mixed_factor_has_two_canonical_half_panels() -> None:
    panel = pps.mixed_factor(config=SMALL, seed=407)
    metadata = panel.metadata["outcome"]
    assert metadata["first_group_controls"].size == 4
    assert metadata["second_group_controls"].size == 8
    assert metadata["first_group_factors"].shape == (2, SMALL.n_periods)
    assert metadata["second_group_factors"].shape == (2, SMALL.n_periods)


def test_primary_config_and_compatibility_aliases_are_explicit() -> None:
    assert pps.PanelConfig is pps.CanonicalPanelConfig
    assert pps.PanelData is pps.PanelDataset


def test_invalid_canonical_arguments_fail_with_actionable_messages() -> None:
    with pytest.raises(ValueError, match="active_share"):
        pps.synthetic_control(active_share=0.0, config=SMALL)
    with pytest.raises(ValueError, match="coefficient"):
        pps.time_series(coefficient=1.0, config=SMALL)
    with pytest.raises(ValueError, match="noise_variance"):
        pps.ClassicFactorOutcome(noise_variance=-1.0)
    with pytest.raises(ValueError, match="even number"):
        pps.mixed_factor(
            config=pps.CanonicalPanelConfig(
                n_control=10, n_treated=3, n_pre=8, n_post=3
            )
        )
