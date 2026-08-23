import numpy as np
import pytest

from pypanelsim import PanelDataset


def make_staggered_panel() -> PanelDataset:
    treatment = np.zeros((5, 6))
    treatment[:2, 2:] = 1.0
    treatment[2:5, 4:] = 1.0
    surface = np.zeros_like(treatment)
    surface[:2, 2:] = np.array([1.0, 2.0, 3.0, 4.0])
    surface[2:5, 4:] = np.array([10.0, 20.0])
    effect = surface * treatment
    return PanelDataset(
        outcome=effect,
        treatment=treatment,
        untreated_outcome=np.zeros_like(treatment),
        treatment_effect=effect,
        effect_surface=surface,
        name="truth_fixture",
        unit_annotations={"group": np.array(["a", "a", "b", "b", "b"])},
        time_annotations={"calendar_year": np.arange(2000, 2006)},
    )


def test_effect_surface_and_annotations_are_explicit_opt_in_columns() -> None:
    panel = make_staggered_panel()

    assert "effect_surface" not in panel.as_long_dict()
    expanded = panel.as_long_dict(
        include_effect_surface=True, include_annotations=True
    )
    assert "effect_surface" in expanded
    np.testing.assert_array_equal(expanded["group"][:6], ["a"] * 6)
    np.testing.assert_array_equal(expanded["calendar_year"][:6], range(2000, 2006))
    assert not panel.unit_annotations["group"].flags.writeable


def test_dataset_rejects_effect_surface_that_disagrees_with_realized_effect() -> None:
    treatment = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="effect_surface times treatment"):
        PanelDataset(
            outcome=np.array([[0.0, 1.0]]),
            treatment=treatment,
            untreated_outcome=np.zeros((1, 2)),
            treatment_effect=np.array([[0.0, 1.0]]),
            effect_surface=np.array([[0.0, 2.0]]),
            name="invalid_surface",
        )


def test_truth_reports_cohort_cells_and_support_weighted_event_target() -> None:
    panel = make_staggered_panel()
    cells = panel.truth.cohort_event(event_times=(-1, 0, 1, 2, 3))
    target = panel.truth.event_study(
        event_times=(-1, 0, 1, 2, 3), weighting="cohort_size"
    )

    assert set(np.unique(cells["cohort"])) == {2, 4}
    np.testing.assert_array_equal(target["supported_cohorts"], [2, 2, 2, 1, 1])
    np.testing.assert_allclose(
        target["effect"],
        [0.0, (2 * 1.0 + 3 * 10.0) / 5, (2 * 2.0 + 3 * 20.0) / 5, 3.0, 4.0],
    )
    np.testing.assert_array_equal(target["target_unit_count"], [5, 5, 5, 2, 2])


def test_truth_returns_nan_when_requested_event_has_no_panel_support() -> None:
    target = make_staggered_panel().truth.event_study(event_times=(100,))

    assert np.isnan(target["effect"][0])
    assert target["supported_cohorts"][0] == 0
    assert target["target_unit_count"][0] == 0


def test_truth_rejects_nonabsorbing_or_never_treated_panels() -> None:
    for treatment, message in (
        (np.array([[0.0, 1.0, 0.0]]), "absorbing"),
        (np.zeros((2, 3)), "treated cohort"),
    ):
        panel = PanelDataset(
            outcome=np.zeros_like(treatment),
            treatment=treatment,
            untreated_outcome=np.zeros_like(treatment),
            treatment_effect=np.zeros_like(treatment),
            name="unsupported_truth",
        )
        with pytest.raises(ValueError, match=message):
            panel.truth.event_study()
