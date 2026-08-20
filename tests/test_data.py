from types import MappingProxyType

import numpy as np
import pytest

from pypanelsim import PanelDataset


def make_panel() -> PanelDataset:
    treatment = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
    )
    untreated = np.arange(12, dtype=float).reshape(3, 4)
    effect = treatment * 2.0
    return PanelDataset(
        outcome=untreated + effect,
        treatment=treatment,
        untreated_outcome=untreated,
        treatment_effect=effect,
        name="fixture",
        unit_ids=np.array(["a", "b", "c"]),
        time_ids=np.array([2000, 2001, 2002, 2003]),
        metadata={"array": np.arange(3), "nested": {"items": [1, 2]}},
    )


def test_dataset_owns_read_only_arrays_and_metadata() -> None:
    source = np.arange(12, dtype=float).reshape(3, 4)
    treatment = np.zeros_like(source)
    panel = PanelDataset(
        outcome=source,
        treatment=treatment,
        untreated_outcome=source,
        treatment_effect=treatment,
        name="immutable",
        metadata={"array": source},
    )
    source[0, 0] = 100.0

    assert panel.outcome[0, 0] == 0.0
    assert not panel.outcome.flags.writeable
    assert not panel.metadata["array"].flags.writeable
    assert isinstance(panel.metadata, MappingProxyType)


def test_dataset_exposes_panel_and_causal_properties() -> None:
    panel = make_panel()

    assert panel.shape == (3, 4)
    assert panel.n_units == 3
    assert panel.n_periods == 4
    np.testing.assert_array_equal(panel.control_units, [0])
    np.testing.assert_array_equal(panel.treated_units, [1, 2])
    np.testing.assert_array_equal(panel.adoption_times, [4, 2, 1])
    assert panel.is_absorbing
    assert panel.true_att == 2.0


def test_array_interchange_supports_safe_views_and_explicit_copies() -> None:
    panel = make_panel()

    y_view, w_view = panel.as_arrays()
    y_copy, w_copy = panel.as_arrays(copy=True)
    assert y_view is panel.outcome
    assert w_view is panel.treatment
    assert y_copy.flags.writeable
    assert w_copy.flags.writeable
    selected = panel.arrays(("treatment_effect", "untreated_outcome"))
    assert selected[0] is panel.treatment_effect
    assert selected[1] is panel.untreated_outcome

    with pytest.raises(ValueError, match="unknown panel fields"):
        panel.arrays(("residual",))


def test_long_interchange_has_one_row_per_panel_cell() -> None:
    panel = make_panel()
    long = panel.as_long_dict()

    assert set(long) == {
        "unit",
        "time",
        "outcome",
        "treatment",
        "untreated_outcome",
        "treatment_effect",
    }
    assert all(column.shape == (12,) for column in long.values())
    np.testing.assert_array_equal(long["unit"][:4], ["a"] * 4)
    np.testing.assert_array_equal(long["time"][:4], [2000, 2001, 2002, 2003])
    assert not long["outcome"].flags.writeable


@pytest.mark.parametrize(
    "change, message",
    [
        ({"outcome": np.zeros((2, 3, 1))}, "two-dimensional"),
        ({"treatment": np.zeros((3, 3))}, "same shape"),
        ({"treatment": np.full((3, 4), 0.5)}, "zero and one"),
        ({"treatment_effect": np.ones((3, 4))}, "zero outside treated"),
        ({"outcome": np.ones((3, 4))}, "must equal"),
    ],
)
def test_invalid_dataset_contract_fails(
    change: dict[str, np.ndarray], message: str
) -> None:
    base = {
        "outcome": np.zeros((3, 4)),
        "treatment": np.zeros((3, 4)),
        "untreated_outcome": np.zeros((3, 4)),
        "treatment_effect": np.zeros((3, 4)),
        "name": "invalid",
    }
    base.update(change)
    with pytest.raises(ValueError, match=message):
        PanelDataset(**base)


def test_nonabsorbing_treatment_rejects_adoption_times() -> None:
    treatment = np.array([[0.0, 1.0, 0.0]])
    panel = PanelDataset(
        outcome=np.zeros((1, 3)),
        treatment=treatment,
        untreated_outcome=np.zeros((1, 3)),
        treatment_effect=np.zeros((1, 3)),
        name="switching",
    )

    assert not panel.is_absorbing
    with pytest.raises(ValueError, match="absorbing"):
        _ = panel.adoption_times
