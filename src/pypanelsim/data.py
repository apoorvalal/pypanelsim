"""Validated data containers for synthetic panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int64]


def _readonly_matrix(value: Any, *, name: str) -> FloatMatrix:
    array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_ids(value: Any, *, length: int, name: str) -> NDArray[Any]:
    array = np.array(value, copy=True)
    if array.ndim != 1 or array.size != length:
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if np.unique(array).size != length:
        raise ValueError(f"{name} must contain unique values")
    array.setflags(write=False)
    return array


def _readonly_unit_covariates(value: Any | None, *, n_units: int) -> FloatMatrix:
    if value is None:
        array = np.empty((n_units, 0), dtype=float)
    else:
        array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != n_units:
        raise ValueError("unit_covariates must have shape (n_units, n_covariates)")
    if not np.all(np.isfinite(array)):
        raise ValueError("unit_covariates must contain only finite values")
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PanelDataset:
    """One simulated panel and its causal decomposition.

    All matrices use the estimator-neutral ``(unit, time)`` layout. Treatment
    is binary. ``treatment_effect`` is the realized cell effect, so it is zero
    outside treated cells and ``outcome = untreated_outcome + treatment_effect``.
    The container owns read-only copies of all arrays.
    """

    outcome: FloatMatrix
    treatment: FloatMatrix
    untreated_outcome: FloatMatrix
    treatment_effect: FloatMatrix
    name: str
    unit_ids: NDArray[Any] | None = None
    time_ids: NDArray[Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unit_covariates: FloatMatrix | None = None
    unit_covariate_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        outcome = _readonly_matrix(self.outcome, name="outcome")
        treatment = _readonly_matrix(self.treatment, name="treatment")
        untreated = _readonly_matrix(self.untreated_outcome, name="untreated_outcome")
        effect = _readonly_matrix(self.treatment_effect, name="treatment_effect")
        shape = outcome.shape
        if any(array.shape != shape for array in (treatment, untreated, effect)):
            raise ValueError("all panel matrices must have the same shape")
        if not np.all((treatment == 0.0) | (treatment == 1.0)):
            raise ValueError("treatment must contain only zero and one")
        if not np.allclose(effect[treatment == 0.0], 0.0):
            raise ValueError("treatment_effect must be zero outside treated cells")
        if not np.allclose(outcome, untreated + effect):
            raise ValueError(
                "outcome must equal untreated_outcome plus treatment_effect"
            )

        n_units, n_periods = shape
        unit_covariates = _readonly_unit_covariates(
            self.unit_covariates, n_units=n_units
        )
        if self.unit_covariate_names is None:
            unit_covariate_names = tuple(
                f"x{index + 1}" for index in range(unit_covariates.shape[1])
            )
        else:
            unit_covariate_names = tuple(self.unit_covariate_names)
        if len(unit_covariate_names) != unit_covariates.shape[1]:
            raise ValueError("unit_covariate_names must match the unit covariate count")
        if any(not isinstance(name, str) or not name for name in unit_covariate_names):
            raise ValueError("unit_covariate_names must contain non-empty strings")
        if len(set(unit_covariate_names)) != len(unit_covariate_names):
            raise ValueError("unit_covariate_names must be unique")
        reserved = {
            "unit",
            "time",
            "outcome",
            "treatment",
            "untreated_outcome",
            "treatment_effect",
        }
        if reserved.intersection(unit_covariate_names):
            raise ValueError("unit_covariate_names cannot use reserved panel names")
        unit_ids = (
            np.arange(n_units, dtype=np.int64)
            if self.unit_ids is None
            else self.unit_ids
        )
        time_ids = (
            np.arange(n_periods, dtype=np.int64)
            if self.time_ids is None
            else self.time_ids
        )

        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "untreated_outcome", untreated)
        object.__setattr__(self, "treatment_effect", effect)
        object.__setattr__(self, "unit_covariates", unit_covariates)
        object.__setattr__(self, "unit_covariate_names", unit_covariate_names)
        object.__setattr__(
            self,
            "unit_ids",
            _readonly_ids(unit_ids, length=n_units, name="unit_ids"),
        )
        object.__setattr__(
            self,
            "time_ids",
            _readonly_ids(time_ids, length=n_periods, name="time_ids"),
        )
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_units, n_periods)``."""

        return self.outcome.shape

    @property
    def n_units(self) -> int:
        """Return the number of panel units."""

        return self.shape[0]

    @property
    def n_periods(self) -> int:
        """Return the number of time periods."""

        return self.shape[1]

    @property
    def ever_treated(self) -> NDArray[np.bool_]:
        """Return a unit-level mask for units treated at least once."""

        mask = np.any(self.treatment == 1.0, axis=1)
        mask.setflags(write=False)
        return mask

    @property
    def treated_units(self) -> IntVector:
        """Return integer positions of ever-treated units."""

        positions = np.flatnonzero(self.ever_treated).astype(np.int64, copy=False)
        positions.setflags(write=False)
        return positions

    @property
    def control_units(self) -> IntVector:
        """Return integer positions of never-treated units."""

        positions = np.flatnonzero(~self.ever_treated).astype(np.int64, copy=False)
        positions.setflags(write=False)
        return positions

    @property
    def is_absorbing(self) -> bool:
        """Return whether treatment never switches off after adoption."""

        return bool(np.all(np.diff(self.treatment, axis=1) >= 0.0))

    @property
    def adoption_times(self) -> IntVector:
        """Return first-treatment positions; never-treated units use ``n_periods``."""

        if not self.is_absorbing:
            raise ValueError("adoption_times requires absorbing treatment")
        times = np.full(self.n_units, self.n_periods, dtype=np.int64)
        treated = self.ever_treated
        times[treated] = np.argmax(self.treatment[treated] == 1.0, axis=1)
        times.setflags(write=False)
        return times

    @property
    def true_att(self) -> float:
        """Return the average realized effect over treated cells."""

        treated_cells = self.treatment == 1.0
        if not np.any(treated_cells):
            raise ValueError("true_att is undefined because the panel has no treatment")
        return float(self.treatment_effect[treated_cells].mean())

    def as_arrays(self, *, copy: bool = False) -> tuple[FloatMatrix, FloatMatrix]:
        """Return outcome and treatment matrices for a downstream estimator."""

        if copy:
            return self.outcome.copy(), self.treatment.copy()
        return self.outcome, self.treatment

    def arrays(
        self,
        fields: Sequence[str],
        *,
        copy: bool = False,
    ) -> tuple[FloatMatrix, ...]:
        """Return selected panel matrices in the requested order."""

        allowed = {
            "outcome": self.outcome,
            "treatment": self.treatment,
            "untreated_outcome": self.untreated_outcome,
            "treatment_effect": self.treatment_effect,
        }
        unknown = set(fields).difference(allowed)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown panel fields: {names}")
        values = tuple(allowed[field] for field in fields)
        return tuple(value.copy() for value in values) if copy else values

    def as_long_dict(self, *, copy: bool = False) -> Mapping[str, NDArray[Any]]:
        """Return flat columns suitable for pandas, Polars, Arrow, or formulas."""

        columns: dict[str, NDArray[Any]] = {
            "unit": np.repeat(self.unit_ids, self.n_periods),
            "time": np.tile(self.time_ids, self.n_units),
            "outcome": self.outcome.reshape(-1),
            "treatment": self.treatment.reshape(-1),
            "untreated_outcome": self.untreated_outcome.reshape(-1),
            "treatment_effect": self.treatment_effect.reshape(-1),
        }
        for index, name in enumerate(self.unit_covariate_names):
            columns[name] = np.repeat(self.unit_covariates[:, index], self.n_periods)
        if copy:
            return {name: np.array(value, copy=True) for name, value in columns.items()}
        for value in columns.values():
            value.setflags(write=False)
        return MappingProxyType(columns)
