"""Composable assignment, outcome, and effect components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .data import FloatMatrix, IntVector


@dataclass(frozen=True, slots=True)
class PanelDimensions:
    """The rectangular dimensions of a simulated panel."""

    n_units: int = 200
    n_periods: int = 50

    def __post_init__(self) -> None:
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if self.n_periods <= 0:
            raise ValueError("n_periods must be positive")


@dataclass(frozen=True, slots=True)
class ComponentDraw:
    """One matrix produced by a simulation component."""

    values: FloatMatrix
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationContext:
    """Dimensions and realized treatment passed to outcome and effect models."""

    dimensions: PanelDimensions
    treatment: FloatMatrix

    def __post_init__(self) -> None:
        treatment = np.array(self.treatment, dtype=float, copy=True)
        expected = (self.dimensions.n_units, self.dimensions.n_periods)
        if treatment.shape != expected:
            raise ValueError(f"treatment must have shape {expected}")
        if not np.all(np.isfinite(treatment)):
            raise ValueError("treatment must contain only finite values")
        if not np.all((treatment == 0.0) | (treatment == 1.0)):
            raise ValueError("treatment must contain only zero and one")
        treatment.setflags(write=False)
        object.__setattr__(self, "treatment", treatment)

    @property
    def ever_treated(self) -> np.ndarray:
        """Return a unit-level mask for units treated at least once."""

        return np.any(self.treatment == 1.0, axis=1)

    @property
    def treated_units(self) -> IntVector:
        """Return integer positions of ever-treated units."""

        return np.flatnonzero(self.ever_treated).astype(np.int64, copy=False)

    @property
    def control_units(self) -> IntVector:
        """Return integer positions of never-treated units."""

        return np.flatnonzero(~self.ever_treated).astype(np.int64, copy=False)

    @property
    def is_absorbing(self) -> bool:
        """Return whether treatment never switches off after adoption."""

        return bool(np.all(np.diff(self.treatment, axis=1) >= 0.0))

    @property
    def adoption_times(self) -> IntVector:
        """Return first-treatment positions; never-treated units use the horizon."""

        if not self.is_absorbing:
            raise ValueError("adoption_times requires absorbing treatment")
        times = np.full(
            self.dimensions.n_units, self.dimensions.n_periods, dtype=np.int64
        )
        treated = self.ever_treated
        times[treated] = np.argmax(self.treatment[treated] == 1.0, axis=1)
        return times


@runtime_checkable
class AssignmentModel(Protocol):
    """Protocol for treatment-assignment components."""

    def assign(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> ComponentDraw:
        """Draw a treatment matrix."""


@runtime_checkable
class OutcomeModel(Protocol):
    """Protocol for untreated-outcome components."""

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        """Draw an untreated-outcome matrix."""


@runtime_checkable
class EffectModel(Protocol):
    """Protocol for realized treatment-effect components."""

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        """Draw a treatment-effect matrix that is zero when untreated."""


@dataclass(frozen=True, slots=True)
class SingleCohortAssignment:
    """Assign one absorbing treatment cohort.

    If ``treated_units`` is omitted, the last ``n_treated`` unit positions are
    used. Explicit positions permit any unit order without changing the data
    contract.
    """

    n_treated: int = 40
    adoption_period: int = 40
    treated_units: tuple[int, ...] | None = None

    def assign(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        if not 0 < self.n_treated <= dimensions.n_units:
            raise ValueError("n_treated must lie between one and n_units")
        if not 0 <= self.adoption_period < dimensions.n_periods:
            raise ValueError("adoption_period must be a valid time position")
        if self.treated_units is None:
            units = np.arange(
                dimensions.n_units - self.n_treated,
                dimensions.n_units,
                dtype=np.int64,
            )
        else:
            units = np.asarray(self.treated_units, dtype=np.int64)
            if units.ndim != 1 or units.size != self.n_treated:
                raise ValueError("treated_units must contain n_treated positions")
            if np.unique(units).size != units.size:
                raise ValueError("treated_units must be unique")
            if np.any((units < 0) | (units >= dimensions.n_units)):
                raise ValueError("treated_units contains an invalid unit position")
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
        treatment[units, self.adoption_period :] = 1.0
        return ComponentDraw(
            treatment,
            {
                "kind": "single_cohort",
                "treated_units": units,
                "adoption_period": self.adoption_period,
            },
        )


@dataclass(frozen=True, slots=True)
class StaggeredAdoption:
    """Assign absorbing treatment from a mapping of unit to adoption period."""

    adoption_times: Mapping[int, int]

    def assign(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
        normalized: dict[int, int] = {}
        for raw_unit, raw_period in self.adoption_times.items():
            unit = int(raw_unit)
            period = int(raw_period)
            if not 0 <= unit < dimensions.n_units:
                raise ValueError(f"invalid treated unit position: {unit}")
            if not 0 <= period < dimensions.n_periods:
                raise ValueError(f"invalid adoption period for unit {unit}: {period}")
            treatment[unit, period:] = 1.0
            normalized[unit] = period
        return ComponentDraw(
            treatment,
            {"kind": "staggered_adoption", "adoption_times": normalized},
        )


@dataclass(frozen=True, slots=True)
class LinearRampEffect:
    """Apply ``slope * event_time`` after each unit adopts treatment."""

    slope: float = 0.2

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope):
            raise ValueError("slope must be finite")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        adoption = context.adoption_times
        periods = np.arange(context.dimensions.n_periods)
        event_time = periods[None, :] - adoption[:, None] + 1
        values = self.slope * np.maximum(event_time, 0) * context.treatment
        return ComponentDraw(values, {"kind": "linear_ramp", "slope": self.slope})


@dataclass(frozen=True, slots=True)
class ConstantEffect:
    """Apply one constant effect to every treated cell."""

    value: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("value must be finite")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        return ComponentDraw(
            self.value * context.treatment,
            {"kind": "constant", "value": self.value},
        )


OutcomeCallable = Callable[
    [SimulationContext, np.random.Generator], ComponentDraw | FloatMatrix
]


@dataclass(frozen=True, slots=True)
class CallableOutcomeModel:
    """Adapt a plain Python callable to the :class:`OutcomeModel` protocol."""

    function: OutcomeCallable
    name: str = "callable_outcome"

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        result = self.function(context, rng)
        if isinstance(result, ComponentDraw):
            return result
        return ComponentDraw(np.asarray(result, dtype=float), {"kind": self.name})
