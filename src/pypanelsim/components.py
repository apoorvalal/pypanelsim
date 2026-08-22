"""Composable assignment, outcome, and effect components."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
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
class UnitFeatureDraw:
    """Observed covariates and latent unit features shared by DGP components."""

    observables: FloatMatrix
    unobservables: FloatMatrix
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimeFeatureDraw:
    """Time-varying features shared by outcome and effect components."""

    values: FloatMatrix
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _unit_feature_matrix(
    value: Any | None,
    *,
    n_units: int,
    name: str,
) -> FloatMatrix:
    if value is None:
        array = np.empty((n_units, 0), dtype=float)
    else:
        array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != n_units:
        raise ValueError(f"{name} must have shape (n_units, n_features)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _time_feature_matrix(
    value: Any | None,
    *,
    n_periods: int,
    name: str,
) -> FloatMatrix:
    if value is None:
        array = np.empty((n_periods, 0), dtype=float)
    else:
        array = np.array(value, dtype=float, copy=True)
    if array.ndim != 2 or array.shape[0] != n_periods:
        raise ValueError(f"{name} must have shape (n_periods, n_features)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class AssignmentContext:
    """Dimensions and unit features available to an assignment mechanism."""

    dimensions: PanelDimensions
    observables: FloatMatrix | None = None
    unobservables: FloatMatrix | None = None
    feature_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observables",
            _unit_feature_matrix(
                self.observables,
                n_units=self.dimensions.n_units,
                name="observables",
            ),
        )
        object.__setattr__(
            self,
            "unobservables",
            _unit_feature_matrix(
                self.unobservables,
                n_units=self.dimensions.n_units,
                name="unobservables",
            ),
        )
        object.__setattr__(
            self,
            "feature_metadata",
            MappingProxyType(dict(self.feature_metadata)),
        )

    @property
    def n_units(self) -> int:
        """Proxy the unit count for legacy assignment components."""

        return self.dimensions.n_units

    @property
    def n_periods(self) -> int:
        """Proxy the period count for legacy assignment components."""

        return self.dimensions.n_periods


@dataclass(frozen=True, slots=True)
class SimulationContext:
    """Dimensions and realized treatment passed to outcome and effect models."""

    dimensions: PanelDimensions
    treatment: FloatMatrix
    observables: FloatMatrix | None = None
    unobservables: FloatMatrix | None = None
    feature_metadata: Mapping[str, Any] = field(default_factory=dict)
    time_features: FloatMatrix | None = None
    time_feature_metadata: Mapping[str, Any] = field(default_factory=dict)

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
        object.__setattr__(
            self,
            "observables",
            _unit_feature_matrix(
                self.observables,
                n_units=self.dimensions.n_units,
                name="observables",
            ),
        )
        object.__setattr__(
            self,
            "unobservables",
            _unit_feature_matrix(
                self.unobservables,
                n_units=self.dimensions.n_units,
                name="unobservables",
            ),
        )
        object.__setattr__(
            self,
            "feature_metadata",
            MappingProxyType(dict(self.feature_metadata)),
        )
        object.__setattr__(
            self,
            "time_features",
            _time_feature_matrix(
                self.time_features,
                n_periods=self.dimensions.n_periods,
                name="time_features",
            ),
        )
        object.__setattr__(
            self,
            "time_feature_metadata",
            MappingProxyType(dict(self.time_feature_metadata)),
        )

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
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        """Draw a treatment matrix."""


@runtime_checkable
class UnitFeatureModel(Protocol):
    """Protocol for shared observed and latent unit features."""

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> UnitFeatureDraw:
        """Draw unit features before treatment assignment."""


@runtime_checkable
class TimeFeatureModel(Protocol):
    """Protocol for time-varying features shared by DGP components."""

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> TimeFeatureDraw:
        """Draw one feature row per panel period."""


@dataclass(frozen=True, slots=True)
class GaussianUnitFeatures:
    """Draw independent standard-normal observed and latent unit features."""

    n_observables: int = 2
    n_unobservables: int = 2

    def __post_init__(self) -> None:
        if self.n_observables < 0:
            raise ValueError("n_observables must be nonnegative")
        if self.n_unobservables < 0:
            raise ValueError("n_unobservables must be nonnegative")

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> UnitFeatureDraw:
        observables = rng.normal(size=(dimensions.n_units, self.n_observables))
        unobservables = rng.normal(size=(dimensions.n_units, self.n_unobservables))
        return UnitFeatureDraw(
            observables,
            unobservables,
            {
                "kind": "gaussian_unit_features",
                "observable_names": tuple(
                    f"x{index + 1}" for index in range(self.n_observables)
                ),
                "unobservable_names": tuple(
                    f"u{index + 1}" for index in range(self.n_unobservables)
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class GaussianTimeFeatures:
    """Draw independent standard-normal time-varying features."""

    n_features: int = 1

    def __post_init__(self) -> None:
        if self.n_features < 0:
            raise ValueError("n_features must be nonnegative")

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> TimeFeatureDraw:
        values = rng.normal(size=(dimensions.n_periods, self.n_features))
        return TimeFeatureDraw(
            values,
            {
                "kind": "gaussian_time_features",
                "feature_names": tuple(
                    f"v{index + 1}" for index in range(self.n_features)
                ),
            },
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adoption_times",
            MappingProxyType(dict(self.adoption_times)),
        )

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


def _as_assignment_context(
    value: AssignmentContext | PanelDimensions,
) -> AssignmentContext:
    if isinstance(value, AssignmentContext):
        return value
    if isinstance(value, PanelDimensions):
        return AssignmentContext(value)
    raise TypeError("assignment context must contain panel dimensions")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(values, dtype=float)
    nonnegative = values >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    return probabilities


@dataclass(frozen=True, slots=True)
class RandomizedSingleCohortAssignment:
    """Randomly assign a fixed-size absorbing cohort without replacement."""

    n_treated: int
    adoption_period: int
    eligible_units: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if self.n_treated <= 0:
            raise ValueError("n_treated must be positive")
        if self.adoption_period < 0:
            raise ValueError("adoption_period must be nonnegative")
        if self.eligible_units is not None:
            object.__setattr__(self, "eligible_units", tuple(self.eligible_units))

    def assign(
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        assignment_context = _as_assignment_context(context)
        dimensions = assignment_context.dimensions
        if self.adoption_period >= dimensions.n_periods:
            raise ValueError("adoption_period must be a valid time position")
        if self.eligible_units is None:
            eligible = np.arange(dimensions.n_units, dtype=np.int64)
        else:
            eligible = np.asarray(self.eligible_units, dtype=np.int64)
            if eligible.ndim != 1 or np.unique(eligible).size != eligible.size:
                raise ValueError("eligible_units must contain unique positions")
            if np.any((eligible < 0) | (eligible >= dimensions.n_units)):
                raise ValueError("eligible_units contains an invalid unit position")
        if self.n_treated > eligible.size:
            raise ValueError("n_treated cannot exceed the eligible unit count")

        treated = np.sort(
            rng.choice(eligible, size=self.n_treated, replace=False).astype(np.int64)
        )
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
        treatment[treated, self.adoption_period :] = 1.0
        propensity = np.zeros(dimensions.n_units, dtype=float)
        propensity[eligible] = self.n_treated / eligible.size
        return ComponentDraw(
            treatment,
            {
                "kind": "randomized_single_cohort",
                "treated_units": treated,
                "eligible_units": eligible,
                "adoption_period": self.adoption_period,
                "propensity_scores": propensity,
            },
        )


@dataclass(frozen=True, slots=True)
class BinaryLogitAssignment:
    """Assign one cohort using a sigmoid propensity based on unit features.

    Observable coefficients give assignment that is unconfounded conditional on
    the recorded covariates. Nonzero unobservable coefficients select on latent
    unit features that an estimator does not receive as covariates.
    """

    adoption_period: int
    intercept: float = 0.0
    observable_coefficients: Sequence[float] = ()
    unobservable_coefficients: Sequence[float] = ()

    def __post_init__(self) -> None:
        if self.adoption_period < 0:
            raise ValueError("adoption_period must be nonnegative")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        observable = tuple(float(value) for value in self.observable_coefficients)
        unobservable = tuple(float(value) for value in self.unobservable_coefficients)
        if not np.all(np.isfinite(observable)):
            raise ValueError("observable_coefficients must be finite")
        if not np.all(np.isfinite(unobservable)):
            raise ValueError("unobservable_coefficients must be finite")
        object.__setattr__(self, "observable_coefficients", observable)
        object.__setattr__(self, "unobservable_coefficients", unobservable)

    def assign(
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        assignment_context = _as_assignment_context(context)
        dimensions = assignment_context.dimensions
        if self.adoption_period >= dimensions.n_periods:
            raise ValueError("adoption_period must be a valid time position")

        observable = np.asarray(self.observable_coefficients, dtype=float)
        unobservable = np.asarray(self.unobservable_coefficients, dtype=float)
        if observable.size != assignment_context.observables.shape[1]:
            raise ValueError(
                "observable_coefficients must match the observable feature count"
            )
        if unobservable.size != assignment_context.unobservables.shape[1]:
            raise ValueError(
                "unobservable_coefficients must match the unobservable feature count"
            )

        linear_predictor = np.full(dimensions.n_units, self.intercept, dtype=float)
        linear_predictor += assignment_context.observables @ observable
        linear_predictor += assignment_context.unobservables @ unobservable
        propensity = _sigmoid(linear_predictor)
        treated_mask = rng.random(dimensions.n_units) < propensity
        treated = np.flatnonzero(treated_mask).astype(np.int64, copy=False)
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
        treatment[treated, self.adoption_period :] = 1.0
        return ComponentDraw(
            treatment,
            {
                "kind": "binary_logit",
                "treated_units": treated,
                "adoption_period": self.adoption_period,
                "linear_predictor": linear_predictor,
                "propensity_scores": propensity,
                "observable_coefficients": observable,
                "unobservable_coefficients": unobservable,
            },
        )


def _coefficient_matrix(
    coefficients: Sequence[Sequence[float]],
    *,
    n_cohorts: int,
    n_features: int,
    name: str,
) -> np.ndarray:
    if len(coefficients) == 0:
        return np.zeros((n_cohorts, n_features), dtype=float)
    matrix = np.asarray(coefficients, dtype=float)
    expected = (n_cohorts, n_features)
    if matrix.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


@dataclass(frozen=True, slots=True)
class GeneralizedPropensityAssignment:
    """Map units to adoption cohorts with a multinomial-logit GPS.

    The supplied adoption periods are the non-baseline categories. Never treated
    is the baseline category with a normalized linear predictor of zero.
    """

    adoption_periods: Sequence[int]
    intercepts: Sequence[float] = ()
    observable_coefficients: Sequence[Sequence[float]] = ()
    unobservable_coefficients: Sequence[Sequence[float]] = ()

    def __post_init__(self) -> None:
        adoption_periods = tuple(int(value) for value in self.adoption_periods)
        if not adoption_periods:
            raise ValueError("adoption_periods must contain at least one cohort")
        if len(set(adoption_periods)) != len(adoption_periods):
            raise ValueError("adoption_periods must be unique")
        intercepts = tuple(float(value) for value in self.intercepts)
        observable = tuple(
            tuple(float(value) for value in row) for row in self.observable_coefficients
        )
        unobservable = tuple(
            tuple(float(value) for value in row)
            for row in self.unobservable_coefficients
        )
        object.__setattr__(self, "adoption_periods", adoption_periods)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "observable_coefficients", observable)
        object.__setattr__(self, "unobservable_coefficients", unobservable)

    def assign(
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        assignment_context = _as_assignment_context(context)
        dimensions = assignment_context.dimensions
        periods = np.asarray(self.adoption_periods, dtype=np.int64)
        if np.any((periods < 0) | (periods >= dimensions.n_periods)):
            raise ValueError("adoption_periods must contain valid time positions")
        n_cohorts = periods.size
        if len(self.intercepts) == 0:
            intercepts = np.zeros(n_cohorts, dtype=float)
        else:
            intercepts = np.asarray(self.intercepts, dtype=float)
            if intercepts.shape != (n_cohorts,):
                raise ValueError("intercepts must match the adoption cohort count")
            if not np.all(np.isfinite(intercepts)):
                raise ValueError("intercepts must be finite")
        observable = _coefficient_matrix(
            self.observable_coefficients,
            n_cohorts=n_cohorts,
            n_features=assignment_context.observables.shape[1],
            name="observable_coefficients",
        )
        unobservable = _coefficient_matrix(
            self.unobservable_coefficients,
            n_cohorts=n_cohorts,
            n_features=assignment_context.unobservables.shape[1],
            name="unobservable_coefficients",
        )

        cohort_logits = np.broadcast_to(intercepts, (dimensions.n_units, n_cohorts))
        cohort_logits = np.array(cohort_logits, copy=True)
        cohort_logits += assignment_context.observables @ observable.T
        cohort_logits += assignment_context.unobservables @ unobservable.T
        logits = np.column_stack((cohort_logits, np.zeros(dimensions.n_units)))
        centered = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(centered)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
        uniforms = rng.random(dimensions.n_units)
        categories = np.sum(
            uniforms[:, None] > np.cumsum(probabilities, axis=1), axis=1
        )
        categories = np.minimum(categories, n_cohorts).astype(np.int64, copy=False)

        adoption_times = np.full(
            dimensions.n_units, dimensions.n_periods, dtype=np.int64
        )
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
        for category, period in enumerate(periods):
            units = np.flatnonzero(categories == category)
            adoption_times[units] = period
            treatment[units, period:] = 1.0
        return ComponentDraw(
            treatment,
            {
                "kind": "generalized_propensity",
                "adoption_periods": periods,
                "adoption_times": adoption_times,
                "assigned_categories": categories,
                "generalized_propensity_scores": probabilities,
                "cohort_linear_predictors": cohort_logits,
                "intercepts": intercepts,
                "observable_coefficients": observable,
                "unobservable_coefficients": unobservable,
                "never_treated_category": n_cohorts,
            },
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


EffectCallable = Callable[[SimulationContext], Any]
UnitEffectCallable = EffectCallable


@dataclass(frozen=True, slots=True)
class CallableEffect:
    """Turn a callable effect surface into an :class:`EffectModel`.

    The callable receives the shared :class:`SimulationContext` and returns
    a finite scalar, unit vector, time vector, or full unit-by-time surface.
    Normalized effects are realized only in treated cells.
    """

    function: EffectCallable
    name: str = "callable_effect"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("function must be callable")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        raw_effects = np.asarray(self.function(context), dtype=float)
        n_units = context.dimensions.n_units
        n_periods = context.dimensions.n_periods
        metadata: dict[str, Any] = {"kind": self.name}
        if raw_effects.ndim == 0:
            unit_effects = np.full(n_units, float(raw_effects), dtype=float)
            effect_surface = np.repeat(unit_effects[:, None], n_periods, axis=1)
            metadata["normalization"] = "scalar"
            metadata["unit_effects"] = unit_effects
        elif raw_effects.shape == (n_units, 1):
            unit_effects = np.array(raw_effects[:, 0], dtype=float, copy=True)
            effect_surface = np.repeat(unit_effects[:, None], n_periods, axis=1)
            metadata["normalization"] = "unit"
            metadata["unit_effects"] = unit_effects
        elif raw_effects.shape == (n_units,):
            unit_effects = np.array(raw_effects, dtype=float, copy=True)
            effect_surface = np.repeat(unit_effects[:, None], n_periods, axis=1)
            metadata["normalization"] = "unit"
            metadata["unit_effects"] = unit_effects
        elif raw_effects.shape == (1, n_periods):
            time_effects = np.array(raw_effects[0], dtype=float, copy=True)
            effect_surface = np.repeat(time_effects[None, :], n_units, axis=0)
            metadata["normalization"] = "time"
            metadata["time_effects"] = time_effects
        elif n_periods != n_units and raw_effects.shape == (n_periods,):
            time_effects = np.array(raw_effects, dtype=float, copy=True)
            effect_surface = np.repeat(time_effects[None, :], n_units, axis=0)
            metadata["normalization"] = "time"
            metadata["time_effects"] = time_effects
        elif raw_effects.shape == (n_units, n_periods):
            effect_surface = np.array(raw_effects, dtype=float, copy=True)
            metadata["normalization"] = "surface"
        else:
            raise ValueError(
                "callable effect must return a scalar or have shape "
                f"({n_units},), ({n_units}, 1), ({n_periods},), "
                f"(1, {n_periods}), or ({n_units}, {n_periods})"
            )
        if not np.all(np.isfinite(effect_surface)):
            raise ValueError("callable effect must contain only finite values")
        metadata["effect_surface"] = effect_surface
        for value in metadata.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return ComponentDraw(
            effect_surface * context.treatment,
            metadata,
        )


@dataclass(frozen=True, slots=True)
class CallableUnitEffect(CallableEffect):
    """Backward-compatible name for callable unit or unit-time effects."""

    name: str = "callable_unit_effect"


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
