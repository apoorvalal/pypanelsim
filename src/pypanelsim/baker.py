"""Baker-Larcker-Wang staggered-adoption event-study design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import (
    AssignmentContext,
    ComponentDraw,
    PanelDimensions,
    SimulationContext,
    UnitFeatureDraw,
)
from .data import PanelDataset
from .simulator import PanelSimulator


@dataclass(frozen=True, slots=True)
class BakerPanelConfig:
    """Configuration for the normalized Baker event-study failure design.

    The original replication samples fixed effects and residuals from private
    Compustat data. This self-contained version preserves its balanced time
    range, randomized state cohorts, and heterogeneous dynamic effect law while
    using configurable Gaussian additive fixed effects and noise.
    """

    n_units: int = 1_000
    start_year: int = 1980
    end_year: int = 2015
    adoption_years: tuple[int, ...] = (1989, 1998, 2007)
    cohort_state_counts: tuple[int, ...] = (17, 18, 15)
    cohort_effect_slopes: tuple[float, ...] = (0.10, 0.05, 0.01)
    effect_scale: float = 1.0
    unit_effect_scale: float = 1.0
    time_effect_scale: float = 0.2
    noise_scale: float = 0.5

    def __post_init__(self) -> None:
        adoption_years = tuple(int(value) for value in self.adoption_years)
        state_counts = tuple(int(value) for value in self.cohort_state_counts)
        slopes = tuple(float(value) for value in self.cohort_effect_slopes)
        object.__setattr__(self, "adoption_years", adoption_years)
        object.__setattr__(self, "cohort_state_counts", state_counts)
        object.__setattr__(self, "cohort_effect_slopes", slopes)

        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        if self.end_year < self.start_year:
            raise ValueError("end_year must not precede start_year")
        if len(adoption_years) == 0:
            raise ValueError("adoption_years must contain at least one cohort")
        if not (len(adoption_years) == len(state_counts) == len(slopes)):
            raise ValueError(
                "adoption_years, cohort_state_counts, and "
                "cohort_effect_slopes must have equal length"
            )
        if tuple(sorted(adoption_years)) != adoption_years or len(
            set(adoption_years)
        ) != len(adoption_years):
            raise ValueError("adoption_years must be strictly increasing")
        if any(
            year < self.start_year or year > self.end_year for year in adoption_years
        ):
            raise ValueError("adoption_years must lie inside the panel")
        if any(count <= 0 for count in state_counts):
            raise ValueError("cohort_state_counts must be positive")
        if not np.all(np.isfinite(slopes)):
            raise ValueError("cohort_effect_slopes must be finite")
        for name in (
            "effect_scale",
            "unit_effect_scale",
            "time_effect_scale",
            "noise_scale",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def n_states(self) -> int:
        """Number of assignment clusters."""

        return sum(self.cohort_state_counts)

    @property
    def n_periods(self) -> int:
        """Number of calendar periods."""

        return self.end_year - self.start_year + 1

    @property
    def years(self) -> np.ndarray:
        """Calendar years corresponding to panel columns."""

        years = np.arange(self.start_year, self.end_year + 1, dtype=np.int64)
        years.setflags(write=False)
        return years

    @property
    def adoption_periods(self) -> tuple[int, ...]:
        """Zero-based adoption positions corresponding to ``adoption_years``."""

        return tuple(year - self.start_year for year in self.adoption_years)

    @property
    def dimensions(self) -> PanelDimensions:
        """Rectangular dimensions for the generic simulation pipeline."""

        return PanelDimensions(self.n_units, self.n_periods)


@dataclass(frozen=True, slots=True)
class BakerStateFeatures:
    """Sample integer state clusters as an observed unit feature."""

    n_states: int

    def __post_init__(self) -> None:
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> UnitFeatureDraw:
        state_ids = rng.integers(
            0,
            self.n_states,
            size=dimensions.n_units,
            dtype=np.int64,
        )
        return UnitFeatureDraw(
            observables=state_ids[:, None],
            unobservables=np.empty((dimensions.n_units, 0), dtype=float),
            metadata={
                "kind": "baker_state_clusters",
                "observable_names": ("state",),
                "state_ids": state_ids,
                "n_states": self.n_states,
            },
        )


@dataclass(frozen=True, slots=True)
class BakerCohortAssignment:
    """Randomize state clusters across Baker's staggered adoption cohorts."""

    adoption_periods: tuple[int, ...]
    cohort_state_counts: tuple[int, ...]
    start_year: int = 1980

    def __post_init__(self) -> None:
        periods = tuple(int(value) for value in self.adoption_periods)
        counts = tuple(int(value) for value in self.cohort_state_counts)
        object.__setattr__(self, "adoption_periods", periods)
        object.__setattr__(self, "cohort_state_counts", counts)
        if len(periods) == 0 or len(periods) != len(counts):
            raise ValueError(
                "adoption_periods and cohort_state_counts must have equal "
                "nonzero length"
            )
        if len(set(periods)) != len(periods) or tuple(sorted(periods)) != periods:
            raise ValueError("adoption_periods must be strictly increasing")
        if any(count <= 0 for count in counts):
            raise ValueError("cohort_state_counts must be positive")

    def assign(
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        if not isinstance(context, AssignmentContext):
            raise ValueError("Baker cohort assignment requires state features")
        dimensions = context.dimensions
        periods = np.asarray(self.adoption_periods, dtype=np.int64)
        if np.any((periods < 0) | (periods >= dimensions.n_periods)):
            raise ValueError("adoption_periods must contain valid time positions")
        if context.observables.shape[1] < 1:
            raise ValueError("Baker cohort assignment requires a state covariate")

        raw_states = context.observables[:, 0]
        state_ids = np.rint(raw_states).astype(np.int64)
        if not np.allclose(raw_states, state_ids):
            raise ValueError("state covariate must contain integer identifiers")
        n_states = sum(self.cohort_state_counts)
        if np.any((state_ids < 0) | (state_ids >= n_states)):
            raise ValueError("state identifiers must lie between zero and n_states - 1")

        randomized_states = rng.permutation(n_states)
        cohort_by_state = np.empty(n_states, dtype=np.int64)
        start = 0
        for period, count in zip(periods, self.cohort_state_counts, strict=True):
            cohort_by_state[randomized_states[start : start + count]] = period
            start += count
        cohort_by_unit = cohort_by_state[state_ids]
        treatment = (
            np.arange(dimensions.n_periods)[None, :] >= cohort_by_unit[:, None]
        ).astype(float)
        return ComponentDraw(
            treatment,
            {
                "kind": "baker_randomized_state_cohorts",
                "state_ids": state_ids,
                "randomized_states": randomized_states,
                "cohort_period_by_state": cohort_by_state,
                "cohort_period_by_unit": cohort_by_unit,
                "adoption_periods": periods,
                "adoption_years": periods + self.start_year,
                "cohort_state_counts": np.asarray(
                    self.cohort_state_counts, dtype=np.int64
                ),
            },
        )


@dataclass(frozen=True, slots=True)
class BakerAdditiveOutcome:
    """Gaussian additive unit and time fixed effects plus idiosyncratic noise."""

    unit_effect_scale: float = 1.0
    time_effect_scale: float = 0.2
    noise_scale: float = 0.5

    def __post_init__(self) -> None:
        for name in ("unit_effect_scale", "time_effect_scale", "noise_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        unit_effects = rng.normal(
            scale=self.unit_effect_scale,
            size=context.dimensions.n_units,
        )
        time_effects = rng.normal(
            scale=self.time_effect_scale,
            size=context.dimensions.n_periods,
        )
        errors = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        values = unit_effects[:, None] + time_effects[None, :] + errors
        return ComponentDraw(
            values,
            {
                "kind": "baker_additive_fixed_effects",
                "unit_effects": unit_effects,
                "time_effects": time_effects,
                "errors": errors,
                "unit_effect_scale": self.unit_effect_scale,
                "time_effect_scale": self.time_effect_scale,
                "noise_scale": self.noise_scale,
            },
        )


@dataclass(frozen=True, slots=True)
class BakerDynamicEffect:
    """Cohort-specific linear event-time effects from the Baker failure DGP."""

    adoption_periods: tuple[int, ...]
    cohort_effect_slopes: tuple[float, ...]
    effect_scale: float = 1.0

    def __post_init__(self) -> None:
        periods = tuple(int(value) for value in self.adoption_periods)
        slopes = tuple(float(value) for value in self.cohort_effect_slopes)
        object.__setattr__(self, "adoption_periods", periods)
        object.__setattr__(self, "cohort_effect_slopes", slopes)
        if len(periods) == 0 or len(periods) != len(slopes):
            raise ValueError(
                "adoption_periods and cohort_effect_slopes must have equal "
                "nonzero length"
            )
        if not np.all(np.isfinite(slopes)):
            raise ValueError("cohort_effect_slopes must be finite")
        if not np.isfinite(self.effect_scale) or self.effect_scale < 0.0:
            raise ValueError("effect_scale must be finite and nonnegative")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        adoption = context.adoption_times
        unit_slopes = np.full(context.dimensions.n_units, np.nan, dtype=float)
        for period, slope in zip(
            self.adoption_periods, self.cohort_effect_slopes, strict=True
        ):
            unit_slopes[adoption == period] = slope * self.effect_scale
        if np.any(np.isnan(unit_slopes)):
            raise ValueError("all Baker units must belong to a configured cohort")
        event_time = (
            np.arange(context.dimensions.n_periods)[None, :] - adoption[:, None] + 1
        )
        values = unit_slopes[:, None] * np.maximum(event_time, 0) * context.treatment
        return ComponentDraw(
            values,
            {
                "kind": "baker_dynamic_heterogeneous_effect",
                "adoption_periods": np.asarray(self.adoption_periods, dtype=np.int64),
                "cohort_effect_slopes": np.asarray(
                    self.cohort_effect_slopes, dtype=float
                ),
                "effect_scale": self.effect_scale,
                "unit_effect_slopes": unit_slopes,
            },
        )


def baker_design(config: BakerPanelConfig | None = None) -> PanelSimulator:
    """Build the Baker staggered-adoption event-study failure design."""

    resolved = BakerPanelConfig() if config is None else config
    return PanelSimulator(
        name="baker_event_study",
        dimensions=resolved.dimensions,
        feature_model=BakerStateFeatures(
            n_states=resolved.n_states,
        ),
        assignment=BakerCohortAssignment(
            adoption_periods=resolved.adoption_periods,
            cohort_state_counts=resolved.cohort_state_counts,
            start_year=resolved.start_year,
        ),
        outcome_model=BakerAdditiveOutcome(
            unit_effect_scale=resolved.unit_effect_scale,
            time_effect_scale=resolved.time_effect_scale,
            noise_scale=resolved.noise_scale,
        ),
        effect_model=BakerDynamicEffect(
            adoption_periods=resolved.adoption_periods,
            cohort_effect_slopes=resolved.cohort_effect_slopes,
            effect_scale=resolved.effect_scale,
        ),
    )


def baker(
    *,
    config: BakerPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one Baker event-study panel."""

    return baker_design(config).simulate(seed=seed, rng=rng)
