"""Data-generating processes from the F-test event-study paper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .components import (
    CallableOutcomeModel,
    CohortEventTimeEffect,
    ComponentDraw,
    PanelDimensions,
    RandomizedSingleCohortAssignment,
    RandomizedStaggeredAdoption,
    SimulationContext,
)
from .data import PanelDataset
from .simulator import PanelSimulator, resolve_rng

_TEMPORAL_LABELS = MappingProxyType(
    {
        "constant": "Constant",
        "linear": "Linear",
        "concave": "Concave",
        "positive_then_negative": "Positive then negative",
        "exponential": "Exponential",
        "sinusoidal": "Sinusoidal",
        "random_walk": "Random walk",
    }
)

_COHORT_LABELS = MappingProxyType(
    {
        "homogeneous": "Homogeneous",
        "log_vs_linear_vs_sin": "Log vs linear vs sin",
        "small_differences": "Small differences",
        "large_differences": "Large differences",
        "selection_on_gains": "Selection on gains",
        "novelty_effects": "Novelty effects",
        "activity_bias": "Activity bias",
    }
)


def available_ftest_temporal_designs() -> tuple[str, ...]:
    """Return the seven single-cohort effect laws used in the paper."""

    return tuple(_TEMPORAL_LABELS)


def available_ftest_cohort_designs() -> tuple[str, ...]:
    """Return the seven three-cohort effect laws used in the paper."""

    return tuple(_COHORT_LABELS)


def _validate_scale(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_rho(name: str, value: float) -> None:
    if not np.isfinite(value) or abs(value) >= 1.0:
        raise ValueError(f"{name} must lie strictly between -1 and 1")


@dataclass(frozen=True, slots=True)
class FTestTemporalConfig:
    """Panel and noise parameters for the paper's single-cohort designs.

    The paper uses 50,000 units for its Monte Carlo experiments. The smaller
    default retains the same probability law while keeping an interactive draw
    inexpensive; set ``n_units=50_000`` for the published simulation scale.
    """

    n_units: int = 1_000
    n_periods: int = 35
    adoption_period: int = 15
    n_treated: int | None = None
    max_effect: float = 1.0
    unit_effect_scale: float = 5.0
    time_effect_scale: float = 2.0
    unit_trend_scale: float = 0.01
    noise_scale: float = 2.0
    error_rho: float = 0.7

    def __post_init__(self) -> None:
        if self.n_units <= 1:
            raise ValueError("n_units must exceed one")
        if self.n_periods <= 1:
            raise ValueError("n_periods must exceed one")
        if not 0 <= self.adoption_period < self.n_periods:
            raise ValueError("adoption_period must be a valid time position")
        if self.n_treated is not None and not 0 < self.n_treated < self.n_units:
            raise ValueError("n_treated must lie strictly between zero and n_units")
        if not np.isfinite(self.max_effect):
            raise ValueError("max_effect must be finite")
        for name in (
            "unit_effect_scale",
            "time_effect_scale",
            "unit_trend_scale",
            "noise_scale",
        ):
            _validate_scale(name, getattr(self, name))
        _validate_rho("error_rho", self.error_rho)

    @property
    def treated_count(self) -> int:
        """Return the fixed randomized treatment-cohort size."""

        return self.n_units // 2 if self.n_treated is None else self.n_treated

    @property
    def dimensions(self) -> PanelDimensions:
        """Return generic panel dimensions for composition."""

        return PanelDimensions(self.n_units, self.n_periods)


@dataclass(frozen=True, slots=True)
class FTestCohortConfig:
    """Panel and noise parameters for the paper's three-cohort designs.

    The paper uses 20,000 units. Cohort shares reproduce its 1/8, 1/4, 1/8
    split and leave half of the units never treated.
    """

    n_units: int = 1_000
    n_periods: int = 30
    adoption_periods: tuple[int, int, int] = (10, 15, 20)
    cohort_shares: tuple[float, float, float] = (1 / 8, 1 / 4, 1 / 8)
    unit_effect_scale: float = 2.0
    time_effect_scale: float = 1.0
    noise_scale: float = 1.0
    unit_error_rho: float = 0.8
    time_error_rho: float = 0.2
    periodic_time_effects: tuple[float, ...] = (
        -0.1,
        0.1,
        0.0,
        0.0,
        0.1,
        0.5,
        0.5,
    )

    def __post_init__(self) -> None:
        periods = tuple(int(value) for value in self.adoption_periods)
        shares = tuple(float(value) for value in self.cohort_shares)
        periodic = tuple(float(value) for value in self.periodic_time_effects)
        object.__setattr__(self, "adoption_periods", periods)
        object.__setattr__(self, "cohort_shares", shares)
        object.__setattr__(self, "periodic_time_effects", periodic)
        if self.n_units <= 1:
            raise ValueError("n_units must exceed one")
        if self.n_periods <= 1:
            raise ValueError("n_periods must exceed one")
        if len(periods) != 3 or len(set(periods)) != 3:
            raise ValueError("adoption_periods must contain three unique periods")
        if tuple(sorted(periods)) != periods:
            raise ValueError("adoption_periods must be strictly increasing")
        if any(period < 0 or period >= self.n_periods for period in periods):
            raise ValueError("adoption_periods must lie inside the panel")
        if len(shares) != 3 or not np.all(np.isfinite(shares)):
            raise ValueError("cohort_shares must contain three finite values")
        if any(share <= 0.0 for share in shares) or sum(shares) > 1.0:
            raise ValueError("cohort_shares must be positive and sum to at most one")
        if not periodic or not np.all(np.isfinite(periodic)):
            raise ValueError("periodic_time_effects must be nonempty and finite")
        for name in ("unit_effect_scale", "time_effect_scale", "noise_scale"):
            _validate_scale(name, getattr(self, name))
        _validate_rho("unit_error_rho", self.unit_error_rho)
        _validate_rho("time_error_rho", self.time_error_rho)
        if any(size <= 0 for size in self.cohort_sizes):
            raise ValueError("n_units is too small for the requested cohort shares")

    @property
    def cohort_sizes(self) -> tuple[int, ...]:
        """Return fixed cohort counts, flooring each configured share."""

        return tuple(int(self.n_units * share) for share in self.cohort_shares)

    @property
    def dimensions(self) -> PanelDimensions:
        """Return generic panel dimensions for composition."""

        return PanelDimensions(self.n_units, self.n_periods)


def _unknown_design(name: str, labels: Mapping[str, str], suite: str) -> None:
    available = ", ".join(labels)
    raise KeyError(f"unknown {suite} F-test design {name!r}; available: {available}")


def _freeze_profile(values: Any) -> np.ndarray:
    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1 or profile.size == 0 or not np.all(np.isfinite(profile)):
        raise ValueError("effect profile must be a nonempty finite vector")
    profile = np.array(profile, copy=True)
    profile.setflags(write=False)
    return profile


def ftest_temporal_profile(
    name: str,
    *,
    config: FTestTemporalConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct one of the paper's seven post-adoption effect paths."""

    resolved = FTestTemporalConfig() if config is None else config
    if name not in _TEMPORAL_LABELS:
        _unknown_design(name, _TEMPORAL_LABELS, "temporal")
    length = resolved.n_periods - resolved.adoption_period
    scale = resolved.max_effect
    event_time = np.arange(length, dtype=float)
    if name == "constant":
        profile = np.full(length, scale)
    elif name == "linear":
        profile = np.linspace(0.0, scale, length)
    elif name == "concave":
        profile = scale * 0.5 * np.log(2.0 * (event_time + 1.0) / length + 1.0)
    elif name == "positive_then_negative":
        midpoint = length // 2
        profile = np.concatenate(
            (
                np.linspace(0.0, scale, midpoint),
                np.linspace(scale, -scale, length - midpoint),
            )
        )
    elif name == "exponential":
        profile = scale * (1.0 - np.exp(-np.linspace(0.0, 5.0, length)))
    elif name == "sinusoidal":
        profile = scale * np.sin(np.linspace(0.0, 2.0 * np.pi, length))
    else:
        generator = resolve_rng(seed=seed, rng=rng)
        profile = scale * np.cumsum(generator.normal(size=length))
    return _freeze_profile(profile)


def ftest_cohort_profiles(
    name: str,
    *,
    config: FTestCohortConfig | None = None,
) -> Mapping[int, np.ndarray]:
    """Construct the cohort/event-time profiles for one paper design."""

    resolved = FTestCohortConfig() if config is None else config
    if name not in _COHORT_LABELS:
        _unknown_design(name, _COHORT_LABELS, "cohort")
    periods = resolved.adoption_periods
    lengths = tuple(resolved.n_periods - period for period in periods)
    logs = tuple(np.log(np.arange(1, length + 1)) for length in lengths)

    if name == "homogeneous":
        profiles = logs
    elif name == "log_vs_linear_vs_sin":
        first_length = lengths[0]
        if first_length < 10:
            raise ValueError(
                "log_vs_linear_vs_sin requires at least ten periods after "
                "the first adoption"
            )
        profiles = (
            np.concatenate(
                (
                    np.linspace(2.0, 0.0, first_length - 10),
                    np.zeros(10),
                )
            ),
            np.log(2.0 * np.arange(1, lengths[1] + 1)),
            np.sin(np.arange(1, lengths[2] + 1)),
        )
    elif name == "small_differences":
        profiles = tuple(
            profile * (1.0 + 0.1 * index) for index, profile in enumerate(logs)
        )
    elif name == "large_differences":
        profiles = tuple(
            profile * (period / 10.0)
            for period, profile in zip(periods, logs, strict=True)
        )
    elif name == "selection_on_gains":
        profiles = tuple(
            profile * (1.0 - 0.1 * index) for index, profile in enumerate(logs)
        )
    elif name == "novelty_effects":
        profiles = tuple(
            2.0 * np.exp(-0.3 * np.arange(length)) + 0.5 for length in lengths
        )
    else:
        profiles = (
            np.full(lengths[0], 2.5),
            logs[1],
            logs[2],
        )
    return MappingProxyType(
        {
            period: _freeze_profile(profile)
            for period, profile in zip(periods, profiles, strict=True)
        }
    )


def _temporal_outcome(
    context: SimulationContext,
    rng: np.random.Generator,
    *,
    config: FTestTemporalConfig,
) -> ComponentDraw:
    n_units = context.dimensions.n_units
    n_periods = context.dimensions.n_periods
    unit_effects = rng.normal(scale=config.unit_effect_scale, size=n_units)
    time_effects = rng.normal(scale=config.time_effect_scale, size=n_periods)
    unit_trends = rng.normal(scale=config.unit_trend_scale, size=n_units)
    residuals = np.zeros((n_units, n_periods), dtype=float)
    residuals[:, 0] = rng.normal(scale=config.noise_scale, size=n_units)
    innovations = rng.normal(size=(n_units, n_periods - 1))
    innovation_scale = config.noise_scale * np.sqrt(1.0 - config.error_rho**2)
    for period in range(1, n_periods):
        residuals[:, period] = (
            config.error_rho * residuals[:, period - 1]
            + innovation_scale * innovations[:, period - 1]
        )
    time = np.arange(n_periods, dtype=float)
    untreated = (
        unit_effects[:, None]
        + time_effects[None, :]
        + unit_trends[:, None] * time[None, :]
        + residuals
    )
    return ComponentDraw(
        untreated,
        {
            "kind": "ftest_temporal_outcome",
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "unit_trends": unit_trends,
            "residuals": residuals,
            "error_rho": config.error_rho,
        },
    )


def _cohort_outcome(
    context: SimulationContext,
    rng: np.random.Generator,
    *,
    config: FTestCohortConfig,
) -> ComponentDraw:
    n_units = context.dimensions.n_units
    n_periods = context.dimensions.n_periods
    unit_effects = rng.normal(scale=config.unit_effect_scale, size=n_units)

    periodic = np.resize(np.asarray(config.periodic_time_effects), n_periods)
    time_innovations = rng.normal(scale=config.time_effect_scale, size=n_periods)
    autoregressive_time = np.empty(n_periods, dtype=float)
    autoregressive_time[0] = time_innovations[0]
    for period in range(1, n_periods):
        autoregressive_time[period] = (
            config.time_error_rho * autoregressive_time[period - 1]
            + time_innovations[period]
        )
    time_effects = periodic + autoregressive_time - autoregressive_time.mean()

    innovations = rng.normal(scale=config.noise_scale, size=(n_units, n_periods))
    residuals = np.empty((n_units, n_periods), dtype=float)
    residuals[:, 0] = innovations[:, 0]
    for period in range(1, n_periods):
        residuals[:, period] = (
            config.unit_error_rho * residuals[:, period - 1] + innovations[:, period]
        )
    untreated = unit_effects[:, None] + time_effects[None, :] + residuals
    return ComponentDraw(
        untreated,
        {
            "kind": "ftest_cohort_outcome",
            "unit_effects": unit_effects,
            "time_effects": time_effects,
            "autoregressive_time_effects": autoregressive_time,
            "residuals": residuals,
            "unit_error_rho": config.unit_error_rho,
            "time_error_rho": config.time_error_rho,
        },
    )


def ftest_temporal_design(
    name: str,
    *,
    config: FTestTemporalConfig | None = None,
    profile_seed: int | np.random.SeedSequence | None = 42,
) -> PanelSimulator:
    """Build a reusable simulator for a single-cohort temporal-effect DGP."""

    resolved = FTestTemporalConfig() if config is None else config
    profile = ftest_temporal_profile(name, config=resolved, seed=profile_seed)
    return PanelSimulator(
        name=f"ftest_temporal_{name}",
        dimensions=resolved.dimensions,
        assignment=RandomizedSingleCohortAssignment(
            n_treated=resolved.treated_count,
            adoption_period=resolved.adoption_period,
        ),
        outcome_model=CallableOutcomeModel(
            lambda context, rng: _temporal_outcome(context, rng, config=resolved),
            name="ftest_temporal_outcome",
        ),
        effect_model=CohortEventTimeEffect({resolved.adoption_period: profile}),
    )


def ftest_cohort_design(
    name: str,
    *,
    config: FTestCohortConfig | None = None,
) -> PanelSimulator:
    """Build a reusable simulator for a three-cohort heterogeneity DGP."""

    resolved = FTestCohortConfig() if config is None else config
    profiles = ftest_cohort_profiles(name, config=resolved)
    return PanelSimulator(
        name=f"ftest_cohort_{name}",
        dimensions=resolved.dimensions,
        assignment=RandomizedStaggeredAdoption(
            adoption_periods=resolved.adoption_periods,
            cohort_sizes=resolved.cohort_sizes,
        ),
        outcome_model=CallableOutcomeModel(
            lambda context, rng: _cohort_outcome(context, rng, config=resolved),
            name="ftest_cohort_outcome",
        ),
        effect_model=CohortEventTimeEffect(profiles),
    )


def ftest_temporal(
    name: str,
    *,
    config: FTestTemporalConfig | None = None,
    profile_seed: int | np.random.SeedSequence | None = 42,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one panel from a single-cohort temporal-effect DGP."""

    return ftest_temporal_design(
        name, config=config, profile_seed=profile_seed
    ).simulate(seed=seed, rng=rng)


def ftest_cohort(
    name: str,
    *,
    config: FTestCohortConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one panel from a three-cohort heterogeneity DGP."""

    return ftest_cohort_design(name, config=config).simulate(seed=seed, rng=rng)


__all__ = [
    "FTestCohortConfig",
    "FTestTemporalConfig",
    "available_ftest_cohort_designs",
    "available_ftest_temporal_designs",
    "ftest_cohort",
    "ftest_cohort_design",
    "ftest_cohort_profiles",
    "ftest_temporal",
    "ftest_temporal_design",
    "ftest_temporal_profile",
]
