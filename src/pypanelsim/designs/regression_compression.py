"""Designs adapted from the regression-compression projects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..components import (
    AdditiveFactorOutcome,
    BinaryLogitAssignment,
    CohortEventTimeEffect,
    ConstantEffect,
    PanelDimensions,
    RandomizedSingleCohortAssignment,
    RandomizedStaggeredAdoption,
    SumOutcomeModel,
)
from ..outcomes import ARMAErrorOutcome, LowRankFactorOutcome, UnitTrendOutcome
from ..profiles import EventTimeProfileEffect, RandomUnitEffect, RandomWalkEffect
from ..simulator import PanelSimulator

_ANSCOMBE_DESIGNS = MappingProxyType(
    {
        "zero": "Zero effect with one adoption cohort",
        "unit": "Zero-mean unit heterogeneity with one cohort",
        "time": "Zero-mean event-time heterogeneity with one cohort",
        "cohort_time": "Zero-mean cohort and event-time heterogeneity",
    }
)


def available_anscombe_designs() -> tuple[str, ...]:
    """Return the four longitudinal Anscombe design names."""

    return tuple(_ANSCOMBE_DESIGNS)


@dataclass(frozen=True, slots=True)
class RegressionCompressionConfig:
    """Parameters for the large additive, trend, and AR(1) panel law."""

    n_units: int = 1_000
    n_periods: int = 35
    adoption_period: int = 15
    n_treated: int | None = None
    effect_profile: str = "concave"
    max_effect: float = 1.0
    heterogeneous_effects: bool = False
    unit_effect_scale: float = 5.0
    time_effect_scale: float = 2.0
    unit_trend_scale: float = 0.01
    error_scale: float = 2.0
    error_rho: float = 0.7

    def __post_init__(self) -> None:
        if self.n_units <= 1 or self.n_periods <= 1:
            raise ValueError("panel dimensions must exceed one")
        if not 0 <= self.adoption_period < self.n_periods:
            raise ValueError("adoption_period must lie inside the panel")
        if self.n_treated is not None and not 0 < self.n_treated < self.n_units:
            raise ValueError("n_treated must lie between zero and n_units")
        if self.effect_profile not in {
            "constant",
            "linear",
            "concave",
            "positive_then_negative",
            "exponential",
            "sinusoidal",
            "random_walk",
        }:
            raise ValueError("unknown regression-compression effect profile")
        for name in (
            "unit_effect_scale",
            "time_effect_scale",
            "unit_trend_scale",
            "error_scale",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.error_rho) or abs(self.error_rho) >= 1.0:
            raise ValueError("error_rho must lie strictly between -1 and 1")
        if not np.isfinite(self.max_effect):
            raise ValueError("max_effect must be finite")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, self.n_periods)


@dataclass(frozen=True, slots=True)
class AnscombePanelConfig:
    """Parameters for the longitudinal Anscombe quartet."""

    n_units: int = 1_000
    n_periods: int = 20
    first_adoption: int = 10
    second_adoption: int = 15
    rank: int = 3
    noise_scale: float = 0.1
    unit_trend_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.n_units < 4 or self.n_units % 4:
            raise ValueError("n_units must be a multiple of four")
        if not 0 <= self.first_adoption < self.second_adoption < self.n_periods:
            raise ValueError("adoption periods must be ordered inside the panel")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        for name in ("noise_scale", "unit_trend_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, self.n_periods)


def _effect_profile(config: RegressionCompressionConfig):
    length = config.n_periods - config.adoption_period
    maximum = config.max_effect
    name = config.effect_profile
    if name == "random_walk":
        return RandomWalkEffect(maximum)

    def profile(event_time: np.ndarray) -> np.ndarray:
        event = np.asarray(event_time, dtype=int)
        if name == "constant":
            path = np.full(length, maximum)
        elif name == "linear":
            path = np.linspace(0.0, maximum, length)
        elif name == "concave":
            position = np.arange(1, length + 1)
            path = maximum * 0.5 * np.log(2.0 * position / length + 1.0)
        elif name == "positive_then_negative":
            midpoint = length // 2
            path = np.concatenate(
                (
                    np.linspace(0.0, maximum, midpoint),
                    np.linspace(maximum, -maximum, length - midpoint),
                )
            )
        elif name == "exponential":
            path = maximum * (1.0 - np.exp(-np.linspace(0.0, 5.0, length)))
        else:
            path = maximum * np.sin(np.linspace(0.0, 2.0 * np.pi, length))
        return path[np.minimum(event, length - 1)]

    bounds = (0.5, 1.5) if config.heterogeneous_effects else None
    return EventTimeProfileEffect(
        profile,
        unit_multiplier_bounds=bounds,
        name=f"regression_compression_{name}",
    )


def regression_compression_design(
    *, config: RegressionCompressionConfig | None = None
) -> PanelSimulator:
    """Build the large-panel TWFE compression benchmark."""

    resolved = RegressionCompressionConfig() if config is None else config
    if resolved.n_treated is None:
        assignment = BinaryLogitAssignment(
            adoption_period=resolved.adoption_period,
            intercept=0.0,
            observable_coefficients=(),
            unobservable_coefficients=(),
        )
    else:
        assignment = RandomizedSingleCohortAssignment(
            resolved.n_treated, resolved.adoption_period
        )
    innovation_scale = resolved.error_scale * np.sqrt(1.0 - resolved.error_rho**2)
    outcome = SumOutcomeModel(
        (
            AdditiveFactorOutcome(
                resolved.unit_effect_scale,
                resolved.time_effect_scale,
                0.0,
            ),
            UnitTrendOutcome(resolved.unit_trend_scale),
            ARMAErrorOutcome(
                (resolved.error_rho,),
                innovation_scale=innovation_scale,
            ),
        )
    )
    return PanelSimulator(
        name=f"regression_compression_{resolved.effect_profile}",
        dimensions=resolved.dimensions,
        assignment=assignment,
        outcome_model=outcome,
        effect_model=_effect_profile(resolved),
    )


def _indexed_path(path: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def profile(event_time: np.ndarray) -> np.ndarray:
        event = np.asarray(event_time, dtype=int)
        return path[np.minimum(event, path.size - 1)]

    return profile


def anscombe_design(
    name: str,
    *,
    config: AnscombePanelConfig | None = None,
) -> PanelSimulator:
    """Build one member of the longitudinal Anscombe quartet."""

    if name not in _ANSCOMBE_DESIGNS:
        available = ", ".join(_ANSCOMBE_DESIGNS)
        raise KeyError(f"unknown Anscombe design {name!r}; available: {available}")
    resolved = AnscombePanelConfig() if config is None else config
    outcome = SumOutcomeModel(
        (
            LowRankFactorOutcome(
                rank=resolved.rank,
                loading_scale=0.2,
                factor_scale=0.1,
                noise_scale=resolved.noise_scale,
                distribution="uniform",
            ),
            UnitTrendOutcome(resolved.unit_trend_scale),
        )
    )
    if name == "cohort_time":
        assignment = RandomizedStaggeredAdoption(
            (resolved.first_adoption, resolved.second_adoption),
            (resolved.n_units // 4, resolved.n_units // 4),
        )
        profiles = {
            resolved.first_adoption: np.linspace(
                -0.25, 0.25, resolved.n_periods - resolved.first_adoption
            ),
            resolved.second_adoption: np.linspace(
                -0.75, 0.75, resolved.n_periods - resolved.second_adoption
            ),
        }
        effect = CohortEventTimeEffect(profiles)
    else:
        assignment = RandomizedSingleCohortAssignment(
            resolved.n_units // 2, resolved.first_adoption
        )
        if name == "zero":
            effect = ConstantEffect(0.0)
        elif name == "unit":
            effect = RandomUnitEffect(0.0, 0.25)
        else:
            path = np.linspace(
                -0.5, 0.5, resolved.n_periods - resolved.first_adoption
            )
            effect = EventTimeProfileEffect(
                _indexed_path(path), name="anscombe_time"
            )
    return PanelSimulator(
        name=f"regression_compression_anscombe_{name}",
        dimensions=resolved.dimensions,
        assignment=assignment,
        outcome_model=outcome,
        effect_model=effect,
    )
