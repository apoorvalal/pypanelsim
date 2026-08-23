"""Reusable event-time effect profiles and adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .components import ComponentDraw, SimulationContext

ProfileCallable = Callable[[np.ndarray], np.ndarray]


def constant_profile(event_time: np.ndarray, *, value: float = 1.0) -> np.ndarray:
    """Return a constant post-treatment profile."""

    return np.full(np.asarray(event_time).shape, value, dtype=float)


def linear_profile(event_time: np.ndarray, *, slope: float = 0.2) -> np.ndarray:
    """Return a linear event-time profile starting at one slope unit."""

    return slope * (np.asarray(event_time, dtype=float) + 1.0)


def concave_profile(event_time: np.ndarray, *, height: float = 1.0) -> np.ndarray:
    """Return a normalized logarithmic growth profile."""

    event = np.asarray(event_time, dtype=float)
    return height * np.log1p(event) / np.log1p(max(float(event.max()), 1.0))


def saturating_profile(
    event_time: np.ndarray,
    *,
    height: float = 1.0,
    speed: float = 0.25,
    floor: float = 0.0,
) -> np.ndarray:
    """Return an exponentially saturating post-treatment profile."""

    event = np.asarray(event_time, dtype=float)
    return floor + height * (1.0 - np.exp(-speed * event))


def hump_profile(
    event_time: np.ndarray,
    *,
    height: float = 1.0,
    center: float = 4.0,
    width: float = 2.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Return a Gaussian-shaped response over event time."""

    event = np.asarray(event_time, dtype=float)
    return floor + height * np.exp(-0.5 * ((event - center) / width) ** 2)


def sinusoidal_profile(
    event_time: np.ndarray,
    *,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Return a sinusoidal event-time response."""

    event = np.asarray(event_time, dtype=float)
    return offset + amplitude * np.sin(frequency * event + phase)


@dataclass(frozen=True, slots=True)
class EventTimeProfileEffect:
    """Apply a callable profile after adoption with optional unit variation."""

    profile: ProfileCallable
    unit_multiplier_bounds: tuple[float, float] | None = None
    name: str = "event_time_profile"

    def __post_init__(self) -> None:
        if not callable(self.profile):
            raise TypeError("profile must be callable")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if self.unit_multiplier_bounds is not None:
            lower, upper = self.unit_multiplier_bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("unit_multiplier_bounds must be finite and ordered")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        adoption = context.adoption_times
        periods = np.arange(context.dimensions.n_periods)
        event_time = periods[None, :] - adoption[:, None]
        post_event = np.maximum(event_time, 0)
        base = np.asarray(self.profile(post_event), dtype=float)
        try:
            surface = np.broadcast_to(
                base,
                (context.dimensions.n_units, context.dimensions.n_periods),
            ).astype(float, copy=True)
        except ValueError as error:
            raise ValueError(
                "profile must return a broadcastable effect surface"
            ) from error
        if not np.all(np.isfinite(surface)):
            raise ValueError("profile must return only finite values")
        if self.unit_multiplier_bounds is None:
            multipliers = np.ones(context.dimensions.n_units)
        else:
            lower, upper = self.unit_multiplier_bounds
            multipliers = rng.uniform(lower, upper, context.dimensions.n_units)
        surface *= multipliers[:, None]
        return ComponentDraw(
            surface * context.treatment,
            {
                "kind": self.name,
                "effect_surface": surface,
                "unit_multipliers": multipliers,
            },
        )


@dataclass(frozen=True, slots=True)
class RandomUnitEffect:
    """Draw a time-constant treatment effect independently for each unit."""

    mean: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("scale must be finite and nonnegative")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        unit_effects = rng.normal(
            self.mean, self.scale, size=context.dimensions.n_units
        )
        surface = np.repeat(
            unit_effects[:, None], context.dimensions.n_periods, axis=1
        )
        return ComponentDraw(
            surface * context.treatment,
            {
                "kind": "random_unit_effect",
                "unit_effects": unit_effects,
                "effect_surface": surface,
            },
        )


@dataclass(frozen=True, slots=True)
class RandomWalkEffect:
    """Draw one common random-walk profile over event time."""

    innovation_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.innovation_scale) or self.innovation_scale < 0.0:
            raise ValueError("innovation_scale must be finite and nonnegative")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        if not context.is_absorbing:
            raise ValueError("RandomWalkEffect requires absorbing treatment")
        adoption = context.adoption_times
        first_adoption = int(adoption[context.ever_treated].min())
        max_length = context.dimensions.n_periods - first_adoption
        profile = np.cumsum(rng.normal(scale=self.innovation_scale, size=max_length))
        surface = np.zeros(
            (context.dimensions.n_units, context.dimensions.n_periods)
        )
        for unit in context.treated_units:
            start = int(adoption[unit])
            surface[unit, start:] = profile[: context.dimensions.n_periods - start]
        return ComponentDraw(
            surface * context.treatment,
            {
                "kind": "random_walk",
                "profile": profile,
                "effect_surface": surface,
            },
        )
