"""Many-cohort designs from the FTestEventStudy Lepskii draft."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..components import (
    CohortEventTimeEffect,
    ComponentDraw,
    PanelDimensions,
    RandomizedStaggeredAdoption,
    SimulationContext,
)
from ..data import PanelDataset
from ..simulator import PanelSimulator, SimulationSeeds

_LABELS = MappingProxyType(
    {
        "complete_homogeneity": "Complete homogeneity",
        "common_path": "Common path",
        "cohort_only_blocks": "Cohort-only blocks",
        "time_only_blocks": "Time-only blocks",
        "tensor_blocks": "Tensor blocks",
        "early_late_split": "Early versus late",
        "three_epoch": "Three epochs",
        "smooth_dose_gradient": "Smooth dose gradient",
        "response_timing": "Response timing",
        "local_window_shock": "Local window shock",
        "log_shark_sin": "Log versus shark versus sine",
        "no_pool_unique_signatures": "No pooling: signatures",
        "no_pool_unique_peaks": "No pooling: peak timing",
        "no_pool_signed_paths": "No pooling: signed paths",
    }
)


def available_lepskii_designs() -> tuple[str, ...]:
    """Return all fourteen designs from the many-cohort draft."""

    return tuple(_LABELS)


@dataclass(frozen=True, slots=True)
class LepskiiPanelConfig:
    """Panel dimensions and assignment shares for the Lepskii draft."""

    n_units: int = 1_200
    n_periods: int = 36
    adoption_periods: tuple[int, ...] = (8, 10, 12, 14, 16, 18, 20, 22, 24)
    cohort_shares: tuple[float, ...] = (
        0.05,
        0.07,
        0.09,
        0.12,
        0.14,
        0.13,
        0.10,
        0.07,
        0.05,
    )
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
        if self.n_units <= 1 or self.n_periods <= 1:
            raise ValueError("panel dimensions must exceed one")
        if not self.adoption_periods or len(self.adoption_periods) != len(
            self.cohort_shares
        ):
            raise ValueError("adoption_periods and cohort_shares must align")
        if tuple(sorted(set(self.adoption_periods))) != self.adoption_periods:
            raise ValueError("adoption_periods must be unique and increasing")
        if any(
            period < 0 or period >= self.n_periods for period in self.adoption_periods
        ):
            raise ValueError("adoption_periods must lie inside the panel")
        if any(share <= 0.0 for share in self.cohort_shares):
            raise ValueError("cohort_shares must be positive")
        if sum(self.cohort_shares) > 1.0:
            raise ValueError("cohort_shares must sum to at most one")
        for name in ("unit_effect_scale", "time_effect_scale", "noise_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("unit_error_rho", "time_error_rho"):
            value = getattr(self, name)
            if not np.isfinite(value) or abs(value) >= 1.0:
                raise ValueError(f"{name} must lie strictly between -1 and 1")
        if not self.periodic_time_effects or not np.all(
            np.isfinite(self.periodic_time_effects)
        ):
            raise ValueError("periodic_time_effects must be nonempty and finite")
        if any(size <= 0 for size in self.cohort_sizes):
            raise ValueError("n_units is too small for the configured shares")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, self.n_periods)

    @property
    def cohort_sizes(self) -> tuple[int, ...]:
        return tuple(int(self.n_units * share) for share in self.cohort_shares)


def _constant(length: int, level: float) -> np.ndarray:
    return np.full(length, level, dtype=float)


def _common(length: int, height: float = 1.6) -> np.ndarray:
    event = np.arange(length)
    return 0.15 + height * (1.0 - np.exp(-0.22 * event))


def _ramp(length: int, start: float, height: float) -> np.ndarray:
    return np.linspace(start, height, length)


def _front_loaded(
    length: int, height: float, decay: float = 0.32, floor: float = 0.15
) -> np.ndarray:
    event = np.arange(length)
    return height * np.exp(-decay * event) + floor


def _delayed(
    length: int, height: float, delay: int = 2, speed: float = 0.28
) -> np.ndarray:
    event = np.arange(length)
    return height * (1.0 - np.exp(-speed * np.maximum(event - delay, 0)))


def _hump(
    length: int,
    height: float,
    center: float = 5.0,
    width: float = 2.5,
    floor: float = 0.05,
) -> np.ndarray:
    event = np.arange(length)
    return floor + height * np.exp(-0.5 * ((event - center) / width) ** 2)


def _persistent(length: int, height: float) -> np.ndarray:
    event = np.arange(length)
    return height * (1.0 - np.exp(-0.5 * event))


def _rebound(length: int, height: float) -> np.ndarray:
    event = np.arange(length)
    return (
        0.35
        + height * (1.0 - np.exp(-0.18 * event))
        - 0.9 * np.exp(-0.5 * ((event - 2.5) / 1.6) ** 2)
    )


def _step(length: int, low: float, high: float, step_time: int) -> np.ndarray:
    return np.where(np.arange(length) < step_time, low, high)


def _tensor(length: int, early: bool) -> np.ndarray:
    return np.where(np.arange(length) <= 3, 0.45, 2.5 if early else -0.55)


def _damped_wave(
    length: int, amplitude: float, phase: float, trend: float
) -> np.ndarray:
    event = np.arange(length)
    return trend * (1.0 - np.exp(-0.16 * event)) + amplitude * np.exp(
        -0.08 * event
    ) * np.sin(1.15 * event + phase)


def _signature(length: int, index: int) -> np.ndarray:
    event = np.arange(length)
    if index == 0:
        return _front_loaded(length, 2.9, decay=0.42, floor=0.05)
    if index == 1:
        return _delayed(length, 2.6, delay=5, speed=0.5)
    if index == 2:
        return _hump(length, 2.9, center=3, width=1.2, floor=0.0)
    if index == 3:
        return _step(length, 0.15, 2.6, 4)
    if index == 4:
        return 2.2 - 0.16 * event
    if index == 5:
        return _damped_wave(length, 1.6, 0.2, 1.2)
    if index == 6:
        return -0.8 * np.exp(-0.5 * ((event - 2.0) / 1.3) ** 2) + _common(length, 1.7)
    if index == 7:
        return _ramp(length, -0.35, 2.4)
    return _hump(length, 1.5, center=8, width=3.2, floor=0.2) - 0.5


def _unique_peak(length: int, index: int) -> np.ndarray:
    centers = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    heights = (2.4, 1.6, 2.9, 1.9, 3.1, 1.5, 2.6, 1.8, 2.2)
    widths = (0.9, 1.5, 1.1, 2.2, 1.6, 2.8, 1.3, 2.5, 1.8)
    floors = (0.0, 0.35, -0.2, 0.15, 0.4, -0.1, 0.25, 0.05, 0.3)
    event = np.arange(length)
    return floors[index] + heights[index] * np.exp(
        -0.5 * ((event - centers[index]) / widths[index]) ** 2
    )


def _unique_signed(length: int, index: int) -> np.ndarray:
    event = np.arange(length)
    curves = (
        1.8 * (1.0 - np.exp(-0.25 * event)),
        -1.2 * (1.0 - np.exp(-0.35 * event)),
        2.0 * np.exp(-0.5 * ((event - 2.0) / 1.2) ** 2) - 0.25 * event / max(length, 1),
        -1.5 * np.exp(-0.5 * ((event - 3.0) / 1.5) ** 2)
        + 1.3 * (1.0 - np.exp(-0.18 * event)),
        0.9 + 0.8 * np.sin(0.8 * event),
        1.9 - 0.22 * event,
        -0.4 + 0.18 * event,
        1.4 * np.sign(np.sin(0.65 * event + 0.3)),
        0.5 + 1.6 * np.exp(-0.11 * event) * np.cos(0.75 * event),
    )
    return curves[index]


def _profile_builders(
    config: LepskiiPanelConfig,
) -> Mapping[str, Callable[[int, int], np.ndarray]]:
    midpoint = (len(config.adoption_periods) - 1) / 2
    return {
        "complete_homogeneity": lambda length, index: _constant(length, 1.4),
        "common_path": lambda length, index: _common(length, 1.55),
        "cohort_only_blocks": lambda length, index: _constant(
            length, 0.45 if index < 3 else 1.45 if index < 5 else 2.65
        ),
        "time_only_blocks": lambda length, index: np.where(
            np.arange(length) <= 6, 0.35, 2.1
        ),
        "tensor_blocks": lambda length, index: _tensor(length, index < 5),
        "early_late_split": lambda length, index: (
            _front_loaded(length, 2.6, decay=0.24, floor=0.35)
            if index < 4
            else _delayed(length, 2.4, delay=2, speed=0.24)
        ),
        "three_epoch": lambda length, index: (
            _ramp(length, 0.2, 2.7)
            if index < 3
            else _persistent(length, 1.7)
            if index < 6
            else _hump(length, 2.4, center=4, width=2.1, floor=0.1)
        ),
        "smooth_dose_gradient": lambda length, index: (
            _common(length, 1.9) * (0.55 + 0.17 * index)
        ),
        "response_timing": lambda length, index: (
            _persistent(length, 1.9)
            if index < 3
            else _delayed(length, 1.9, delay=4, speed=0.36)
            if index < 6
            else _rebound(length, 1.75)
        ),
        "local_window_shock": lambda length, index: (
            _common(length, 1.2)
            if abs(index - midpoint) > 2
            else _common(length, 1.45)
            if abs(index - midpoint) == 2
            else _constant(length, 2.9)
        ),
        "log_shark_sin": lambda length, index: _log_shark_sin(length, index),
        "no_pool_unique_signatures": _signature,
        "no_pool_unique_peaks": _unique_peak,
        "no_pool_signed_paths": _unique_signed,
    }


def _log_shark_sin(length: int, index: int) -> np.ndarray:
    event = np.arange(length)
    if index < 3:
        height = 2.1 + 0.25 * index
        peak = 2 + index
        rise = height * np.minimum(event / max(peak, 1), 1.0)
        decline_span = max((length - 1) - peak, 1)
        decline = height - height * np.maximum(event - peak, 0) / decline_span
        return np.where(event <= peak, rise, decline)
    if index < 6:
        raw = np.log(np.arange(1, length + 1))
        return (1.7 + 0.25 * (index - 3)) * raw / raw.max()
    return 0.35 + (0.9 + 0.15 * (index - 6)) * np.sin(0.9 * event + 0.7 * index)


def lepskii_profiles(
    name: str, *, config: LepskiiPanelConfig | None = None
) -> Mapping[int, np.ndarray]:
    """Return full post-adoption paths for one Lepskii design."""

    if name not in _LABELS:
        available = ", ".join(_LABELS)
        raise KeyError(f"unknown Lepskii design {name!r}; available: {available}")
    resolved = LepskiiPanelConfig() if config is None else config
    builder = _profile_builders(resolved)[name]
    profiles: dict[int, np.ndarray] = {}
    for index, period in enumerate(resolved.adoption_periods):
        profile = np.asarray(builder(resolved.n_periods - period, index), dtype=float)
        profile.setflags(write=False)
        profiles[period] = profile
    return MappingProxyType(profiles)


@dataclass(frozen=True, slots=True)
class _LepskiiOutcome:
    config: LepskiiPanelConfig

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        n_units = context.dimensions.n_units
        n_periods = context.dimensions.n_periods
        unit_effects = rng.normal(scale=self.config.unit_effect_scale, size=n_units)
        periodic = np.resize(
            np.asarray(self.config.periodic_time_effects, dtype=float), n_periods
        )
        time_innovations = rng.normal(
            scale=self.config.time_effect_scale, size=n_periods
        )
        autoregressive_time = np.empty(n_periods)
        autoregressive_time[0] = time_innovations[0]
        for period in range(1, n_periods):
            autoregressive_time[period] = (
                self.config.time_error_rho * autoregressive_time[period - 1]
                + time_innovations[period]
            )
        time_effects = periodic + autoregressive_time - autoregressive_time.mean()
        innovations = rng.normal(
            scale=self.config.noise_scale, size=(n_units, n_periods)
        )
        residuals = np.empty_like(innovations)
        residuals[:, 0] = innovations[:, 0]
        for period in range(1, n_periods):
            residuals[:, period] = (
                self.config.unit_error_rho * residuals[:, period - 1]
                + innovations[:, period]
            )
        values = unit_effects[:, None] + time_effects[None, :] + residuals
        return ComponentDraw(
            values,
            {
                "kind": "lepskii_outcome",
                "unit_effects": unit_effects,
                "time_effects": time_effects,
                "residuals": residuals,
            },
        )


def lepskii_design(
    name: str, *, config: LepskiiPanelConfig | None = None
) -> PanelSimulator:
    """Build one many-cohort Lepskii-draft simulator."""

    resolved = LepskiiPanelConfig() if config is None else config
    return PanelSimulator(
        name=f"lepskii_{name}",
        dimensions=resolved.dimensions,
        assignment=RandomizedStaggeredAdoption(
            resolved.adoption_periods, resolved.cohort_sizes
        ),
        outcome_model=_LepskiiOutcome(resolved),
        effect_model=CohortEventTimeEffect(lepskii_profiles(name, config=resolved)),
    )


def lepskii(
    name: str,
    *,
    config: LepskiiPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
    streams: SimulationSeeds | None = None,
) -> PanelDataset:
    """Draw one panel from a Lepskii-draft design."""

    return lepskii_design(name, config=config).simulate(
        seed=seed, rng=rng, streams=streams
    )
