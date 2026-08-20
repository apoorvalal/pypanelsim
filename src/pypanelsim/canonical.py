"""Canonical panel designs from the augmented-balancing experiments."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .components import (
    ComponentDraw,
    LinearRampEffect,
    OutcomeModel,
    PanelDimensions,
    SimulationContext,
    SingleCohortAssignment,
)
from .data import FloatMatrix, PanelDataset
from .registry import DGPRegistry
from .simulator import PanelSimulator


@dataclass(frozen=True, slots=True)
class CanonicalPanelConfig:
    """Dimensions and shared parameters for the canonical experiment designs."""

    n_control: int = 160
    n_treated: int = 40
    n_pre: int = 40
    n_post: int = 10
    effect_slope: float = 0.2
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        for name in ("n_control", "n_treated", "n_pre", "n_post"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.effect_slope):
            raise ValueError("effect_slope must be finite")
        if not np.isfinite(self.noise_variance) or self.noise_variance < 0.0:
            raise ValueError("noise_variance must be finite and nonnegative")

    @property
    def n_units(self) -> int:
        """Return the total number of units."""

        return self.n_control + self.n_treated

    @property
    def n_periods(self) -> int:
        """Return the total number of periods."""

        return self.n_pre + self.n_post

    @property
    def dimensions(self) -> PanelDimensions:
        """Return generic panel dimensions for the simulation pipeline."""

        return PanelDimensions(self.n_units, self.n_periods)


def _validate_overlap(overlap: float) -> None:
    if not np.isfinite(overlap) or overlap < 0.0:
        raise ValueError("overlap must be finite and nonnegative")


def _validate_noise_variance(noise_variance: float) -> None:
    if not np.isfinite(noise_variance) or noise_variance < 0.0:
        raise ValueError("noise_variance must be finite and nonnegative")


def _stationary_ar1(
    rng: np.random.Generator,
    n_periods: int,
    coefficient: float = 0.5,
    *,
    innovation_mean: float = 0.0,
    innovation_scale: float = 1.0,
    discard: int = 0,
) -> FloatMatrix:
    if n_periods <= 0 or discard < 0:
        raise ValueError("n_periods must be positive and discard must be nonnegative")
    if not np.isfinite(coefficient) or abs(coefficient) >= 1.0:
        raise ValueError("coefficient must lie strictly between -1 and 1")
    if not np.isfinite(innovation_scale) or innovation_scale < 0.0:
        raise ValueError("innovation_scale must be finite and nonnegative")
    total = n_periods + discard
    long_run_mean = innovation_mean / (1.0 - coefficient)
    long_run_scale = innovation_scale / np.sqrt(1.0 - coefficient**2)
    path = np.empty(total, dtype=float)
    path[0] = rng.normal(long_run_mean, long_run_scale)
    innovations = rng.normal(innovation_mean, innovation_scale, size=total - 1)
    for period in range(1, total):
        path[period] = coefficient * path[period - 1] + innovations[period - 1]
    return path[discard:]


def _factor_path(
    rng: np.random.Generator,
    n_periods: int,
    factor_type: str,
    *,
    size: float = 1.0,
) -> FloatMatrix:
    time = np.arange(1, n_periods + 1, dtype=float)
    if factor_type == "ar1":
        raw = _stationary_ar1(rng, n_periods, discard=20)
    elif factor_type == "drift":
        raw = _stationary_ar1(rng, n_periods, discard=20) + 0.5 * time
    elif factor_type == "trend":
        raw = time + rng.normal(size=n_periods)
    elif factor_type == "cyclical":
        raw = np.sin(time * np.pi / 15.0)
    elif factor_type in {"white_noise", "noise"}:
        raw = rng.normal(size=n_periods)
    else:
        raise ValueError(f"unknown factor_type: {factor_type!r}")
    scale = raw.std(ddof=1)
    if scale == 0.0:
        raise ValueError("factor path has zero sample standard deviation")
    return size * raw / scale


def _factor_components(
    rng: np.random.Generator,
    context: SimulationContext,
    *,
    factor_types: Sequence[str],
    control_loading_mean: float,
    treated_loading_mean: float,
    factor_size: float | Sequence[float] = 1.0,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    n_factors = len(factor_types)
    if n_factors == 0:
        raise ValueError("factor_types must contain at least one factor")
    sizes = np.broadcast_to(np.asarray(factor_size, dtype=float), (n_factors,))
    controls = context.control_units
    treated = context.treated_units
    control_loadings = np.empty((controls.size, n_factors), dtype=float)
    treated_loadings = np.empty((treated.size, n_factors), dtype=float)
    for factor in range(n_factors):
        control_loadings[:, factor] = rng.normal(
            control_loading_mean, 1.0, controls.size
        )
        treated_loadings[:, factor] = rng.normal(
            treated_loading_mean, 1.0, treated.size
        )
    loadings = np.empty((context.dimensions.n_units, n_factors), dtype=float)
    loadings[controls] = control_loadings
    loadings[treated] = treated_loadings
    factors = np.vstack(
        [
            _factor_path(rng, context.dimensions.n_periods, factor_type, size=sizes[i])
            for i, factor_type in enumerate(factor_types)
        ]
    )
    return loadings, factors, loadings @ factors


def _error(
    rng: np.random.Generator, n_units: int, n_periods: int, scale: float
) -> FloatMatrix:
    return rng.normal(0.0, scale, size=(n_units, n_periods))


@dataclass(frozen=True, slots=True)
class ClassicFactorOutcome:
    """Two drift factors with a configurable treated-control loading gap."""

    overlap: float = 0.0
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        _validate_overlap(self.overlap)
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        loadings, factors, signal = _factor_components(
            rng,
            context,
            factor_types=("drift", "drift"),
            control_loading_mean=-self.overlap,
            treated_loading_mean=self.overlap,
        )
        untreated = signal + _error(
            rng,
            context.dimensions.n_units,
            context.dimensions.n_periods,
            np.sqrt(self.noise_variance),
        )
        return ComponentDraw(
            untreated,
            {"overlap": self.overlap, "loadings": loadings, "factors": factors},
        )


@dataclass(frozen=True, slots=True)
class WeakFactorOutcome:
    """Five weak drift factors and five weak cyclical factors."""

    overlap: float = 0.0
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        _validate_overlap(self.overlap)
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        loadings, factors, signal = _factor_components(
            rng,
            context,
            factor_types=("drift",) * 5 + ("cyclical",) * 5,
            control_loading_mean=-self.overlap,
            treated_loading_mean=self.overlap,
            factor_size=0.2,
        )
        untreated = signal + _error(
            rng,
            context.dimensions.n_units,
            context.dimensions.n_periods,
            np.sqrt(self.noise_variance),
        )
        return ComponentDraw(
            untreated,
            {"overlap": self.overlap, "loadings": loadings, "factors": factors},
        )


def _sample_active_donors(
    rng: np.random.Generator,
    control_loadings: FloatMatrix,
    n_active: int,
) -> np.ndarray:
    scores = control_loadings.sum(axis=1)
    favored = np.argsort(scores)[-n_active:]
    probabilities = np.full(control_loadings.shape[0], 0.1, dtype=float)
    probabilities[favored] = 1.0
    probabilities /= probabilities.sum()
    return np.sort(
        rng.choice(control_loadings.shape[0], n_active, replace=False, p=probabilities)
    )


def _synthetic_component(
    rng: np.random.Generator,
    context: SimulationContext,
    factor_signal: FloatMatrix,
    loadings: FloatMatrix,
    *,
    n_active: int,
    noise_scale: float,
) -> tuple[FloatMatrix, np.ndarray, FloatMatrix, FloatMatrix]:
    controls = context.control_units
    treated = context.treated_units
    active_local = _sample_active_donors(rng, loadings[controls], n_active)
    control_weights = np.zeros(controls.size, dtype=float)
    control_weights[active_local] = 1.0 / n_active
    donor_signal = control_weights @ factor_signal[controls]
    treated_signal = np.repeat(donor_signal[None, :], treated.size, axis=0)
    treated_signal += rng.normal(0.0, noise_scale, treated_signal.shape)
    component = np.empty_like(factor_signal)
    component[controls] = factor_signal[controls]
    component[treated] = treated_signal
    donor_weights = np.zeros(context.dimensions.n_units, dtype=float)
    donor_weights[controls] = control_weights
    return component, controls[active_local], donor_weights, control_weights


@dataclass(frozen=True, slots=True)
class SyntheticControlOutcome:
    """Sparse-donor synthetic-control outcome process."""

    active_share: float = 0.1
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        if not np.isfinite(self.active_share) or not 0.0 < self.active_share <= 1.0:
            raise ValueError("active_share must lie in (0, 1]")
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        loadings, factors, factor_signal = _factor_components(
            rng,
            context,
            factor_types=("drift", "drift"),
            control_loading_mean=-0.5,
            treated_loading_mean=0.5,
        )
        n_control = context.control_units.size
        n_active = max(1, int(np.ceil(self.active_share * n_control)))
        synthetic, active, donor_weights, control_weights = _synthetic_component(
            rng,
            context,
            factor_signal,
            loadings,
            n_active=n_active,
            noise_scale=1.0,
        )
        factor_weight = 1e-9 / (1.0 + 1e-9)
        synthetic_weight = 1.0 / (1.0 + 1e-9)
        untreated = factor_weight * factor_signal + synthetic_weight * synthetic
        untreated += _error(
            rng,
            context.dimensions.n_units,
            context.dimensions.n_periods,
            np.sqrt(self.noise_variance),
        )
        return ComponentDraw(
            untreated,
            {
                "active_share": self.active_share,
                "active_donors": active,
                "donor_weights": donor_weights,
                "control_donor_weights": control_weights,
                "loadings": loadings,
                "factors": factors,
            },
        )


@dataclass(frozen=True, slots=True)
class FactorSyntheticOutcome:
    """Equal mixture of factor and sparse-donor outcome components."""

    overlap: float = 0.0
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        _validate_overlap(self.overlap)
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        loadings, factors, factor_signal = _factor_components(
            rng,
            context,
            factor_types=("drift", "cyclical"),
            control_loading_mean=-self.overlap,
            treated_loading_mean=self.overlap,
        )
        n_active = min(context.control_units.size, context.treated_units.size)
        synthetic, active, donor_weights, control_weights = _synthetic_component(
            rng,
            context,
            factor_signal,
            loadings,
            n_active=n_active,
            noise_scale=2.0,
        )
        untreated = 0.5 * factor_signal + 0.5 * synthetic
        untreated += _error(
            rng,
            context.dimensions.n_units,
            context.dimensions.n_periods,
            np.sqrt(self.noise_variance),
        )
        return ComponentDraw(
            untreated,
            {
                "overlap": self.overlap,
                "active_donors": active,
                "donor_weights": donor_weights,
                "control_donor_weights": control_weights,
                "loadings": loadings,
                "factors": factors,
            },
        )


def _arima_110_or_100(
    rng: np.random.Generator,
    n_periods: int,
    coefficient: float,
    innovation_mean: float,
    integrated: bool,
) -> FloatMatrix:
    increments = _stationary_ar1(
        rng,
        n_periods,
        coefficient,
        innovation_mean=innovation_mean,
    )
    return np.cumsum(increments) if integrated else increments


@dataclass(frozen=True, slots=True)
class TimeSeriesOutcome:
    """Independent AR(1) or ARIMA(1, 1, 0) unit paths."""

    coefficient: float = 0.2
    integrated: bool = False
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        if not np.isfinite(self.coefficient) or abs(self.coefficient) >= 1.0:
            raise ValueError("coefficient must lie strictly between -1 and 1")
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        treated = context.treated_units
        controls = context.control_units
        treated_paths = np.vstack(
            [
                _arima_110_or_100(
                    rng,
                    context.dimensions.n_periods,
                    self.coefficient,
                    0.25,
                    self.integrated,
                )
                for _ in treated
            ]
        )
        control_paths = np.vstack(
            [
                _arima_110_or_100(
                    rng,
                    context.dimensions.n_periods,
                    self.coefficient,
                    0.0,
                    self.integrated,
                )
                for _ in controls
            ]
        )
        process = np.empty(
            (context.dimensions.n_units, context.dimensions.n_periods), dtype=float
        )
        process[controls] = control_paths
        process[treated] = treated_paths
        untreated = process + _error(
            rng,
            context.dimensions.n_units,
            context.dimensions.n_periods,
            np.sqrt(self.noise_variance),
        )
        return ComponentDraw(
            untreated,
            {
                "coefficient": self.coefficient,
                "integrated": self.integrated,
                "process": process,
            },
        )


def _mixed_group(
    rng: np.random.Generator,
    *,
    n_control: int,
    n_treated: int,
    n_periods: int,
    overlap: float,
    factor_type: str,
    noise_scale: float,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    dimensions = PanelDimensions(n_control + n_treated, n_periods)
    treatment = np.zeros((dimensions.n_units, dimensions.n_periods), dtype=float)
    if n_treated:
        treatment[n_control:, -1] = 1.0
    context = SimulationContext(dimensions, treatment)
    loadings, factors, signal = _factor_components(
        rng,
        context,
        factor_types=(factor_type, factor_type),
        control_loading_mean=-overlap,
        treated_loading_mean=overlap,
    )
    return (
        signal + _error(rng, dimensions.n_units, n_periods, noise_scale),
        loadings,
        factors,
    )


@dataclass(frozen=True, slots=True)
class MixedFactorOutcome:
    """Two half-panels with drift and cyclical factor structures."""

    overlap: float = 0.0
    noise_variance: float = 0.36

    def __post_init__(self) -> None:
        _validate_overlap(self.overlap)
        _validate_noise_variance(self.noise_variance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        n_units = context.dimensions.n_units
        if n_units % 2:
            raise ValueError("mixed factor design requires an even number of units")
        half = n_units // 2
        treated = context.treated_units
        controls = context.control_units
        if treated.size >= half:
            raise ValueError(
                "mixed factor design requires fewer treated units than half the panel"
            )
        first_control_count = half - treated.size
        first_controls = controls[:first_control_count]
        second_controls = controls[first_control_count:]
        second_rng = copy.deepcopy(rng)
        first, first_loadings, first_factors = _mixed_group(
            rng,
            n_control=first_controls.size,
            n_treated=treated.size,
            n_periods=context.dimensions.n_periods,
            overlap=self.overlap,
            factor_type="drift",
            noise_scale=np.sqrt(self.noise_variance),
        )
        second, second_loadings, second_factors = _mixed_group(
            second_rng,
            n_control=second_controls.size,
            n_treated=0,
            n_periods=context.dimensions.n_periods,
            overlap=self.overlap,
            factor_type="cyclical",
            noise_scale=np.sqrt(self.noise_variance),
        )
        untreated = np.empty((n_units, context.dimensions.n_periods), dtype=float)
        untreated[first_controls] = first[: first_controls.size]
        untreated[second_controls] = second
        untreated[treated] = first[first_controls.size :]
        return ComponentDraw(
            untreated,
            {
                "overlap": self.overlap,
                "first_group_controls": first_controls,
                "second_group_controls": second_controls,
                "first_group_loadings": first_loadings,
                "second_group_loadings": second_loadings,
                "first_group_factors": first_factors,
                "second_group_factors": second_factors,
            },
        )


def _config(value: CanonicalPanelConfig | None) -> CanonicalPanelConfig:
    return CanonicalPanelConfig() if value is None else value


def _canonical_simulator(
    *,
    name: str,
    config: CanonicalPanelConfig,
    outcome_model: OutcomeModel,
) -> PanelSimulator:
    return PanelSimulator(
        name=name,
        dimensions=config.dimensions,
        assignment=SingleCohortAssignment(
            n_treated=config.n_treated,
            adoption_period=config.n_pre,
        ),
        outcome_model=outcome_model,
        effect_model=LinearRampEffect(config.effect_slope),
    )


def classic_factor_design(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical two-factor simulator."""

    resolved = _config(config)
    return _canonical_simulator(
        name="classic_factor",
        config=resolved,
        outcome_model=ClassicFactorOutcome(overlap, resolved.noise_variance),
    )


def weak_factor_design(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical weak-factor simulator."""

    resolved = _config(config)
    return _canonical_simulator(
        name="weak_factor",
        config=resolved,
        outcome_model=WeakFactorOutcome(overlap, resolved.noise_variance),
    )


def synthetic_control_design(
    *,
    active_share: float = 0.1,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical sparse-donor simulator."""

    resolved = _config(config)
    return _canonical_simulator(
        name="synthetic_control",
        config=resolved,
        outcome_model=SyntheticControlOutcome(active_share, resolved.noise_variance),
    )


def factor_synthetic_design(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical factor-synthetic mixture simulator."""

    resolved = _config(config)
    return _canonical_simulator(
        name="factor_synthetic",
        config=resolved,
        outcome_model=FactorSyntheticOutcome(overlap, resolved.noise_variance),
    )


def time_series_design(
    *,
    coefficient: float = 0.2,
    integrated: bool = False,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical AR or integrated-AR simulator."""

    resolved = _config(config)
    name = "time_series_integrated" if integrated else "time_series_stationary"
    return _canonical_simulator(
        name=name,
        config=resolved,
        outcome_model=TimeSeriesOutcome(
            coefficient, integrated, resolved.noise_variance
        ),
    )


def mixed_factor_design(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
) -> PanelSimulator:
    """Create the canonical mixed-factor simulator."""

    resolved = _config(config)
    return _canonical_simulator(
        name="mixed_factor",
        config=resolved,
        outcome_model=MixedFactorOutcome(overlap, resolved.noise_variance),
    )


def classic_factor(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical two-factor panel."""

    return classic_factor_design(overlap=overlap, config=config).simulate(
        seed=seed, rng=rng
    )


def weak_factor(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical weak-factor panel."""

    return weak_factor_design(overlap=overlap, config=config).simulate(
        seed=seed, rng=rng
    )


def synthetic_control(
    *,
    active_share: float = 0.1,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical sparse-donor panel."""

    return synthetic_control_design(active_share=active_share, config=config).simulate(
        seed=seed, rng=rng
    )


def factor_synthetic(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical factor-synthetic panel."""

    return factor_synthetic_design(overlap=overlap, config=config).simulate(
        seed=seed, rng=rng
    )


def time_series(
    *,
    coefficient: float = 0.2,
    integrated: bool = False,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical stationary or integrated time-series panel."""

    return time_series_design(
        coefficient=coefficient,
        integrated=integrated,
        config=config,
    ).simulate(seed=seed, rng=rng)


def mixed_factor(
    *,
    overlap: float = 0.0,
    config: CanonicalPanelConfig | None = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> PanelDataset:
    """Draw one canonical mixed-factor panel."""

    return mixed_factor_design(overlap=overlap, config=config).simulate(
        seed=seed, rng=rng
    )


canonical_registry = DGPRegistry()
canonical_registry.register("classic_factor", classic_factor_design)
canonical_registry.register("factor_synthetic", factor_synthetic_design)
canonical_registry.register("mixed_factor", mixed_factor_design)
canonical_registry.register("synthetic_control", synthetic_control_design)
canonical_registry.register("time_series", time_series_design)
canonical_registry.register("weak_factor", weak_factor_design)


def make_canonical(name: str, /, **kwargs: object) -> PanelSimulator:
    """Create a canonical simulator by registry name."""

    return canonical_registry.create(name, **kwargs)


def available_canonical_designs() -> tuple[str, ...]:
    """Return the stable names accepted by :func:`make_canonical`."""

    return canonical_registry.names()


PanelConfig = CanonicalPanelConfig
PanelData = PanelDataset
