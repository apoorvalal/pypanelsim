"""Reusable untreated-outcome components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .components import AdditiveFactorOutcome, ComponentDraw, SimulationContext


def _finite_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _coefficient_vector(
    values: Sequence[float], *, expected: int, name: str
) -> np.ndarray:
    coefficients = np.asarray(values, dtype=float)
    if coefficients.shape != (expected,):
        raise ValueError(f"{name} must contain {expected} values")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must contain only finite values")
    return coefficients


def _readonly_panel_matrix(value: ArrayLike, *, name: str) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True, slots=True)
class EmpiricalPanelOutcome:
    """Use a fixed empirical panel as the untreated signal and add Gaussian noise.

    ``baseline`` uses the package-wide ``(unit, time)`` layout. Set
    ``noise_scale`` for independent Gaussian cell noise, or provide a complete
    positive-semidefinite ``noise_covariance`` to draw one correlated time path
    per unit. The two noise specifications are mutually exclusive.

    This component deliberately does not normalize, impute, smooth, or
    low-rank-project the supplied panel. Those scientific choices remain
    explicit preprocessing steps in the analysis that owns the data.
    """

    baseline: ArrayLike
    noise_scale: float = 0.0
    noise_covariance: ArrayLike | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        baseline = _readonly_panel_matrix(self.baseline, name="baseline")
        _finite_nonnegative("noise_scale", self.noise_scale)
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source.strip()
        ):
            raise ValueError("source must be a non-empty string when provided")

        covariance = None
        if self.noise_covariance is not None:
            if self.noise_scale != 0.0:
                raise ValueError("provide noise_scale or noise_covariance, not both")
            covariance = _readonly_panel_matrix(
                self.noise_covariance,
                name="noise_covariance",
            )
            expected = (baseline.shape[1], baseline.shape[1])
            if covariance.shape != expected:
                raise ValueError(f"noise_covariance must have shape {expected}")
            if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
                raise ValueError("noise_covariance must be symmetric")
            eigenvalues = np.linalg.eigvalsh(covariance)
            tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
            if float(np.min(eigenvalues)) < -tolerance:
                raise ValueError("noise_covariance must be positive semidefinite")

        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "noise_covariance", covariance)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        expected = (context.dimensions.n_units, context.dimensions.n_periods)
        if self.baseline.shape != expected:
            raise ValueError(f"baseline must have shape {expected}")

        if self.noise_covariance is not None:
            errors = rng.multivariate_normal(
                mean=np.zeros(context.dimensions.n_periods),
                cov=self.noise_covariance,
                size=context.dimensions.n_units,
                check_valid="raise",
                tol=1e-10,
            )
            noise_model = "correlated_gaussian"
        elif self.noise_scale > 0.0:
            errors = rng.normal(scale=self.noise_scale, size=expected)
            noise_model = "iid_gaussian"
        else:
            errors = np.zeros(expected, dtype=float)
            noise_model = "none"

        return ComponentDraw(
            self.baseline + errors,
            {
                "kind": "empirical_panel",
                "source": self.source,
                "baseline": self.baseline,
                "errors": errors,
                "noise_model": noise_model,
                "noise_scale": self.noise_scale,
                "noise_covariance": self.noise_covariance,
            },
        )


@dataclass(frozen=True, slots=True)
class TwoWayFixedEffectsOutcome(AdditiveFactorOutcome):
    """Explicitly named additive unit and time fixed-effects primitive."""

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        draw = AdditiveFactorOutcome.generate(self, context, rng)
        return ComponentDraw(
            draw.values,
            {**dict(draw.metadata), "kind": "two_way_fixed_effects"},
        )


@dataclass(frozen=True, slots=True)
class LinearFeatureOutcome:
    """Use unit features for levels and unit-specific linear trends."""

    level_coefficients: Sequence[float] = ()
    trend_coefficients: Sequence[float] = ()
    source: str = "observables"
    noise_scale: float = 0.0
    center_time: bool = False

    def __post_init__(self) -> None:
        if self.source not in {"observables", "unobservables"}:
            raise ValueError("source must be 'observables' or 'unobservables'")
        _finite_nonnegative("noise_scale", self.noise_scale)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        features = getattr(context, self.source)
        level = _coefficient_vector(
            self.level_coefficients,
            expected=features.shape[1],
            name="level_coefficients",
        )
        trend = _coefficient_vector(
            self.trend_coefficients,
            expected=features.shape[1],
            name="trend_coefficients",
        )
        time = np.arange(context.dimensions.n_periods, dtype=float)
        if self.center_time:
            time -= time.mean()
        unit_levels = features @ level
        unit_trends = features @ trend
        errors = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        values = unit_levels[:, None] + unit_trends[:, None] * time + errors
        return ComponentDraw(
            values,
            {
                "kind": "linear_feature",
                "source": self.source,
                "level_coefficients": level,
                "trend_coefficients": trend,
                "unit_levels": unit_levels,
                "unit_trends": unit_trends,
                "errors": errors,
            },
        )


@dataclass(frozen=True, slots=True)
class LowRankFactorOutcome:
    """Generate a generic rank-k interactive fixed-effects outcome."""

    rank: int = 2
    loading_scale: float = 1.0
    factor_scale: float = 1.0
    noise_scale: float = 0.0
    distribution: str = "normal"
    loading_mean: float = 0.0
    control_loading_mean: float | None = None
    treated_loading_mean: float | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        for name in ("loading_scale", "factor_scale", "noise_scale"):
            _finite_nonnegative(name, getattr(self, name))
        if self.distribution not in {"normal", "uniform"}:
            raise ValueError("distribution must be 'normal' or 'uniform'")
        means = (
            self.loading_mean,
            self.control_loading_mean,
            self.treated_loading_mean,
        )
        if any(value is not None and not np.isfinite(value) for value in means):
            raise ValueError("loading means must be finite")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        unit_means = np.full(context.dimensions.n_units, self.loading_mean)
        if self.control_loading_mean is not None:
            unit_means[context.control_units] = self.control_loading_mean
        if self.treated_loading_mean is not None:
            unit_means[context.treated_units] = self.treated_loading_mean
        if self.distribution == "normal":
            loadings = rng.normal(
                loc=unit_means[:, None],
                scale=self.loading_scale,
                size=(context.dimensions.n_units, self.rank),
            )
            factors = rng.normal(
                scale=self.factor_scale,
                size=(context.dimensions.n_periods, self.rank),
            )
        else:
            loadings = rng.uniform(
                -self.loading_scale,
                self.loading_scale,
                size=(context.dimensions.n_units, self.rank),
            )
            loadings += unit_means[:, None]
            factors = rng.uniform(
                -self.factor_scale,
                self.factor_scale,
                size=(context.dimensions.n_periods, self.rank),
            )
        signal = loadings @ factors.T
        errors = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        return ComponentDraw(
            signal + errors,
            {
                "kind": "low_rank_factor",
                "rank": self.rank,
                "unit_loading_means": unit_means,
                "loadings": loadings,
                "factors": factors,
                "signal": signal,
                "errors": errors,
            },
        )


@dataclass(frozen=True, slots=True)
class UnitTrendOutcome:
    """Generate unit-specific linear trends."""

    scale: float = 1.0
    center_time: bool = False

    def __post_init__(self) -> None:
        _finite_nonnegative("scale", self.scale)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        slopes = rng.normal(scale=self.scale, size=context.dimensions.n_units)
        time = np.arange(context.dimensions.n_periods, dtype=float)
        if self.center_time:
            time -= time.mean()
        return ComponentDraw(
            slopes[:, None] * time,
            {"kind": "unit_trend", "slopes": slopes, "time_scores": time},
        )


@dataclass(frozen=True, slots=True)
class UnitPositionOutcome:
    """Generate unit levels around a linear position-dependent mean."""

    location_start: float = 0.0
    location_stop: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.location_start) or not np.isfinite(self.location_stop):
            raise ValueError("locations must be finite")
        _finite_nonnegative("scale", self.scale)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        means = np.linspace(
            self.location_start,
            self.location_stop,
            context.dimensions.n_units,
            endpoint=False,
        )
        means += (self.location_stop - self.location_start) / context.dimensions.n_units
        levels = rng.normal(means, self.scale)
        values = np.broadcast_to(
            levels[:, None],
            (context.dimensions.n_units, context.dimensions.n_periods),
        )
        return ComponentDraw(
            values,
            {
                "kind": "unit_position",
                "unit_means": means,
                "unit_levels": levels,
            },
        )


@dataclass(frozen=True, slots=True)
class PeriodicTimeOutcome:
    """Repeat a fixed finite time-effect pattern across the panel."""

    pattern: Sequence[float] = (-0.1, 0.1, 0.0, 0.0, 0.1, 0.5, 0.5)
    scale: float = 1.0

    def __post_init__(self) -> None:
        pattern = tuple(float(value) for value in self.pattern)
        if not pattern or not np.all(np.isfinite(pattern)):
            raise ValueError("pattern must contain finite values")
        if not np.isfinite(self.scale):
            raise ValueError("scale must be finite")
        object.__setattr__(self, "pattern", pattern)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        del rng
        repeated = np.resize(
            np.asarray(self.pattern, dtype=float), context.dimensions.n_periods
        )
        repeated *= self.scale
        values = np.broadcast_to(
            repeated, (context.dimensions.n_units, context.dimensions.n_periods)
        )
        return ComponentDraw(
            values,
            {"kind": "periodic_time", "time_effects": repeated},
        )


@dataclass(frozen=True, slots=True)
class ARMAErrorOutcome:
    """Generate independent unit-level ARMA error paths."""

    ar_coefficients: Sequence[float] = (0.7,)
    ma_coefficients: Sequence[float] = ()
    innovation_scale: float = 1.0
    burn_in: int = 100

    def __post_init__(self) -> None:
        ar = tuple(float(value) for value in self.ar_coefficients)
        ma = tuple(float(value) for value in self.ma_coefficients)
        if not np.all(np.isfinite(ar)) or not np.all(np.isfinite(ma)):
            raise ValueError("ARMA coefficients must be finite")
        _finite_nonnegative("innovation_scale", self.innovation_scale)
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")
        object.__setattr__(self, "ar_coefficients", ar)
        object.__setattr__(self, "ma_coefficients", ma)

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        ar = np.asarray(self.ar_coefficients, dtype=float)
        ma = np.asarray(self.ma_coefficients, dtype=float)
        total = context.dimensions.n_periods + self.burn_in
        innovations = rng.normal(
            scale=self.innovation_scale,
            size=(context.dimensions.n_units, total),
        )
        paths = np.zeros_like(innovations)
        for period in range(total):
            paths[:, period] = innovations[:, period]
            for lag, coefficient in enumerate(ar, start=1):
                if period >= lag:
                    paths[:, period] += coefficient * paths[:, period - lag]
            for lag, coefficient in enumerate(ma, start=1):
                if period >= lag:
                    paths[:, period] += coefficient * innovations[:, period - lag]
        values = paths[:, self.burn_in :]
        return ComponentDraw(
            values,
            {
                "kind": "arma_error",
                "ar_coefficients": ar,
                "ma_coefficients": ma,
                "innovations": innovations,
            },
        )


@dataclass(frozen=True, slots=True)
class RandomARMAErrorOutcome:
    """Draw one stationary ARMA specification shared by all unit paths."""

    max_ar_order: int = 2
    max_ma_order: int = 2
    ar_bound: float = 0.5
    ma_bound: float = 0.5
    innovation_scale: float = 1.0
    burn_in: int = 100

    def __post_init__(self) -> None:
        if self.max_ar_order <= 0 or self.max_ma_order <= 0:
            raise ValueError("maximum ARMA orders must be positive")
        for name in ("ar_bound", "ma_bound", "innovation_scale"):
            _finite_nonnegative(name, getattr(self, name))
        if self.burn_in < 0:
            raise ValueError("burn_in must be nonnegative")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        ar_order = int(rng.integers(1, self.max_ar_order + 1))
        ma_order = int(rng.integers(1, self.max_ma_order + 1))
        for _ in range(10_000):
            ar = rng.uniform(-self.ar_bound, self.ar_bound, ar_order)
            if np.abs(ar).sum() < 1.0:
                break
        else:
            raise ValueError(
                "could not draw stationary AR coefficients within 10,000 attempts"
            )
        ma = rng.uniform(-self.ma_bound, self.ma_bound, ma_order)
        draw = ARMAErrorOutcome(
            tuple(ar),
            tuple(ma),
            innovation_scale=self.innovation_scale,
            burn_in=self.burn_in,
        ).generate(context, rng)
        return ComponentDraw(
            draw.values,
            {
                **dict(draw.metadata),
                "kind": "random_arma_error",
                "selected_ar_order": ar_order,
                "selected_ma_order": ma_order,
            },
        )


@dataclass(frozen=True, slots=True)
class ClusteredTrendOutcome:
    """Generate latent groups whose members share a linear time slope."""

    n_clusters: int = 5
    slope_scale: float = 1.0
    within_cluster_scale: float = 0.0
    center_distribution: str = "normal"
    center_time: bool = False

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        _finite_nonnegative("slope_scale", self.slope_scale)
        _finite_nonnegative("within_cluster_scale", self.within_cluster_scale)
        if self.center_distribution not in {"normal", "uniform"}:
            raise ValueError("center_distribution must be 'normal' or 'uniform'")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        clusters = rng.integers(0, self.n_clusters, size=context.dimensions.n_units)
        if self.center_distribution == "normal":
            slopes = rng.normal(scale=self.slope_scale, size=self.n_clusters)
        else:
            slopes = rng.uniform(
                -self.slope_scale, self.slope_scale, size=self.n_clusters
            )
        unit_slopes = slopes[clusters] + rng.normal(
            scale=self.within_cluster_scale, size=context.dimensions.n_units
        )
        time = np.arange(context.dimensions.n_periods, dtype=float)
        if self.center_time:
            time -= time.mean()
        values = unit_slopes[:, None] * time
        return ComponentDraw(
            values,
            {
                "kind": "clustered_trend",
                "cluster_assignments": clusters,
                "cluster_slopes": slopes,
                "unit_slopes": unit_slopes,
            },
        )


@dataclass(frozen=True, slots=True)
class LatentSelectionOutcome:
    """Outcome law from the ATT-DML latent-selection factor experiment."""

    stable: bool = False
    noise_scale: float = 0.1
    trend_loading_bounds: tuple[float, float] = (0.1, 0.5)

    def __post_init__(self) -> None:
        _finite_nonnegative("noise_scale", self.noise_scale)
        lower, upper = self.trend_loading_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("trend_loading_bounds must be finite and ordered")

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        if context.unobservables.shape[1] < 1:
            raise ValueError("LatentSelectionOutcome requires one latent feature")
        latent = context.unobservables[:, 0]
        time = np.arange(1, context.dimensions.n_periods + 1, dtype=float)
        if self.stable:
            loadings = np.ones(context.dimensions.n_periods)
            signal = latent[:, None] + 0.1 * time
        else:
            lower, upper = self.trend_loading_bounds
            loadings = rng.uniform(lower, upper, context.dimensions.n_periods)
            signal = latent[:, None] * (loadings * time)[None, :]
        errors = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        return ComponentDraw(
            signal + errors,
            {
                "kind": "latent_selection",
                "stable": self.stable,
                "time_loadings": loadings,
                "signal": signal,
                "errors": errors,
            },
        )
