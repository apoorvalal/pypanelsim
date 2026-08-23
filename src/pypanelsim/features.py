"""Reusable observed and latent feature generators."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .components import PanelDimensions, UnitFeatureDraw

FeatureTransform = Callable[[np.ndarray], np.ndarray]


def att_dml_nonlinear_basis(features: np.ndarray) -> np.ndarray:
    """Append the three nonlinear terms used in the ATT-DML simulation."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("ATT-DML nonlinear features require at least six columns")
    safe_absolute = np.maximum(np.abs(values[:, 2]), np.finfo(float).tiny)
    additions = np.column_stack(
        (
            np.sin(values[:, 0] ** 2),
            values[:, 5] ** 2 * values[:, 1],
            np.log(safe_absolute) * values[:, 3],
        )
    )
    return np.column_stack((values, additions))


@dataclass(frozen=True, slots=True)
class CorrelatedGaussianFeatures:
    """Draw Gaussian covariates with Toeplitz or explicit covariance.

    A callable ``transform`` may expand the features used by assignment and
    outcome models. With ``estimator_uses_raw=True``, the returned panel keeps
    the original Gaussian columns as estimator-visible covariates. This makes
    nonlinear nuisance-model misspecification explicit.
    """

    n_features: int = 10
    correlation: float = 0.5
    covariance: Sequence[Sequence[float]] | None = None
    transform: FeatureTransform | None = None
    estimator_uses_raw: bool = True
    name_prefix: str = "x"

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if not np.isfinite(self.correlation) or abs(self.correlation) >= 1.0:
            raise ValueError("correlation must lie strictly between -1 and 1")
        if self.transform is not None and not callable(self.transform):
            raise TypeError("transform must be callable")
        if not isinstance(self.name_prefix, str) or not self.name_prefix:
            raise ValueError("name_prefix must be a non-empty string")
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            expected = (self.n_features, self.n_features)
            if covariance.shape != expected:
                raise ValueError(f"covariance must have shape {expected}")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("covariance must contain only finite values")
            if not np.allclose(covariance, covariance.T):
                raise ValueError("covariance must be symmetric")
            if np.linalg.eigvalsh(covariance).min() <= 0.0:
                raise ValueError("covariance must be positive definite")

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> UnitFeatureDraw:
        if self.covariance is None:
            positions = np.arange(self.n_features)
            covariance = self.correlation ** np.abs(
                positions[:, None] - positions[None, :]
            )
        else:
            covariance = np.asarray(self.covariance, dtype=float)
        raw = rng.multivariate_normal(
            np.zeros(self.n_features), covariance, size=dimensions.n_units
        )
        transformed = raw if self.transform is None else self.transform(raw)
        transformed = np.asarray(transformed, dtype=float)
        if transformed.ndim != 2 or transformed.shape[0] != dimensions.n_units:
            raise ValueError("transform must return shape (n_units, n_features)")
        if not np.all(np.isfinite(transformed)):
            raise ValueError("transformed features must contain only finite values")
        names = tuple(
            f"{self.name_prefix}{index + 1}" for index in range(transformed.shape[1])
        )
        raw_names = tuple(
            f"{self.name_prefix}{index + 1}" for index in range(self.n_features)
        )
        estimator = raw if self.estimator_uses_raw else transformed
        estimator_names = raw_names if self.estimator_uses_raw else names
        return UnitFeatureDraw(
            observables=transformed,
            unobservables=np.empty((dimensions.n_units, 0)),
            metadata={
                "kind": "correlated_gaussian",
                "covariance": covariance,
                "observable_names": names,
                "estimator_observable_names": estimator_names,
                "raw_features": raw,
                "transformed": self.transform is not None,
            },
            estimator_observables=estimator,
        )


@dataclass(frozen=True, slots=True)
class LatentGradientFeatures:
    """Draw one latent factor with a unit-position-dependent mean."""

    location_start: float = -0.5
    location_stop: float = 0.5
    scale: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.location_start) or not np.isfinite(self.location_stop):
            raise ValueError("locations must be finite")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("scale must be finite and nonnegative")

    def generate(
        self, dimensions: PanelDimensions, rng: np.random.Generator
    ) -> UnitFeatureDraw:
        means = np.linspace(
            self.location_start,
            self.location_stop,
            dimensions.n_units,
            endpoint=False,
        )
        means += (self.location_stop - self.location_start) / dimensions.n_units
        latent = rng.normal(means, self.scale)[:, None]
        return UnitFeatureDraw(
            observables=np.empty((dimensions.n_units, 0)),
            unobservables=latent,
            metadata={
                "kind": "latent_gradient",
                "unobservable_names": ("latent_factor",),
                "latent_means": means,
            },
        )
