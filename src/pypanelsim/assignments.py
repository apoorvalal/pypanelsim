"""Additional reusable treatment-assignment mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import AssignmentContext, ComponentDraw, PanelDimensions


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    nonnegative = values >= 0.0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


@dataclass(frozen=True, slots=True)
class SparseLogitAssignment:
    """Select a random sparse feature index for one-cohort logit assignment."""

    adoption_period: int
    n_active: int = 5
    intercept: float = -1.0
    coefficient_bounds: tuple[float, float] = (-1.0, 1.0)
    logit_noise_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.adoption_period < 0:
            raise ValueError("adoption_period must be nonnegative")
        if self.n_active <= 0:
            raise ValueError("n_active must be positive")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        lower, upper = self.coefficient_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("coefficient_bounds must be finite and ordered")
        if not np.isfinite(self.logit_noise_scale) or self.logit_noise_scale < 0.0:
            raise ValueError("logit_noise_scale must be finite and nonnegative")

    def assign(
        self,
        context: AssignmentContext | PanelDimensions,
        rng: np.random.Generator,
    ) -> ComponentDraw:
        assignment_context = (
            context
            if isinstance(context, AssignmentContext)
            else AssignmentContext(context)
        )
        dimensions = assignment_context.dimensions
        if self.adoption_period >= dimensions.n_periods:
            raise ValueError("adoption_period must be a valid time position")
        n_features = assignment_context.observables.shape[1]
        if self.n_active > n_features:
            raise ValueError("n_active cannot exceed the observable feature count")
        active = np.sort(rng.choice(n_features, self.n_active, replace=False))
        coefficients = np.zeros(n_features)
        lower, upper = self.coefficient_bounds
        coefficients[active] = rng.uniform(lower, upper, self.n_active)
        linear_predictor = (
            self.intercept + assignment_context.observables @ coefficients
        )
        if self.logit_noise_scale:
            linear_predictor += rng.normal(
                scale=self.logit_noise_scale, size=dimensions.n_units
            )
        propensity = _sigmoid(linear_predictor)
        treated = rng.random(dimensions.n_units) < propensity
        treatment = np.zeros((dimensions.n_units, dimensions.n_periods))
        treatment[treated, self.adoption_period :] = 1.0
        return ComponentDraw(
            treatment,
            {
                "kind": "sparse_logit",
                "adoption_period": self.adoption_period,
                "active_features": active,
                "coefficients": coefficients,
                "linear_predictor": linear_predictor,
                "propensity_scores": propensity,
            },
        )
