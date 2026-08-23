"""Composite design adapted from the original gsynth2 DGP constructor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..canonical import SyntheticControlOutcome
from ..components import (
    AdditiveFactorOutcome,
    LinearRampEffect,
    PanelDimensions,
    SingleCohortAssignment,
    SumOutcomeModel,
)
from ..outcomes import ARMAErrorOutcome, LowRankFactorOutcome
from ..simulator import PanelSimulator


@dataclass(frozen=True, slots=True)
class GSynthCompositeConfig:
    """Weights and dimensions for a factor/TS/SCM/TWFE composite panel."""

    n_control: int = 160
    n_treated: int = 40
    n_pre: int = 40
    n_post: int = 10
    factor_weight: float = 0.25
    time_series_weight: float = 0.25
    synthetic_weight: float = 0.25
    fixed_effect_weight: float = 0.25
    factor_rank: int = 2
    ar_coefficient: float = 0.5
    effect_slope: float = 0.2
    noise_scale: float = 0.6

    def __post_init__(self) -> None:
        for name in ("n_control", "n_treated", "n_pre", "n_post", "factor_rank"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        weights = self.weights
        if not np.all(np.isfinite(weights)) or any(value < 0.0 for value in weights):
            raise ValueError("component weights must be finite and nonnegative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one component weight must be positive")
        if not np.isfinite(self.ar_coefficient) or abs(self.ar_coefficient) >= 1.0:
            raise ValueError("ar_coefficient must lie strictly between -1 and 1")
        if not np.isfinite(self.effect_slope):
            raise ValueError("effect_slope must be finite")
        if not np.isfinite(self.noise_scale) or self.noise_scale < 0.0:
            raise ValueError("noise_scale must be finite and nonnegative")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(
            self.n_control + self.n_treated, self.n_pre + self.n_post
        )

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (
            self.factor_weight,
            self.time_series_weight,
            self.synthetic_weight,
            self.fixed_effect_weight,
        )

    @property
    def normalized_weights(self) -> tuple[float, float, float, float]:
        total = sum(self.weights)
        return tuple(value / total for value in self.weights)


def gsynth_composite_design(
    *, config: GSynthCompositeConfig | None = None
) -> PanelSimulator:
    """Build the weighted gsynth2 factor/TS/SCM/TWFE outcome design."""

    resolved = GSynthCompositeConfig() if config is None else config
    return PanelSimulator(
        name="gsynth_composite",
        dimensions=resolved.dimensions,
        assignment=SingleCohortAssignment(
            resolved.n_treated, resolved.n_pre
        ),
        outcome_model=SumOutcomeModel(
            (
                LowRankFactorOutcome(
                    rank=resolved.factor_rank,
                    noise_scale=resolved.noise_scale,
                ),
                ARMAErrorOutcome(
                    (resolved.ar_coefficient,),
                    innovation_scale=1.0,
                ),
                SyntheticControlOutcome(
                    active_share=min(1.0, 5.0 / resolved.n_control),
                    noise_variance=resolved.noise_scale**2,
                ),
                AdditiveFactorOutcome(1.0, 1.0, resolved.noise_scale),
            ),
            weights=resolved.normalized_weights,
        ),
        effect_model=LinearRampEffect(resolved.effect_slope),
    )
