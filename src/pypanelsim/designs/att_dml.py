"""Designs adapted from the ATT-DML simulation studies."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..assignments import SparseLogitAssignment
from ..components import (
    AdditiveFactorOutcome,
    BinaryLogitAssignment,
    ComponentDraw,
    ConstantEffect,
    PanelDimensions,
    RandomizedSingleCohortAssignment,
    RandomizedStaggeredAdoption,
    SimulationContext,
    SumOutcomeModel,
)
from ..data import PanelDataset
from ..features import (
    CorrelatedGaussianFeatures,
    LatentGradientFeatures,
    att_dml_nonlinear_basis,
)
from ..outcomes import (
    ARMAErrorOutcome,
    ClusteredTrendOutcome,
    LatentSelectionOutcome,
    RandomARMAErrorOutcome,
    UnitPositionOutcome,
)
from ..profiles import GaussianCellEffect
from ..simulator import PanelSimulator, SimulationSeeds

_DESIGNS = MappingProxyType(
    {
        "conditional_parallel_trends": "Two-period sparse conditional trends",
        "conditional_parallel_trends_nonlinear": (
            "Two-period sparse nonlinear conditional trends"
        ),
        "latent_parallel": "Selection on a stable latent outcome factor",
        "latent_nonparallel": "Selection on an interactive latent outcome factor",
        "clustered_arma": "Fixed effects, clustered trends, and ARMA errors",
    }
)


def available_att_dml_designs() -> tuple[str, ...]:
    """Return the stable names in the ATT-DML design suite."""

    return tuple(_DESIGNS)


@dataclass(frozen=True, slots=True)
class ATTDMConditionalConfig:
    """Parameters for the two-period conditional-trends experiment."""

    n_units: int = 2_000
    n_features: int = 20
    n_active: int = 5
    treatment_effect: float = 3.0
    treatment_effect_noise_scale: float = 0.3
    trend_strength: float = 0.0
    nonlinear: bool = False
    treated_baseline_shift: bool = False

    def __post_init__(self) -> None:
        if self.n_units <= 1:
            raise ValueError("n_units must exceed one")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if self.n_active <= 0 or self.n_active > self.n_features:
            raise ValueError("n_active must lie between one and n_features")
        if self.nonlinear and self.n_features < 6:
            raise ValueError("nonlinear ATT-DML designs require six features")
        if not np.isfinite(self.treatment_effect):
            raise ValueError("treatment_effect must be finite")
        if (
            not np.isfinite(self.treatment_effect_noise_scale)
            or self.treatment_effect_noise_scale < 0.0
        ):
            raise ValueError(
                "treatment_effect_noise_scale must be finite and nonnegative"
            )
        if not np.isfinite(self.trend_strength) or self.trend_strength < 0.0:
            raise ValueError("trend_strength must be finite and nonnegative")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, 2)


@dataclass(frozen=True, slots=True)
class ATTDMLatentConfig:
    """Parameters for the latent-factor selection experiment."""

    n_units: int = 1_000
    n_periods: int = 7
    noise_scale: float = 0.1
    treatment_effect: float = 0.0
    unconfounded_assignment: bool = False

    def __post_init__(self) -> None:
        if self.n_units <= 1 or self.n_periods <= 1:
            raise ValueError("latent design dimensions must exceed one")
        if not np.isfinite(self.noise_scale) or self.noise_scale < 0.0:
            raise ValueError("noise_scale must be finite and nonnegative")
        if not np.isfinite(self.treatment_effect):
            raise ValueError("treatment_effect must be finite")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, self.n_periods)


@dataclass(frozen=True, slots=True)
class ATTDMClusteredConfig:
    """Parameters for the clustered-trend ARMA panel experiment."""

    n_units: int = 50
    n_periods: int = 50
    n_treated: int = 10
    adoption_period: int = 34
    n_clusters: int = 10
    assignment_pattern: str = "simultaneous"
    ar_coefficients: tuple[float, ...] = (0.3, -0.1)
    ma_coefficients: tuple[float, ...] = (0.2,)
    randomize_arma: bool = True
    max_ar_order: int = 2
    max_ma_order: int = 2
    ar_bound: float = 0.5
    ma_bound: float = 0.5
    trend_scale: float = 0.6
    trim_share: float = 0.05

    def __post_init__(self) -> None:
        if self.n_units <= 1 or self.n_periods <= 1:
            raise ValueError("clustered design dimensions must exceed one")
        if not 0 < self.n_treated < self.n_units:
            raise ValueError("n_treated must lie between zero and n_units")
        if not 0 <= self.adoption_period < self.n_periods:
            raise ValueError("adoption_period must lie inside the panel")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.max_ar_order <= 0 or self.max_ma_order <= 0:
            raise ValueError("maximum ARMA orders must be positive")
        if self.assignment_pattern not in {"simultaneous", "staggered"}:
            raise ValueError("assignment_pattern must be simultaneous or staggered")
        for name in ("trend_scale", "ar_bound", "ma_bound"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.trim_share) or not 0.0 <= self.trim_share < 0.5:
            raise ValueError("trim_share must lie in [0, 0.5)")

    @property
    def dimensions(self) -> PanelDimensions:
        return PanelDimensions(self.n_units, self.n_periods)


def _random_sparse(
    rng: np.random.Generator,
    n_features: int,
    n_active: int,
    bounds: tuple[float, float],
    required: tuple[int, ...] = (),
    required_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    random_active = rng.choice(n_features, n_active, replace=False)
    required_array = np.asarray(required, dtype=int)
    required_array = np.where(
        required_array < 0, required_array + n_features, required_array
    )
    active = np.unique(np.concatenate((random_active, required_array)))
    coefficients = np.zeros(n_features)
    coefficients[random_active] = rng.uniform(*bounds, n_active)
    if required_array.size:
        required_range = bounds if required_bounds is None else required_bounds
        coefficients[required_array] = rng.uniform(*required_range, required_array.size)
    return coefficients, active


@dataclass(frozen=True, slots=True)
class _ConditionalTrendOutcome:
    config: ATTDMConditionalConfig

    def generate(
        self, context: SimulationContext, rng: np.random.Generator
    ) -> ComponentDraw:
        features = context.observables
        n_features = features.shape[1]
        n_active = self.config.n_active
        required = (-3, -2, -1) if self.config.nonlinear else ()
        baseline_coefficients, baseline_active = _random_sparse(
            rng, n_features, n_active, (-1.0, 1.0), required
        )
        treated_coefficients, treated_active = _random_sparse(
            rng, n_features, n_active, (0.0, 2.0), required
        )
        trend_coefficients, trend_active = _random_sparse(
            rng,
            n_features,
            n_active,
            (0.0, 4.0 * self.config.trend_strength),
            required,
            (0.0, 2.0),
        )
        baseline_error = rng.normal(0.0, 0.2, context.dimensions.n_units)
        treated_error = rng.normal(0.0, 0.2, context.dimensions.n_units)
        change_error = rng.normal(0.0, 0.3, context.dimensions.n_units)
        baseline = features @ baseline_coefficients + baseline_error
        if self.config.treated_baseline_shift:
            treated_baseline = features @ treated_coefficients + treated_error
            baseline = np.where(context.ever_treated, treated_baseline, baseline)
        change = (
            rng.uniform(2.0, 3.0, context.dimensions.n_units)
            + features @ trend_coefficients
            + change_error
        )
        values = np.column_stack((baseline, baseline + change))
        return ComponentDraw(
            values,
            {
                "kind": "att_dml_conditional_trend",
                "baseline_coefficients": baseline_coefficients,
                "baseline_active": baseline_active,
                "treated_level_coefficients": treated_coefficients,
                "treated_level_active": treated_active,
                "trend_coefficients": trend_coefficients,
                "trend_active": trend_active,
                "treated_baseline_shift": self.config.treated_baseline_shift,
                "untreated_change": change,
            },
        )


def _conditional_design(config: ATTDMConditionalConfig) -> PanelSimulator:
    transform = att_dml_nonlinear_basis if config.nonlinear else None
    return PanelSimulator(
        name=(
            "att_dml_conditional_parallel_trends_nonlinear"
            if config.nonlinear
            else "att_dml_conditional_parallel_trends"
        ),
        dimensions=config.dimensions,
        feature_model=CorrelatedGaussianFeatures(
            n_features=config.n_features,
            correlation=0.5,
            transform=transform,
            estimator_uses_raw=True,
        ),
        assignment=SparseLogitAssignment(
            adoption_period=1,
            n_active=config.n_active,
            intercept=-1.0,
            coefficient_bounds=(-1.0, 1.0),
            logit_noise_scale=0.05,
            required_active_features=((-3, -2, -1) if config.nonlinear else ()),
        ),
        outcome_model=_ConditionalTrendOutcome(config),
        effect_model=GaussianCellEffect(
            config.treatment_effect,
            config.treatment_effect_noise_scale,
        ),
        time_ids=(0, 1),
    )


def _latent_design(config: ATTDMLatentConfig, *, stable: bool) -> PanelSimulator:
    assignment = BinaryLogitAssignment(
        adoption_period=config.n_periods - 1,
        intercept=0.0,
        observable_coefficients=(),
        unobservable_coefficients=(
            (0.0,) if config.unconfounded_assignment else (1.0,)
        ),
    )
    return PanelSimulator(
        name=f"att_dml_latent_{'parallel' if stable else 'nonparallel'}",
        dimensions=config.dimensions,
        feature_model=LatentGradientFeatures(),
        assignment=assignment,
        outcome_model=LatentSelectionOutcome(
            stable=stable, noise_scale=config.noise_scale
        ),
        effect_model=ConstantEffect(config.treatment_effect),
    )


def _clustered_assignment(config: ATTDMClusteredConfig):
    if config.assignment_pattern == "simultaneous":
        trimmed = int(np.ceil(config.trim_share * config.n_units))
        eligible = tuple(range(trimmed, config.n_units - trimmed))
        if config.n_treated > len(eligible):
            raise ValueError("n_treated exceeds the trimmed eligible unit count")
        return RandomizedSingleCohortAssignment(
            config.n_treated,
            config.adoption_period,
            eligible_units=eligible,
        )
    available_periods = config.n_periods - config.adoption_period
    n_cohorts = min(config.n_treated, available_periods)
    periods = tuple(range(config.adoption_period, config.adoption_period + n_cohorts))
    quotient, remainder = divmod(config.n_treated, n_cohorts)
    sizes = tuple(quotient + (index < remainder) for index in range(n_cohorts))
    return RandomizedStaggeredAdoption(periods, sizes)


def _clustered_design(config: ATTDMClusteredConfig) -> PanelSimulator:
    if config.randomize_arma:
        arma_model = RandomARMAErrorOutcome(
            max_ar_order=config.max_ar_order,
            max_ma_order=config.max_ma_order,
            ar_bound=config.ar_bound,
            ma_bound=config.ma_bound,
        )
    else:
        arma_model = ARMAErrorOutcome(
            config.ar_coefficients,
            config.ma_coefficients,
            innovation_scale=1.0,
        )
    return PanelSimulator(
        name=f"att_dml_clustered_arma_{config.assignment_pattern}",
        dimensions=config.dimensions,
        assignment=_clustered_assignment(config),
        outcome_model=SumOutcomeModel(
            (
                UnitPositionOutcome(0.0, 2.0, 1.0),
                AdditiveFactorOutcome(0.0, 1.0, 0.5),
                ClusteredTrendOutcome(
                    config.n_clusters,
                    config.trend_scale,
                    within_cluster_scale=0.05,
                    center_distribution="uniform",
                ),
                arma_model,
            )
        ),
        effect_model=ConstantEffect(0.0),
    )


def att_dml_design(
    name: str,
    *,
    config: (
        ATTDMConditionalConfig | ATTDMLatentConfig | ATTDMClusteredConfig | None
    ) = None,
) -> PanelSimulator:
    """Build one named ATT-DML simulator."""

    if name not in _DESIGNS:
        available = ", ".join(_DESIGNS)
        raise KeyError(f"unknown ATT-DML design {name!r}; available: {available}")
    if name.startswith("conditional_parallel_trends"):
        if config is not None and not isinstance(config, ATTDMConditionalConfig):
            raise TypeError("conditional designs require ATTDMConditionalConfig")
        resolved = ATTDMConditionalConfig() if config is None else config
        nonlinear = name.endswith("_nonlinear")
        if resolved.nonlinear != nonlinear:
            resolved = ATTDMConditionalConfig(
                n_units=resolved.n_units,
                n_features=resolved.n_features,
                n_active=resolved.n_active,
                treatment_effect=resolved.treatment_effect,
                treatment_effect_noise_scale=(resolved.treatment_effect_noise_scale),
                trend_strength=resolved.trend_strength,
                nonlinear=nonlinear,
                treated_baseline_shift=resolved.treated_baseline_shift,
            )
        return _conditional_design(resolved)
    if name.startswith("latent_"):
        if config is not None and not isinstance(config, ATTDMLatentConfig):
            raise TypeError("latent designs require ATTDMLatentConfig")
        latent = ATTDMLatentConfig() if config is None else config
        return _latent_design(latent, stable=name == "latent_parallel")
    if config is not None and not isinstance(config, ATTDMClusteredConfig):
        raise TypeError("clustered_arma requires ATTDMClusteredConfig")
    clustered = ATTDMClusteredConfig() if config is None else config
    return _clustered_design(clustered)


def att_dml(
    name: str,
    *,
    config: (
        ATTDMConditionalConfig | ATTDMLatentConfig | ATTDMClusteredConfig | None
    ) = None,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
    streams: SimulationSeeds | None = None,
) -> PanelDataset:
    """Draw one panel from an ATT-DML design."""

    return att_dml_design(name, config=config).simulate(
        seed=seed, rng=rng, streams=streams
    )
