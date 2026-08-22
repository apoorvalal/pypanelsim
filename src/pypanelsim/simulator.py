"""Simulation orchestration independent of estimator libraries."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .components import (
    AssignmentContext,
    AssignmentModel,
    CallableEffect,
    EffectCallable,
    EffectModel,
    OutcomeModel,
    PanelDimensions,
    SimulationContext,
    TimeFeatureModel,
    UnitFeatureModel,
)
from .data import FloatMatrix, PanelDataset


def resolve_rng(
    *,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Return one explicit NumPy generator and reject ambiguous RNG input."""

    if seed is not None and rng is not None:
        raise ValueError("provide seed or rng, not both")
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return rng
    return np.random.default_rng(seed)


def _component_matrix(
    values: FloatMatrix,
    *,
    dimensions: PanelDimensions,
    name: str,
) -> FloatMatrix:
    array = np.asarray(values, dtype=float)
    expected = (dimensions.n_units, dimensions.n_periods)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class PanelSimulator:
    """Compose assignment, untreated outcomes, and treatment effects."""

    name: str
    dimensions: PanelDimensions
    assignment: AssignmentModel
    outcome_model: OutcomeModel
    effect_model: EffectModel | EffectCallable
    feature_model: UnitFeatureModel | None = None
    time_feature_model: TimeFeatureModel | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.effect_model, EffectModel):
            if not callable(self.effect_model):
                raise TypeError("effect_model must satisfy EffectModel or be callable")
            object.__setattr__(
                self,
                "effect_model",
                CallableEffect(self.effect_model),
            )

    def simulate(
        self,
        *,
        seed: int | np.random.SeedSequence | None = None,
        rng: np.random.Generator | None = None,
    ) -> PanelDataset:
        """Draw one panel using an explicit seed or generator."""

        generator = resolve_rng(seed=seed, rng=rng)
        if self.feature_model is None:
            assignment_context = AssignmentContext(self.dimensions)
            feature_metadata = None
        else:
            feature_draw = self.feature_model.generate(self.dimensions, generator)
            assignment_context = AssignmentContext(
                self.dimensions,
                feature_draw.observables,
                feature_draw.unobservables,
                feature_draw.metadata,
            )
            feature_metadata = {
                "model": type(self.feature_model).__name__,
                **dict(feature_draw.metadata),
                "unobservables": assignment_context.unobservables,
            }
        assignment_draw = self.assignment.assign(assignment_context, generator)
        treatment = _component_matrix(
            assignment_draw.values,
            dimensions=self.dimensions,
            name="assignment",
        )
        if self.time_feature_model is None:
            time_feature_draw = None
        else:
            time_feature_draw = self.time_feature_model.generate(
                self.dimensions, generator
            )
        context = SimulationContext(
            dimensions=self.dimensions,
            treatment=treatment,
            observables=assignment_context.observables,
            unobservables=assignment_context.unobservables,
            feature_metadata=assignment_context.feature_metadata,
            time_features=(
                None if time_feature_draw is None else time_feature_draw.values
            ),
            time_feature_metadata=(
                {} if time_feature_draw is None else time_feature_draw.metadata
            ),
        )
        outcome_draw = self.outcome_model.generate(context, generator)
        untreated = _component_matrix(
            outcome_draw.values,
            dimensions=self.dimensions,
            name="untreated outcome",
        )
        effect_draw = self.effect_model.generate(context, generator)
        effect = _component_matrix(
            effect_draw.values,
            dimensions=self.dimensions,
            name="treatment effect",
        )
        if not np.allclose(effect[treatment == 0.0], 0.0):
            raise ValueError("effect model returned nonzero effects when untreated")

        metadata = {
            "simulator": {
                "name": self.name,
                "dimensions": {
                    "n_units": self.dimensions.n_units,
                    "n_periods": self.dimensions.n_periods,
                },
            },
            "assignment": {
                "model": type(self.assignment).__name__,
                **dict(assignment_draw.metadata),
            },
            "outcome": {
                "model": type(self.outcome_model).__name__,
                **dict(outcome_draw.metadata),
            },
            "effect": {
                "model": type(self.effect_model).__name__,
                **dict(effect_draw.metadata),
            },
        }
        if feature_metadata is not None:
            metadata["features"] = feature_metadata
        if time_feature_draw is not None:
            metadata["time_features"] = {
                "model": type(self.time_feature_model).__name__,
                **dict(time_feature_draw.metadata),
                "values": context.time_features,
            }

        observable_names = assignment_context.feature_metadata.get(
            "observable_names", None
        )
        return PanelDataset(
            outcome=untreated + effect,
            treatment=treatment,
            untreated_outcome=untreated,
            treatment_effect=effect,
            name=self.name,
            metadata=metadata,
            unit_covariates=assignment_context.observables,
            unit_covariate_names=observable_names,
        )

    def iter_simulations(
        self,
        replications: int,
        *,
        seed: int | np.random.SeedSequence | None = None,
    ) -> Iterator[PanelDataset]:
        """Yield independent replications from spawned NumPy seed sequences."""

        if replications <= 0:
            raise ValueError("replications must be positive")
        sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        for child in sequence.spawn(replications):
            yield self.simulate(seed=child)
