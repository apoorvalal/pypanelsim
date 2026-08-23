"""Simulation orchestration independent of estimator libraries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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

SeedInput = int | np.random.SeedSequence | None


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


@dataclass(frozen=True, slots=True)
class SimulationSeeds:
    """Named component seeds for stable, independently editable DGP streams.

    Use :meth:`from_seed` to spawn all five streams from one root seed. This
    opt-in contract prevents a new outcome component from changing assignment
    or feature draws. Plain ``simulate(seed=...)`` keeps the legacy shared
    stream for exact backward compatibility.
    """

    features: SeedInput = None
    assignment: SeedInput = None
    time_features: SeedInput = None
    outcome: SeedInput = None
    effect: SeedInput = None

    @classmethod
    def from_seed(cls, seed: int | np.random.SeedSequence) -> SimulationSeeds:
        """Spawn reproducible named streams from one root seed."""

        sequence = (
            seed
            if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        children = sequence.spawn(5)
        return cls(*children)

    def generators(self) -> Mapping[str, np.random.Generator]:
        """Return a fresh generator for each named component."""

        return {
            "features": np.random.default_rng(self.features),
            "assignment": np.random.default_rng(self.assignment),
            "time_features": np.random.default_rng(self.time_features),
            "outcome": np.random.default_rng(self.outcome),
            "effect": np.random.default_rng(self.effect),
        }


def _merge_annotations(
    *sources: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        conflicts = set(merged).intersection(source)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(f"duplicate annotation names: {names}")
        merged.update(source)
    return merged


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


def _component_feature_matrix(
    values: FloatMatrix,
    *,
    n_units: int,
    name: str,
) -> FloatMatrix:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != n_units:
        raise ValueError(f"{name} must have shape (n_units, n_features)")
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
    unit_ids: Sequence[Any] | None = None
    time_ids: Sequence[Any] | None = None
    unit_annotations: Mapping[str, Any] = field(default_factory=dict)
    time_annotations: Mapping[str, Any] = field(default_factory=dict)

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
        streams: SimulationSeeds | None = None,
    ) -> PanelDataset:
        """Draw one panel using an explicit seed or generator."""

        if streams is not None and (seed is not None or rng is not None):
            raise ValueError("provide seed, rng, or streams, not more than one")
        generator = resolve_rng(seed=seed, rng=rng)
        generators = (
            {name: generator for name in (
                "features", "assignment", "time_features", "outcome", "effect"
            )}
            if streams is None
            else streams.generators()
        )
        if self.feature_model is None:
            assignment_context = AssignmentContext(self.dimensions)
            feature_metadata = None
            estimator_observables = assignment_context.observables
            feature_unit_annotations: Mapping[str, Any] = {}
        else:
            feature_draw = self.feature_model.generate(
                self.dimensions, generators["features"]
            )
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
            estimator_observables = (
                assignment_context.observables
                if feature_draw.estimator_observables is None
                else _component_feature_matrix(
                    feature_draw.estimator_observables,
                    n_units=self.dimensions.n_units,
                    name="estimator_observables",
                )
            )
            feature_unit_annotations = feature_draw.unit_annotations
        assignment_draw = self.assignment.assign(
            assignment_context, generators["assignment"]
        )
        treatment = _component_matrix(
            assignment_draw.values,
            dimensions=self.dimensions,
            name="assignment",
        )
        if self.time_feature_model is None:
            time_feature_draw = None
        else:
            time_feature_draw = self.time_feature_model.generate(
                self.dimensions, generators["time_features"]
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
        outcome_draw = self.outcome_model.generate(context, generators["outcome"])
        untreated = _component_matrix(
            outcome_draw.values,
            dimensions=self.dimensions,
            name="untreated outcome",
        )
        effect_draw = self.effect_model.generate(context, generators["effect"])
        effect = _component_matrix(
            effect_draw.values,
            dimensions=self.dimensions,
            name="treatment effect",
        )
        if not np.allclose(effect[treatment == 0.0], 0.0):
            raise ValueError("effect model returned nonzero effects when untreated")
        effect_surface = _component_matrix(
            effect_draw.metadata.get("effect_surface", effect),
            dimensions=self.dimensions,
            name="effect surface",
        )
        if not np.allclose(effect, effect_surface * treatment):
            raise ValueError(
                "effect model values must equal its effect_surface times treatment"
            )

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
        if (
            self.feature_model is not None
            and feature_draw.estimator_observables is not None
        ):
            observable_names = assignment_context.feature_metadata.get(
                "estimator_observable_names", observable_names
            )
        adoption_times = np.full(
            self.dimensions.n_units, self.dimensions.n_periods, dtype=np.int64
        )
        ever_treated = np.any(treatment == 1.0, axis=1)
        if np.all(np.diff(treatment, axis=1) >= 0.0):
            adoption_times[ever_treated] = np.argmax(
                treatment[ever_treated] == 1.0, axis=1
            )
        assignment_annotations: dict[str, Any] = {
            "ever_treated": ever_treated,
            "adoption_period": adoption_times,
        }
        if "assigned_categories" in assignment_draw.metadata:
            assignment_annotations["assignment_category"] = assignment_draw.metadata[
                "assigned_categories"
            ]
        unit_annotations = _merge_annotations(
            self.unit_annotations,
            feature_unit_annotations,
            assignment_annotations,
        )
        time_feature_annotations = (
            {} if time_feature_draw is None else time_feature_draw.time_annotations
        )
        time_annotations = _merge_annotations(
            self.time_annotations,
            time_feature_annotations,
        )
        return PanelDataset(
            outcome=untreated + effect,
            treatment=treatment,
            untreated_outcome=untreated,
            treatment_effect=effect,
            effect_surface=effect_surface,
            name=self.name,
            metadata=metadata,
            unit_covariates=estimator_observables,
            unit_covariate_names=observable_names,
            unit_ids=self.unit_ids,
            time_ids=self.time_ids,
            unit_annotations=unit_annotations,
            time_annotations=time_annotations,
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
