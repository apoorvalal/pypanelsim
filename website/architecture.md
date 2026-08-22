# Architecture and extension contract

## One-way simulation pipeline

`PanelSimulator` evaluates components in a fixed order:

```text
PanelDimensions
      |
      v
UnitFeatureModel.generate(dimensions, rng) [optional]
      |
      v
AssignmentContext(dimensions, observables, unobservables)
      |
      v
AssignmentModel.assign(context, rng)
      |
      v
TimeFeatureModel.generate(dimensions, rng) [optional]
      |
      v
SimulationContext(dimensions, treatment, unit features, time features)
      |
      +-------------------------+
      |                         |
      v                         v
OutcomeModel.generate()   EffectModel.generate()
      |                         |
      +------------+------------+
                   |
                   v
              PanelDataset
```

Shared unit features are drawn before assignment and can reach assignment,
outcomes, and effects. Time features are drawn after assignment and reach
outcomes and effects, so adding $V_t$ does not perturb a randomized assignment
stream. This permits $\tau_{it}=f(X_i,U_i,V_t)$ without passing arrays through
diagnostic metadata. Assignment still precedes the outcome draw, and the
untreated outcome model does not receive realized effects. The effect model
does not receive outcomes. This keeps the causal decomposition explicit.

## Core objects

### `PanelDimensions`

`PanelDimensions(n_units, n_periods)` defines only shape. It does not assume a
control-first unit order, one treatment cohort, or a specific intervention
date.

### `ComponentDraw`

Every component returns `ComponentDraw(values, metadata)`. `values` must be a
matrix with the configured shape. Metadata can contain diagnostics or latent
simulation truth. The final dataset copies and recursively freezes metadata.

### `AssignmentContext`

The assignment context contains dimensions, observed unit covariates, and
latent unit features. Its matrices have shape `(n_units, n_features)`. The
feature model is optional, so legacy and canonical assignments receive empty
feature matrices without consuming additional random draws.

### `SimulationContext`

The context contains dimensions, realized treatment, the unit features used by
assignment, and a `(n_periods, n_time_features)` `time_features` matrix. It
derives:

- `ever_treated`;
- `control_units`;
- `treated_units`;
- `is_absorbing`;
- `adoption_times` for absorbing treatment.

Outcome models can therefore depend on treatment status without assuming that
controls appear before treated units.

### `PanelDataset`

The dataset owns the final arrays. It validates shape, finiteness, binary
treatment, effect support, and the causal decomposition. Its read-only arrays
are safe to share among estimators. `copy=True` creates writable estimator
inputs when required. Observed time-invariant covariates are available through
`unit_covariates`; latent features remain namespaced simulation metadata.

## Protocols

The component interfaces use `typing.Protocol`. Inheritance is optional.

```python
class UnitFeatureModel(Protocol):
    def generate(self, dimensions, rng) -> UnitFeatureDraw: ...


class TimeFeatureModel(Protocol):
    def generate(self, dimensions, rng) -> TimeFeatureDraw: ...


class AssignmentModel(Protocol):
    def assign(self, context, rng) -> ComponentDraw: ...


class OutcomeModel(Protocol):
    def generate(self, context, rng) -> ComponentDraw: ...


class EffectModel(Protocol):
    def generate(self, context, rng) -> ComponentDraw: ...
```

The package validates the returned matrices at the composition boundary. A
custom class can focus on its probability law.

## Random-number policy

`PanelSimulator.simulate()` resolves one `numpy.random.Generator`. Components
receive the same generator in pipeline order. A deterministic component should
not consume random draws.

Accepted inputs are:

- `seed=<integer>`;
- `seed=<numpy.random.SeedSequence>`;
- `rng=<numpy.random.Generator>`;
- neither, for a fresh entropy-seeded generator.

Passing both is an error. `iter_simulations()` uses `SeedSequence.spawn()` to
create independent child streams instead of using adjacent integer seeds.

When a design needs two components to restart from the same state, it must clone
the generator explicitly and document that choice. The canonical mixed-factor
design does this because it reproduces two half-panels that restart from the
same random stream.

## Built-in composition components

### Assignment

- `SingleCohortAssignment` creates one absorbing cohort. It supports explicit
  treated-unit positions.
- `StaggeredAdoption` accepts a mapping from unit position to adoption period.
- `RandomizedSingleCohortAssignment` samples a fixed-size cohort without
  replacement.
- `BinaryLogitAssignment` draws treatment from observed and latent unit-feature
  logits.
- `GeneralizedPropensityAssignment` draws adoption cohorts from multinomial
  logits with never treated as the baseline category.

### Unit features

- `GaussianUnitFeatures` draws independent standard-normal observed covariates
  and latent unit factors. Custom feature models can impose correlations or
  generate application-specific covariates.

### Time features

- `GaussianTimeFeatures` draws independent standard-normal $V_t$. Custom time
  feature models can generate trends, cycles, shocks, or observed calendars.

### Effects

- `ConstantEffect` applies one effect to every treated cell.
- `LinearRampEffect` uses unit-specific event time and supports staggered
  adoption.
- `CallableEffect` evaluates scalar, unit, time, or full unit-by-time laws
  against the shared `SimulationContext`; `PanelSimulator` automatically wraps
  a lambda passed as `effect_model`. `CallableUnitEffect` is retained as a
  compatibility name.

### Outcome adapter

`CallableOutcomeModel` turns a function into an outcome component. The function
can return a matrix or a `ComponentDraw` with metadata.

## Add a component

1. Define an immutable configuration object, normally a frozen dataclass.
2. Validate scalar parameters in `__post_init__`.
3. Implement the relevant protocol method.
4. Use only the provided generator for random draws.
5. Return a full `(n_units, n_periods)` matrix.
6. Put diagnostic truth in metadata, not global state.
7. Add contract, invalid-input, and seeded-reproducibility tests.

An outcome model should use `context.control_units` and
`context.treated_units` rather than assume a row order. An effect model must
return zero outside treated cells.

## Add a named DGP

Write a factory that returns a configured `PanelSimulator`:

```python
def unit_trend_design(*, scale=1.0):
    return PanelSimulator(
        name="unit_trend",
        dimensions=PanelDimensions(100, 30),
        assignment=SingleCohortAssignment(20, 20),
        outcome_model=UnitTrendOutcome(scale=scale),
        effect_model=ConstantEffect(1.0),
    )
```

Register it in a project registry:

```python
registry = DGPRegistry()
registry.register("unit_trend", unit_trend_design)
```

Do not add estimator options to a simulation factory. Estimator configuration
belongs in the downstream experiment runner.

## Data ownership

The dataset copies arrays before setting them read-only. Mutating an input array
after construction cannot change the dataset. Nested mappings, arrays, lists,
and sets in metadata are also copied or converted to immutable forms.

The package does not promise that arbitrary scalar objects inside metadata are
deeply immutable. Store simple scalars, arrays, mappings, and sequences.

## Extension limits

The current `PanelDataset` contract requires a balanced rectangular panel and
binary treatment. Missing outcomes can be represented by a separate custom
metadata mask, but the core outcome matrices must remain finite. A future
missing-data extension should add an explicit observation mask instead of
using `NaN` as an implicit contract.
