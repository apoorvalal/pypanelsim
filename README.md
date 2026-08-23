# pypanelsim

`pypanelsim` builds synthetic panel data with explicit treatment assignment,
untreated outcomes, treatment effects, and causal truth. It is an
estimator-neutral simulation package: use the same generated panel with matrix,
formula, pandas, Polars, Arrow, R, or custom estimators.

Every simulation answers four separate questions:

1. Which features exist?
2. Who receives treatment, and when?
3. What would outcomes be without treatment?
4. How does treatment change those outcomes?

This separation makes simulation assumptions visible and lets you change one
part of a design without rewriting the others.

## Install

The package requires Python 3.10 or later.

```bash
uv add "pypanelsim @ git+https://github.com/apoorvalal/pypanelsim.git"
```

To work on the repository:

```bash
git clone https://github.com/apoorvalal/pypanelsim.git
cd pypanelsim
uv sync --extra docs
```

## Public namespaces

The public API has three descriptive namespaces:

- `pypanelsim.core` contains the simulator, data container, random streams, and
  component protocols.
- `pypanelsim.primitives` contains reusable feature, assignment, outcome, and
  effect laws.
- `pypanelsim.designs` contains configured design families and direct simulation
  functions.

Old root-level imports remain available for compatibility. New code should use
the descriptive namespaces.

## Simulate a configured design

```python
from pypanelsim import designs

panel = designs.classic_factor(overlap=1.0, seed=42)

print(panel.shape)
print(panel.true_att)
print(panel.outcome.shape, panel.treatment.shape)
```

A function such as `designs.classic_factor(...)` returns a simulated
`PanelDataset`. Its matching `designs.classic_factor_design(...)` function
returns a reusable `PanelSimulator`.

```python
simulator = designs.classic_factor_design(overlap=1.0)
panels = list(simulator.iter_simulations(100, seed=42))
```

## Compose a design from primitives

This example combines two-way fixed effects and a low-rank factor structure in
the untreated outcome:

```python
from pypanelsim import core, primitives

simulator = core.PanelSimulator(
    name="fixed_effects_plus_factors",
    dimensions=core.PanelDimensions(n_units=200, n_periods=40),
    feature_model=primitives.GaussianUnitFeatures(
        n_observables=2,
        n_unobservables=1,
    ),
    assignment=primitives.RandomizedSingleCohortAssignment(
        n_treated=80,
        adoption_period=25,
    ),
    outcome_model=primitives.SumOutcomeModel(
        models=(
            primitives.TwoWayFixedEffectsOutcome(
                unit_effect_scale=1.0,
                time_effect_scale=0.5,
                noise_scale=0.2,
            ),
            primitives.LowRankFactorOutcome(rank=2, factor_scale=0.7),
        )
    ),
    effect_model=primitives.LinearRampEffect(slope=0.2),
)

panel = simulator.simulate(seed=42)
```

## Data contract

All panel matrices have shape `(n_units, n_periods)`.

| Field | Meaning |
|---|---|
| `outcome` | Observed outcome, $Y_{it}$ |
| `treatment` | Binary treatment state, $D_{it}$ |
| `untreated_outcome` | Untreated potential outcome, $Y_{it}(0)$ |
| `effect_surface` | Cell-level effect if treated, $\tau_{it}$ |
| `treatment_effect` | Realized effect, $D_{it}\tau_{it}$ |

The container validates

$$
Y_{it} = Y_{it}(0) + D_{it}\tau_{it}.
$$

It also provides cohort and event-time truth, unit and time identifiers,
annotations, covariates, and conversion helpers:

```python
long_columns = panel.as_long_dict(include_annotations=True)

import pandas as pd

frame = pd.DataFrame(long_columns)
truth = panel.truth.event_study()
```

Pandas and Polars are optional. The NumPy arrays and long dictionary have no
data-frame dependency.

## Design families

| Scientific question | Constructors |
|---|---|
| Canonical panel prediction | `classic_factor`, `weak_factor`, `synthetic_control`, `time_series`, `mixed_factor` |
| TWFE under dynamic cohort effects | `baker` |
| Event-study heterogeneity tests | `ftest_temporal`, `ftest_cohort` |
| Adaptive pooling across cohorts | `lepskii` |
| Conditional or latent trend adjustment | `att_dml` |
| Large-panel regression summaries | `regression_compression`, `anscombe` |
| Composite estimator stress test | `gsynth_composite` |

Use `available_*_designs()` functions in `pypanelsim.designs` to list named
variants.

The `gsynth_composite` family is a package benchmark assembled from reusable
components. It is not presented as a reproduction of a published design.

## Reproducible random streams

`simulate(seed=...)` uses one shared NumPy stream. Use named streams when you
want changes in one component to leave the other component draws unchanged:

```python
from pypanelsim import core, designs

simulator = designs.lepskii_design("tensor_blocks")
panel = simulator.simulate(streams=core.SimulationSeeds.from_seed(42))
```

## Scope

`pypanelsim` owns simulation components, validated data, truth, and export
adapters. It does not own estimators, Monte Carlo runners, result caches, or
paper-specific reports. Keeping these concerns outside the package prevents an
estimator dependency from defining the simulation interface.

## Scientific sources

Several configured families translate public research designs:

- Baker's dynamic-treatment example: [JFE_DID](https://github.com/andrewchbaker/JFE_DID)
- TWFE heterogeneity tests: [TestingInEventStudies](https://github.com/apoorvalal/TestingInEventStudies) and [the paper](https://arxiv.org/abs/2503.05125)
- Large-scale longitudinal experiments: [paper](https://arxiv.org/abs/2410.09952) and [code](https://github.com/py-econometrics/panel-at-scale-code)
- Augmented balancing estimators: [paper](https://apoorvalal.github.io/files/papers/augbal.pdf) and [crabbymetrics](https://github.com/apoorvalal/crabbymetrics)

The documentation states where a family follows a public source, adapts a
working design, or serves only as a package benchmark.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run quarto render website
uv build
```

See the [documentation site](https://apoorvalal.github.io/pypanelsim) for the
first tutorial, design catalog, research tutorials, and API reference.

## License

MIT
