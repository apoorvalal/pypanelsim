# pypanelsim

`pypanelsim` is a composable, estimator-neutral Python library for synthetic
panel data. It separates treatment assignment, untreated outcomes, and
treatment effects. Each simulation returns a validated `PanelDataset` with a
small NumPy interchange contract. The package can therefore feed
`crabbymetrics`, another panel-estimator library, or a project-specific
estimator without importing that estimator into the simulation layer.

The package began as an extraction of the data-generating processes used in an
augmented-balancing experiment. The original reproduction remains a downstream
consumer. This repository contains only simulation infrastructure, canonical
designs, tests, examples, and documentation.

## Design principles

- **Estimator neutral.** NumPy is the only runtime dependency. No estimator is
  imported by the package.
- **Composable.** Assignment, untreated outcomes, and treatment effects are
  independent protocols.
- **Explicit randomness.** Every draw accepts either `seed=` or `rng=`. The
  package does not use global random state.
- **Validated output.** All matrices have `(unit, time)` shape. Treatment is
  binary, realized effects are zero when untreated, and observed outcomes must
  equal untreated outcomes plus realized effects.
- **Immutable results.** `PanelDataset` owns read-only copies of its arrays and
  metadata. Downstream code can request writable copies when necessary.
- **Stable canonical factories.** Named designs are available through a
  registry, class-based simulator factories, and concise one-draw functions.

## Requirements and installation

Python 3.10 or later is required.

For development:

```bash
git clone git@github.com:apoorvalal/pypanelsim.git
cd pypanelsim
uv sync
```

As a dependency of another `uv` project:

```bash
uv add "pypanelsim @ git+ssh://git@github.com/apoorvalal/pypanelsim.git"
```

For a sibling checkout during development:

```toml
[project]
dependencies = ["pypanelsim"]

[tool.uv.sources]
pypanelsim = { path = "../pypanelsim", editable = true }
```

The GitHub repository is private. Installation requires an authenticated GitHub
SSH or HTTPS session.

## Quick start

Use a design factory when the simulator will be reused:

```python
import pypanelsim as pps

simulator = pps.classic_factor_design(overlap=1.0)
panel = simulator.simulate(seed=42)

print(panel.shape)  # (200, 50)
print(panel.true_att)  # 1.1
print(panel.name)  # classic_factor
```

Use a convenience function for one draw:

```python
panel = pps.time_series(
    coefficient=0.9,
    integrated=True,
    seed=42,
)
```

Custom dimensions use `CanonicalPanelConfig`:

```python
config = pps.CanonicalPanelConfig(
    n_control=80,
    n_treated=20,
    n_pre=24,
    n_post=6,
    effect_slope=0.1,
    noise_variance=0.25,
)
panel = pps.weak_factor(config=config, overlap=1.0, seed=42)
```

## The data contract

`PanelDataset` stores four same-shaped float matrices.

| Field | Meaning |
|---|---|
| `outcome` | Observed outcome $Y$ |
| `treatment` | Binary treatment matrix $W$ |
| `untreated_outcome` | Untreated potential outcome $Y(0)$ |
| `treatment_effect` | Realized cell effect, zero when $W=0$ |

The class enforces

$$
Y = Y(0) + \tau,
$$

where `treatment_effect` contains $\tau$ only in treated cells. It also stores
unique `unit_ids`, unique `time_ids`, a stable design `name`, and namespaced
component metadata.

Useful properties include:

```python
panel.n_units
panel.n_periods
panel.control_units
panel.treated_units
panel.ever_treated
panel.is_absorbing
panel.adoption_times
panel.true_att
```

### Wide NumPy interchange

Most panel estimators accept two matrices:

```python
y, w = panel.as_arrays()
estimator.fit(y, w)
```

The returned views are read-only. Request copies if an estimator modifies its
inputs:

```python
y, w = panel.as_arrays(copy=True)
```

Select any causal matrices in a fixed order:

```python
y0, tau = panel.arrays(
    ("untreated_outcome", "treatment_effect"),
    copy=False,
)
```

### Long-column interchange

`as_long_dict()` returns flat columns without adding pandas or Polars as a
dependency:

```python
columns = panel.as_long_dict()

# Optional downstream conversions
# pandas.DataFrame(columns)
# polars.DataFrame(columns)
# pyarrow.table(columns)
```

The columns are `unit`, `time`, `outcome`, `treatment`,
`untreated_outcome`, and `treatment_effect`.

## Canonical designs

The default canonical panel has 160 never-treated units, 40 treated units, 40
pre-treatment periods, and 10 post-treatment periods. The last 40 units adopt
treatment in period 40. The effect increases by 0.2 per post-treatment period,
so the true ATT is 1.1.

| Registry name | Factory | Main parameters | Untreated process |
|---|---|---|---|
| `classic_factor` | `classic_factor_design()` | `overlap` | Two drifting factors |
| `weak_factor` | `weak_factor_design()` | `overlap` | Five weak drift and five weak cyclical factors |
| `synthetic_control` | `synthetic_control_design()` | `active_share` | Sparse convex donor signal |
| `factor_synthetic` | `factor_synthetic_design()` | `overlap` | Equal factor and donor mixture |
| `time_series` | `time_series_design()` | `coefficient`, `integrated` | AR(1) or ARIMA(1,1,0) unit paths |
| `mixed_factor` | `mixed_factor_design()` | `overlap` | Drift-factor and cyclical-factor half-panels |

For the factor designs, `overlap=0` gives overlapping treated and control
loading distributions. `overlap=1` separates their means. For the synthetic
control design, `active_share` controls donor sparsity. For the time-series
design, `integrated=True` cumulatively sums the stationary AR increments.

The exact probability laws and draw order are in
[`docs/canonical-designs.md`](docs/canonical-designs.md).

## Registry API

Create a canonical simulator by stable name:

```python
print(pps.available_canonical_designs())

simulator = pps.make_canonical(
    "synthetic_control",
    active_share=0.25,
)
panel = simulator.simulate(seed=7)
```

Projects can maintain their own registry:

```python
registry = pps.DGPRegistry()
registry.register("my_design", my_simulator_factory)
simulator = registry.create("my_design", scale=2.0)
```

Registration rejects invalid names and accidental replacement. Pass
`replace=True` only when replacement is intentional.

## Compose a new DGP

A `PanelSimulator` requires four objects:

1. `PanelDimensions` defines the rectangular shape.
2. An `AssignmentModel` produces binary treatment.
3. An `OutcomeModel` produces untreated outcomes.
4. An `EffectModel` produces realized treatment effects.

The interfaces use structural typing. A custom component does not need to
inherit from a package base class. It only needs the documented method.

```python
from dataclasses import dataclass

import numpy as np
import pypanelsim as pps


@dataclass(frozen=True)
class UnitTrendOutcome:
    trend_scale: float = 0.1
    noise_scale: float = 1.0

    def generate(self, context, rng):
        unit_intercepts = rng.normal(size=(context.dimensions.n_units, 1))
        unit_slopes = rng.normal(
            scale=self.trend_scale,
            size=(context.dimensions.n_units, 1),
        )
        time = np.arange(context.dimensions.n_periods)[None, :]
        noise = rng.normal(
            scale=self.noise_scale,
            size=(context.dimensions.n_units, context.dimensions.n_periods),
        )
        values = unit_intercepts + unit_slopes * time + noise
        return pps.ComponentDraw(values, {"trend_scale": self.trend_scale})


simulator = pps.PanelSimulator(
    name="unit_trends",
    dimensions=pps.PanelDimensions(n_units=100, n_periods=30),
    assignment=pps.StaggeredAdoption({80: 20, 81: 21, 82: 22}),
    outcome_model=UnitTrendOutcome(),
    effect_model=pps.LinearRampEffect(slope=0.25),
)

panel = simulator.simulate(seed=123)
```

Plain functions can be wrapped with `CallableOutcomeModel`. Built-in
assignment components are `SingleCohortAssignment` and `StaggeredAdoption`.
Built-in effects are `ConstantEffect` and `LinearRampEffect`.

See [`docs/architecture.md`](docs/architecture.md) and
[`examples/custom_dgp.py`](examples/custom_dgp.py) for the extension contract.

## Downstream estimator integration

`pypanelsim` does not define an estimator interface. It produces standard
arrays and lets the estimator own its API.

For crabbymetrics:

```python
import crabbymetrics as cm
import pypanelsim as pps

panel = pps.classic_factor(overlap=1.0, seed=42)

model = cm.AugmentedBalancing(balance="double")
model.fit(panel.outcome, panel.treatment)

estimate = float(model.summary()["att"])
error = estimate - panel.true_att
```

An outcome-model matrix can be passed without changing the simulation object:

```python
model.fit(
    panel.outcome,
    panel.treatment,
    estimated_untreated_surface,
)
```

This separation is deliberate. Simulation truth remains in `PanelDataset`;
estimation state remains in the downstream library.

## Reproducibility and repeated draws

One draw accepts either an integer seed, a NumPy `SeedSequence`, or an existing
`numpy.random.Generator`:

```python
panel_a = simulator.simulate(seed=10)
panel_b = simulator.simulate(seed=10)
np.testing.assert_array_equal(panel_a.outcome, panel_b.outcome)
```

Providing both `seed` and `rng` is an error. Passing an existing generator
advances that generator.

Use spawned seed sequences for independent Monte Carlo replications:

```python
for replication, panel in enumerate(
    simulator.iter_simulations(200, seed=20260819),
    start=1,
):
    estimate = fit_estimator(*panel.as_arrays())
```

The canonical NumPy implementation preserves the seeded draw order of the
extracted `panel_dgps` code. NumPy and R use different random streams, so an
integer seed does not produce the same realized panel across languages.

## Metadata

Metadata is separated by simulation component:

```python
panel.metadata["simulator"]
panel.metadata["assignment"]
panel.metadata["outcome"]
panel.metadata["effect"]
```

Canonical factor designs store factors and unit-aligned loadings under
`metadata["outcome"]`. Sparse-donor designs also store active donor positions
and both unit-aligned and control-only donor weights. Time-series designs store
the latent process before observation noise.

Metadata is diagnostic truth, not an estimator input. It is recursively frozen
and copied when the dataset is created.

## Fail-fast validation

The package raises `ValueError` for malformed simulations, including:

- nonpositive dimensions;
- invalid treatment positions or adoption periods;
- nonbinary treatment;
- mismatched component shapes;
- nonfinite outcomes or effects;
- effects outside treated cells;
- inconsistent observed and untreated outcomes;
- duplicate unit or time identifiers;
- ambiguous `seed` and `rng` arguments.

## Migration from `panel_dgps`

The one-draw functions keep their prior names and seeded numerical behavior.
The main changes are the package name, the primary class names, and namespaced
metadata.

```python
# Before
from panel_dgps import PanelConfig, classic_factor

# After
from pypanelsim import CanonicalPanelConfig, classic_factor
```

`PanelConfig` and `PanelData` remain available as compatibility aliases for
`CanonicalPanelConfig` and `PanelDataset`. New code should use the explicit
names. See [`docs/migration.md`](docs/migration.md) for the full map.

## Repository layout

```text
pypanelsim/
├── src/pypanelsim/
│   ├── data.py          # PanelDataset and interchange
│   ├── components.py    # component protocols and built-ins
│   ├── simulator.py     # orchestration and RNG policy
│   ├── registry.py      # named simulator factories
│   └── canonical.py     # canonical experiment designs
├── tests/               # contract, invariant, and regression tests
├── examples/            # runnable composition and interop examples
└── docs/                # architecture, formulas, and migration
```

## Development

All development commands use `uv`:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

Run the examples with:

```bash
uv run python examples/custom_dgp.py
uv run python examples/estimator_interop.py
```

The test suite covers immutable data contracts, assignment and effect
components, custom outcome composition, registry behavior, canonical DGP
invariants, seeded reproducibility, and downstream array interchange. The
separate migration audit verifies exact cell-by-cell parity with the extracted
implementation. See [`docs/validation.md`](docs/validation.md).

## Scope

`pypanelsim` generates balanced rectangular panels with binary treatment. It
does not fit estimators, provide inference, manage experiment results, or render
reports. Those tasks belong in downstream projects. New assignment and outcome
components can support multiple cohorts and arbitrary treated-unit order while
preserving the same dataset contract.
