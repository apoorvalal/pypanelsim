# Migration from `panel_dgps`

## Package boundary

The former `dgps` project mixed simulation code with crabbymetrics outcome
models, Monte Carlo runners, cached results, and a Quarto report. `pypanelsim`
contains only simulation code and simulation documentation. The reproduction
project now depends on this repository as a normal downstream package.

## Import map

| Previous | Current |
|---|---|
| `panel_dgps.PanelConfig` | `pypanelsim.CanonicalPanelConfig` |
| `panel_dgps.PanelData` | `pypanelsim.PanelDataset` |
| `panel_dgps.classic_factor` | `pypanelsim.classic_factor` |
| `panel_dgps.weak_factor` | `pypanelsim.weak_factor` |
| `panel_dgps.synthetic_control` | `pypanelsim.synthetic_control` |
| `panel_dgps.factor_synthetic` | `pypanelsim.factor_synthetic` |
| `panel_dgps.time_series` | `pypanelsim.time_series` |
| `panel_dgps.mixed_factor` | `pypanelsim.mixed_factor` |

`PanelConfig` and `PanelData` are compatibility aliases. New code should use
the explicit current names.

## One-draw migration

```python
# Previous
from panel_dgps import PanelConfig, classic_factor

config = PanelConfig(n_control=80, n_treated=20, n_pre=24, n_post=6)
panel = classic_factor(config=config, overlap=1.0, seed=10)
```

```python
# Current
from pypanelsim import CanonicalPanelConfig, classic_factor

config = CanonicalPanelConfig(
    n_control=80,
    n_treated=20,
    n_pre=24,
    n_post=6,
)
panel = classic_factor(config=config, overlap=1.0, seed=10)
```

The resulting outcome, treatment, untreated-outcome, and effect matrices are
numerically identical for the same seed and arguments.

## Reusable simulator migration

Previous code configured arguments on each function call. Current code can
create one reusable simulator:

```python
simulator = classic_factor_design(config=config, overlap=1.0)

for panel in simulator.iter_simulations(200, seed=10):
    estimate = estimator.fit(*panel.as_arrays())
```

This uses spawned child streams and avoids coupling replication identifiers to
adjacent integer seeds.

## Metadata migration

Previous canonical metadata was flat:

```python
panel.metadata["factors"]
panel.metadata["loadings"]
```

Current metadata is namespaced by component:

```python
panel.metadata["outcome"]["factors"]
panel.metadata["outcome"]["loadings"]
panel.metadata["assignment"]["treated_units"]
panel.metadata["effect"]["slope"]
```

Sparse-donor `donor_weights` now has one entry per panel unit and is aligned to
the panel row order. `control_donor_weights` contains the prior control-only
vector.

## Data ownership

The previous container set the write flag on supplied arrays. `PanelDataset`
instead owns copies before making them read-only. Code that intentionally
mutates estimator inputs must request copies:

```python
y, w = panel.as_arrays(copy=True)
```

## New composition API

Custom designs no longer need to edit a canonical generator module. Implement
an assignment, outcome, or effect protocol and compose it with
`PanelSimulator`. Project-specific named designs can be stored in a
`DGPRegistry` without changing the package registry.
