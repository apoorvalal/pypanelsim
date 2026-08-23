---
title: "Migration guide"
description: "Move from panel_dgps or flat pypanelsim imports to the documented namespaces."
---

## From flat imports to descriptive namespaces

Root-level imports still work, but new code should state each object's role.

```python
# Compatibility form
from pypanelsim import PanelSimulator, classic_factor

# Preferred form
from pypanelsim import core, designs

simulator_type = core.PanelSimulator
panel = designs.classic_factor(seed=42)
```

The change is organizational. It does not change generated arrays.

## From `panel_dgps`

| Previous | Current |
|---|---|
| `panel_dgps.PanelConfig` | `pypanelsim.designs.CanonicalPanelConfig` |
| `panel_dgps.PanelData` | `pypanelsim.core.PanelDataset` |
| `panel_dgps.classic_factor` | `pypanelsim.designs.classic_factor` |
| `panel_dgps.weak_factor` | `pypanelsim.designs.weak_factor` |
| `panel_dgps.synthetic_control` | `pypanelsim.designs.synthetic_control` |
| `panel_dgps.factor_synthetic` | `pypanelsim.designs.factor_synthetic` |
| `panel_dgps.time_series` | `pypanelsim.designs.time_series` |
| `panel_dgps.mixed_factor` | `pypanelsim.designs.mixed_factor` |

`PanelConfig` and `PanelData` remain compatibility aliases at the package root.

## One-draw migration

```python
# Previous
from panel_dgps import PanelConfig, classic_factor

config = PanelConfig(n_control=80, n_treated=20, n_pre=24, n_post=6)
panel = classic_factor(config=config, overlap=1.0, seed=10)
```

```python
# Current
from pypanelsim import designs

config = designs.CanonicalPanelConfig(
    n_control=80,
    n_treated=20,
    n_pre=24,
    n_post=6,
)
panel = designs.classic_factor(
    config=config,
    overlap=1.0,
    seed=10,
)
```

The extraction audit found exact equality for the canonical outcome, treatment,
untreated-outcome, and realized-effect arrays when arguments and seed matched.

## Reusable simulators

Use a `_design` constructor when an experiment needs many draws:

```python
simulator = designs.classic_factor_design(
    config=config,
    overlap=1.0,
)

for panel in simulator.iter_simulations(200, seed=10):
    outcome, treatment = panel.as_arrays()
```

`iter_simulations` uses child seed sequences. It does not rely on adjacent
integer seeds.

## Metadata layout

Earlier canonical metadata was flat. Current metadata is namespaced by
component:

```python
panel.metadata["outcome"]
panel.metadata["assignment"]
panel.metadata["effect"]
```

Feature and time-feature metadata appear only when those components exist.

## Data ownership

`PanelDataset` owns read-only copies. Code that intentionally mutates estimator
inputs must request copies:

```python
outcome, treatment = panel.as_arrays(copy=True)
```

## Custom designs

Do not edit a canonical generator to add a project-specific law. Compose
`core.PanelSimulator` from `primitives`, or register a local factory in
`core.DGPRegistry`. This keeps the package API small and makes the scientific
assumptions explicit at the call site.
