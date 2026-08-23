---
title: "Architecture and data contract"
description: "The component sequence, public namespaces, validation rules, and package boundary."
---

## The simulation sequence

A `PanelSimulator` evaluates components in this order:

1. The unit-feature model draws observed and latent features.
2. The assignment model draws the binary treatment matrix.
3. The time-feature model draws shared time features, if present.
4. The outcome model draws untreated outcomes.
5. The effect model draws a complete effect surface and realized effects.
6. The simulator validates and freezes the result.

Later components can use earlier draws through typed contexts. For example, an
outcome model can use both unit features and treatment timing. This permits
selection on untreated trends. The package keeps that dependence visible in
the component definition.

## Three public namespaces

Use imports that state what an object does:

```python
from pypanelsim import core, designs, primitives
```

| Namespace | Contents | Typical use |
|---|---|---|
| `core` | contracts, simulator, data, truth, random streams | compose and inspect simulations |
| `primitives` | feature, assignment, outcome, and effect laws | build a DGP from parts |
| `designs` | configured scientific and benchmark families | run a named design |

Root-level imports remain available for compatibility. New examples do not use
them because a call such as `designs.baker(...)` or
`primitives.LowRankFactorOutcome(...)` carries useful meaning.

## Core protocols

Each component satisfies a small protocol:

- `UnitFeatureModel.generate(dimensions, rng)` returns a `UnitFeatureDraw`.
- `AssignmentModel.assign(context, rng)` returns a `ComponentDraw`.
- `TimeFeatureModel.generate(dimensions, rng)` returns a `TimeFeatureDraw`.
- `OutcomeModel.generate(context, rng)` returns untreated outcomes.
- `EffectModel.generate(context, rng)` returns realized effects.

The protocols are structural. A custom dataclass does not need to inherit from
a package base class. It only needs the correct method and return type.

## Matrix conventions

All panel matrices use `(unit, time)` order. For $N$ units and $T$ periods,
each matrix has shape $N \times T$.

| Matrix | Definition |
|---|---|
| `outcome` | observed outcome $Y$ |
| `treatment` | binary assignment $D$ |
| `untreated_outcome` | untreated potential outcome $Y(0)$ |
| `effect_surface` | effect if treated, $\tau$ |
| `treatment_effect` | realized effect $D \odot \tau$ |

The simulator enforces

$$
Y = Y(0) + D \odot \tau.
$$

It rejects non-finite values, incompatible shapes, non-binary treatment, a
realized effect outside treated cells, and a broken causal identity.

## Immutable results

`PanelDataset` copies its input arrays and marks them read-only. This prevents
an estimator from changing the stored causal truth by accident. Metadata and
annotations are also frozen.

The result supplies:

- `as_arrays()` for matrix estimators;
- `as_long_dict()` for data-frame and formula estimators;
- `true_att` for the average realized treated-cell effect;
- `truth.cohort_event()` for supported cohort-by-event-time cells;
- `truth.event_study()` for event-time aggregates;
- `truth.att_by_cohort()` for cohort ATT values.

## Randomness

`simulate(seed=42)` uses one shared NumPy generator. This is convenient and
preserves earlier package behavior.

`SimulationSeeds.from_seed(42)` creates separate feature, assignment,
time-feature, outcome, and effect streams. Use it when adding a new outcome
component must not change the assignment draw.

```python
from pypanelsim import core, designs

simulator = designs.baker_design()
panel = simulator.simulate(streams=core.SimulationSeeds.from_seed(42))
```

Never set NumPy's process-wide random seed for package simulations.

## Package boundary

The package contains reusable simulation infrastructure and configured data
laws. It does not contain estimator implementations, paper build systems,
Monte Carlo schedulers, or cached reports. Research tutorials can call an
external estimator, but the estimator does not become part of the core API.

This boundary has two benefits. A DGP remains usable after an estimator library
changes, and different estimators can receive exactly the same immutable panel.
