---
title: "How simulation works"
description: "See the order of simulation steps, stored matrices, random streams, and extension points."
---

## The simulation sequence

A `PanelSimulator` evaluates components in this order:

1. The unit-feature model draws observed and latent features.
2. The assignment model draws the binary treatment matrix.
3. The time-feature model draws shared time features, if present.
4. The outcome model draws untreated outcomes.
5. The effect model draws potential effects for every cell and realized effects.
6. The simulator validates and freezes the result.

Each step can use results from earlier steps. For example, an outcome model can
use both unit features and treatment timing, which permits selection on
untreated trends. The model definition makes that dependence explicit.

## Three modules to remember

Use imports that state what an object does:

```python
from pypanelsim import core, designs, primitives
```

| Module | Contents | Typical use |
|---|---|---|
| `core` | simulator, results, known effects, random streams | build and inspect simulations |
| `primitives` | feature, assignment, outcome, and effect models | assemble a design from parts |
| `designs` | ready-made scientific examples and benchmarks | run a named design |

Root-level imports also work. The documentation uses module names because
`designs.baker(...)` and `primitives.LowRankFactorOutcome(...)` show where each
object belongs.

## Write a custom component

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

It rejects non-finite values, mismatched shapes, non-binary treatment, a
realized effect outside treated cells, and a broken causal identity.

## Read-only results

`PanelDataset` copies its input arrays and marks them read-only. This prevents
an estimator from changing stored untreated outcomes or effects by accident.
Metadata and annotations are also frozen.

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

## What belongs in this package

The package contains reusable simulation code and ready-made designs. Estimator
implementations, paper builds, experiment schedulers, and saved results stay in
the analysis project. Tutorials can still call external estimators.

As a result, a design still works when an estimator library changes, and
different estimators can receive exactly the same read-only panel.
