---
title: "Tests and sources"
description: "See what the automated checks cover and where the research designs come from."
---

## Automated checks

The repository checks three things.

### Unit tests

Unit tests check component shapes, binary and absorbing treatment, immutable
arrays, treatment-effect identities, event-time support, annotations, and random
stream behavior.

### Ready-made designs

Every named design is constructed and simulated. Tests also cover invalid
configurations and public imports.

### Built packages

The wheel is built and installed into a fresh environment. A smoke test imports
`core`, `primitives`, and `designs`, then creates and exports a panel.

## Standard commands

```bash
uv sync --extra docs --extra estimators
uv run ruff format --check src tests examples website
uv run ruff check src tests examples website
uv run pytest
uv run quarto render website --execute
uv build
```

Release notes or CI should record the exact commands that passed.

## R and Python comparison

R and NumPy do not share seeded random streams. The [balancing validation
snapshot](vignettes/balancing-reproduction.qmd) therefore compares Monte Carlo
RMSE values relative to their combined Monte Carlo standard errors. It does not
claim matched R and Python draws.

## Design sources

Some designs follow public code or a paper. Others adapt ongoing work, and the
composite design was made only to test the package. Each page states the origin
next to the relevant source.

## Sources

- Baker dynamic effects: [JFE_DID](https://github.com/andrewchbaker/JFE_DID)
- TWFE heterogeneity tests: [repository](https://github.com/apoorvalal/TestingInEventStudies) and [paper](https://arxiv.org/abs/2503.05125)
- Large longitudinal experiments: [paper](https://arxiv.org/abs/2410.09952) and [code](https://github.com/py-econometrics/panel-at-scale-code)
- Augmented balancing: [paper](https://apoorvalal.github.io/files/papers/augbal.pdf) and [estimator source](https://github.com/apoorvalal/crabbymetrics)

The adaptive-pooling family is a working design. The composite family is a
package benchmark.
