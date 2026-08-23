---
title: "Validation and provenance"
description: "What package checks establish and how research translations are classified."
---

## Validation layers

The repository uses three layers of checks.

### Contract tests

Unit tests check component shapes, binary and absorbing treatment, immutable
arrays, effect-surface identities, event-time support, annotations, and random
stream behavior.

### Design tests

Every named design is constructed and simulated. Tests also cover invalid
configurations and public namespace aliases.

### Artifact tests

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

The exact commands that pass for a release should appear in its release or CI
record. This page describes the required classes of checks; it is not a live CI
status badge.

## Cross-language validation

R and NumPy do not share seeded random streams. The [balancing validation
snapshot](vignettes/balancing-reproduction.qmd) therefore compares Monte Carlo
RMSE values relative to their combined Monte Carlo standard errors. It does not
claim matched R and Python draws.

## Provenance labels

Research pages use three labels:

- **Public translation:** a design follows a cited public repository or paper.
- **Working design:** a design adapts an unpublished research specification and
  can change as that work develops.
- **Package benchmark:** maintainers assembled the law to test software or an
  estimator; it is not attributed to a paper.

These labels prevent a constructor name from making a stronger provenance claim
than the evidence supports.

## Sources

- Baker dynamic effects: [JFE_DID](https://github.com/andrewchbaker/JFE_DID)
- TWFE heterogeneity tests: [repository](https://github.com/apoorvalal/TestingInEventStudies) and [paper](https://arxiv.org/abs/2503.05125)
- Large longitudinal experiments: [paper](https://arxiv.org/abs/2410.09952) and [code](https://github.com/py-econometrics/panel-at-scale-code)
- Augmented balancing: [paper](https://apoorvalal.github.io/files/papers/augbal.pdf) and [estimator source](https://github.com/apoorvalal/crabbymetrics)

The adaptive-pooling family is a working design. The composite family is a
package benchmark.
