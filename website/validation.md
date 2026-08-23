---
title: "Validation and provenance"
description: "What package checks establish and how research translations are classified."
---

## Validation layers

The repository uses four layers of checks.

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

### Downstream tests

The separate `dgps` analysis consumes `pypanelsim` as a dependency. Its test
suite checks that the extracted package still supports the balancing experiment
runner. This does not make its estimators part of `pypanelsim`.

## Standard commands

```bash
uv sync --extra docs
uv run ruff format --check src tests examples website
uv run ruff check src tests examples website
uv run pytest
uv run quarto render website
uv build
```

The exact commands that pass for a release should appear in its release or CI
record. This page describes the required classes of checks; it is not a live CI
status badge.

## Canonical migration parity

Before the earlier embedded implementation was removed, the extraction audit
loaded the old and new Python implementations together. It compared
`outcome`, `untreated_outcome`, `treatment`, and `treatment_effect` with exact
NumPy equality for the same small configuration and seed.

| Design | Parameters | Extraction result |
|---|---|---|
| Classic factor | `overlap=1.0` | exact arrays |
| Weak factor | `overlap=1.0` | exact arrays |
| Synthetic control | `active_share=0.25` | exact arrays |
| Factor-synthetic | `overlap=1.0` | exact arrays |
| Stationary time series | `coefficient=0.9`, `integrated=False` | exact arrays |
| Integrated time series | `coefficient=0.2`, `integrated=True` | exact arrays |
| Mixed factor | `overlap=1.0` | exact arrays |

This was a one-time Python-to-Python extraction check. Current releases rely on
the repository tests and stored fixtures.

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
