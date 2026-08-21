# Changelog

## Unreleased

- Add shared observed and latent unit features to the simulation pipeline.
- Add fixed-size randomized and binary-logit single-cohort assignments.
- Add multinomial-logit generalized propensity scores for adoption cohorts.
- Expose observed unit covariates on `PanelDataset` and long interchange.
- Add Matplotlib and a rendered comparison of assignment DGP outcome and
  treatment matrices.
- Add a Quarto documentation website with detailed public API reference pages.
- Add optional docs dependencies for crabbymetrics, PyFixest, Jupyter, and
  pandas.
- Add the complete balancing-reproduction numerical experiment as an executable
  vignette with cached 398-cell results, parity fixtures, and runnable support
  modules.

## 0.1.0 - 2026-08-19

- Extract the panel simulation layer from the balancing reproduction project.
- Add estimator-neutral `PanelDataset` wide and long interchange contracts.
- Add composable assignment, outcome, and effect protocols.
- Add single-cohort and staggered assignment components.
- Add constant and event-time ramp effect components.
- Add canonical factor, weak-factor, synthetic-control, factor-synthetic,
  time-series, and mixed-factor designs.
- Preserve exact seeded NumPy output from the extracted canonical generators.
- Add a named simulator registry, repeated-draw seed spawning, tests, examples,
  and architecture and migration documentation.
