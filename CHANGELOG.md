# Changelog

## Unreleased

- Allow `effect_model` to be a unit-level callable for treatment-effect
  heterogeneity driven by observed covariates and latent factors.
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
- Add an executable catalog with construction code and side-by-side outcome and
  treatment visualizations for every canonical balancing DGP.
- Add a self-contained Baker staggered-adoption DGP with randomized state
  cohorts and dynamic cohort-specific treatment effects.
- Add an executable PyFixest vignette showing vanilla TWFE contamination and
  recovery with a saturated Sun--Abraham-style cohort/event-time estimator.

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
