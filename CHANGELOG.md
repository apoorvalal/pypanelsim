# Changelog

## Unreleased

- Add explicit `TwoWayFixedEffectsOutcome` and generic rank-$k$
  `LowRankFactorOutcome` primitives and use both in the getting-started guide.
- Add correlated and transformed Gaussian features, feature-linear outcomes,
  unit and clustered trends, periodic time effects, and ARMA errors.
- Add fourteen many-cohort designs from the FTestEventStudy Lepskii draft.
- Add ATT-DML conditional-trend, latent-factor, and clustered-ARMA design
  families, with an explicit switch for the baseline-law discrepancy.
- Add the regression-compression main design, its longitudinal Anscombe
  quartet, and the gsynth2 weighted composite.
- Separate full `effect_surface` truth from realized `treatment_effect` and add
  support-aware cohort/event targets through `PanelTruth`.
- Add immutable unit/time annotations, estimator-visible feature transforms,
  and opt-in named component streams through `SimulationSeeds`.
- Add a research-design map, API pages, and an executable four-project
  vignette.

- Allow `effect_model` to be a scalar, unit, time, or unit-by-time callable for
  treatment-effect heterogeneity driven by $X_i$, $U_i$, and shared $V_t$.
- Add shared time-feature models and Gaussian time features to the simulation
  pipeline.
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
- Add generic fixed-size randomized staggered adoption and cohort/event-time
  profile effects.
- Add all seven temporal and seven across-cohort heterogeneous-effect DGPs from
  the F-test paper, with formula-parity tests, API documentation, and an
  executable visualization vignette.
- Expand the executable getting-started guide with rank-2 and rank-3 factor
  DGPs, PyFixest 2WFE event studies, crabbymetrics matrix-completion and IFE
  event paths, visible clustered uncertainty, and ATT comparisons against
  simulation truth. Both factor designs include observation noise so exact
  reconstruction is not built into the comparison.
- Render binary treatment matrices with a discrete untreated/treated color map
  throughout the documentation.
- Add the public `AdditiveFactorOutcome` rank-2 2WFE DGP and
  `SumOutcomeModel` composition component. The getting-started rank-3 example
  now composes the existing canonical `ClassicFactorOutcome` instead of
  defining notebook-local outcome functions.

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
